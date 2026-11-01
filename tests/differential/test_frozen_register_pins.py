"""7j's frozen register keeps the geometry kinds' versions it froze (those registered on 2026-10-31, the screw 0.2.0
among them) and the body dynamics' (``rigid_inertial`` 0.2.0) until a dated amendment moves them, so that nothing
re-run from its scripts changes silently. Quasi-uniform sampling (7m-a) moved the screw to 0.3.0, and script A's MRS
arm built unpinned moved by +15.6% at point A. The gyroscopic term made implicit (``rigid_inertial`` 0.3.0,
MIME-ANO-129) moved the register's x4.32 transverse-inertia arm from 3.20686 to 3.20747 mm/s (+0.019%), inside a
version, until the law was given its own.

- **Every experiment the register's scripts build** (``scripts/sim2real/``: ``operating_point.experiment`` at each
  rung and surface, which every other script builds through) names each geometry kind, its structures' and its body's
  parts', at the version ``operating_point.FROZEN_VERSIONS`` names, read from the document a save writes; a kind the
  table does not name fails.
- **The table is the register's**: it equals the versions registered when the register froze, stated here apart from
  the scripts, and each of them is registered (the current version or one ``mime.scene.KEPT`` keeps).
- **No other script builds a geometry, a body's dynamics or an experiment of its own**, so the first check covers what
  they run; and in ``operating_point.py`` each geometry built is passed through ``pinned`` and each body's dynamics
  through ``pinned_dynamics``.
- **The dynamics are pinned**: every experiment's document names ``rigid_inertial`` at ``FROZEN_DYNAMICS``' version, the
  table equals the version registered when the register froze, and that version is registered
  (``mime.dynamics.body.KEPT``).
  The transverse-inertia arm (inertia x (1, 4.32, 4.32)) reproduces the register's 3.20686 mm/s to the digits the
  register states, and the same arm on 0.3.0 does not.
- **The surrogate rung does not see the pin**: its arm, built on the screw 0.2.0, is script A's reproduction arm
  (``tests/script_a.py``, the screw 0.3.0) node for node, edge for edge and leaf for leaf, bitwise
  (``same_as_script_a``): script A's and A0's preregistered comparisons run that rung and read no sampling, so they
  build the current kinds and nothing of theirs moved.
- **The interactions too** (phase 7m-b): every law ``operating_point.experiment`` attaches names the version
  ``operating_point.FROZEN_INTERACTIONS`` names, read from the document, and that table is the versions registered
  when the register froze, each registered (the current version or one ``mime.interactions.KEPT`` keeps): drag 0.4.0
  separated the MRS rung's quadrature, and the register's MRS arm keeps 0.3.0's point forces.
- **Controls**, failing on every run: the experiment with the pin bypassed names the screw 0.3.0 and fails the first
  check; a script that builds ``fl9()`` unpinned fails the source scan; the experiment with the law's pin bypassed
  names drag 0.4.0; the experiment with the dynamics' pin bypassed names ``rigid_inertial`` 0.3.0, fails the dynamics
  check, and its transverse-inertia arm runs at 3.20747 mm/s.

Nothing here builds a graph or solves; the register's numbers are its scripts' to reproduce."""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

from mime.dynamics.body import KEPT as KEPT_DYNAMICS
from mime.dynamics.body import RigidInertial
from mime.interactions import KEPT as KEPT_LAWS
from mime.interactions import Contact, Drag, Gravity, Magnetic
from mime.scene import KEPT, Box, Capsule, Cylinder, Ellipsoid, Screw, Sphere, Tube

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "sim2real"
#: The versions registered on 2026-10-31, when 7j's register froze: what its scripts must build until it is amended.
REGISTER_FROZE = {"screw": "0.2.0", "sphere": "0.1.0", "capsule": "0.1.0", "box": "0.1.0", "cylinder": "0.1.0",
                  "ellipsoid": "0.1.0", "tube": "0.1.0"}
REGISTER_FROZE_DYNAMICS = {"rigid_inertial": "0.2.0"}
CURRENT = {cls.name: cls.version for cls in (Screw, Sphere, Capsule, Box, Cylinder, Ellipsoid, Tube)}
#: The interactions' versions registered on 2026-10-31, and the current ones.
REGISTER_FROZE_INTERACTIONS = {"drag": "0.3.0", "magnetic.torque": "0.1.0", "gravity": "0.1.0", "contact": "0.1.0"}
CURRENT_INTERACTIONS = {cls.name: cls.version for cls in (Drag, Magnetic.Torque, Gravity, Contact)}
#: What builds a geometry or an experiment in a script: a kind's class, script A's shell or arms, an experiment.
BUILDERS = {cls.__name__ for cls in (Screw, Sphere, Capsule, Box, Cylinder, Ellipsoid, Tube)} | {
    held.__name__ for held, _ in KEPT.values()} | {"fl9", "script_a", "record_arm", "Experiment"} | {
    cls.__name__ for cls in (RigidInertial, *(held for held, _ in KEPT_DYNAMICS.values()))}
#: ``same_as_script_a`` builds script A's reproduction arm as the reference its graph is compared with, not as an
#: experiment of the register's.
COMPARED_WITH = {"reproduction_arm"}


def _operating_point():
    spec = importlib.util.spec_from_file_location("_frozen_operating_point", SCRIPTS / "operating_point.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


op = _operating_point()


def geometries(document: dict) -> list[tuple[str, str, str]]:
    """Each geometry the document names, a structure's or a body's part's: where, its kind and its version."""
    found = []
    for thing, held in document["things"].items():
        if "geometry" in held:
            found.append((thing, held["geometry"]["name"], held["geometry"]["version"]))
        for part, declared in held.get("parts", {}).items():
            found.append((f"{thing}.{part}", declared["solid"]["name"], declared["solid"]["version"]))
    return found


def dynamics_of(document: dict) -> list[tuple[str, str, str]]:
    """Each dynamics the document names for a body: where, its entry and its version."""
    return [(thing, held["dynamics"]["motion"]["name"], held["dynamics"]["motion"]["version"])
            for thing, held in document["things"].items() if held["kind"] == "body"]


def unfrozen_dynamics(document: dict) -> list[str]:
    return [f"{where}: {entry} {version}, the register froze {op.FROZEN_DYNAMICS.get(entry, 'none')}"
            for where, entry, version in dynamics_of(document) if op.FROZEN_DYNAMICS.get(entry) != version]


def unfrozen(document: dict) -> list[str]:
    return [f"{where}: {kind} {version}, the register froze {op.FROZEN_VERSIONS.get(kind, 'none')}"
            for where, kind, version in geometries(document) if op.FROZEN_VERSIONS.get(kind) != version]


ARMS = [{"rung": "surrogate"}, {"rung": "bem", "patches": "closed"}, {"rung": "bem", "patches": "lateral"}]


@pytest.mark.parametrize("arm", ARMS, ids=lambda arm: "/".join(arm.values()))
def test_every_experiment_the_registers_scripts_build_names_the_versions_it_froze(arm):
    document = op.experiment(**arm).document()
    assert {kind for _, kind, _ in geometries(document)} == {"screw", "tube"}
    assert unfrozen(document) == []
    assert dynamics_of(document) == [("umr", "rigid_inertial", "0.2.0")] and unfrozen_dynamics(document) == []


def laws(document: dict) -> list[str]:
    return [f"{held['name']} {held['version']}, the register froze "
            f"{op.FROZEN_INTERACTIONS.get(held['name'], 'none')}" for held in document["interactions"]
            if op.FROZEN_INTERACTIONS.get(held["name"]) != held["version"]]


@pytest.mark.parametrize("arm", ARMS, ids=lambda arm: "/".join(arm.values()))
def test_every_law_the_registers_scripts_attach_names_the_version_it_froze(arm):
    document = op.experiment(**arm).document()
    assert {held["name"] for held in document["interactions"]} == set(REGISTER_FROZE_INTERACTIONS)
    assert laws(document) == []


def test_the_scripts_interactions_table_is_the_versions_registered_when_the_register_froze():
    assert op.FROZEN_INTERACTIONS == REGISTER_FROZE_INTERACTIONS
    for name, version in REGISTER_FROZE_INTERACTIONS.items():
        assert CURRENT_INTERACTIONS[name] == version or (name, version) in KEPT_LAWS, f"{name} {version}"


def test_control_the_experiment_with_the_laws_pin_bypassed_names_drag_0_4(monkeypatch):
    monkeypatch.setattr(op, "pinned_law", lambda law: law)
    assert laws(op.experiment(rung="bem").document()) == [f"drag {Drag.version}, the register froze 0.3.0"]
    assert Drag.version != "0.3.0"


def test_the_scripts_table_is_the_versions_registered_when_the_register_froze():
    assert op.FROZEN_VERSIONS == REGISTER_FROZE
    for kind, version in REGISTER_FROZE.items():
        assert CURRENT[kind] == version or (kind, version) in KEPT, f"{kind} {version} is not registered"


def test_the_dynamics_table_is_the_version_registered_when_the_register_froze():
    assert op.FROZEN_DYNAMICS == REGISTER_FROZE_DYNAMICS
    for entry, version in REGISTER_FROZE_DYNAMICS.items():
        assert (entry, version) in KEPT_DYNAMICS and KEPT_DYNAMICS[(entry, version)][0].version == version
        assert RigidInertial.version != version


#: The register's figure for the transverse-inertia arm (its section 2a: x1 to x4.32, 3.2086 to 3.2069 mm/s), to the
#: digits it states, and the half unit of its last digit.
REGISTER_X4_32_MM_S, REGISTER_DIGITS = 3.20686, 5e-6


def transverse_speed_mm_s(pinned: bool) -> float:
    """The arm with the body's transverse inertia scaled by 4.32 (the register's high arm), run as the register runs it
    (5 s at its step, the surrogate), its steady swim speed in mm/s, with the dynamics' pin on or bypassed."""
    saved = op.pinned_dynamics
    if not pinned:
        op.pinned_dynamics = lambda dynamics: dynamics
    try:
        built = op.experiment(inertia_scale=(1.0, 4.32, 4.32)).build()
        return float(op.run_steady(built, op.DT_MS, op.DURATION_S)["swim_speed_axial_m_s"]) * 1e3
    finally:
        op.pinned_dynamics = saved


def test_the_transverse_inertia_arm_reproduces_the_registers_speed():
    speed = transverse_speed_mm_s(pinned=True)
    print(f"\nx4.32 arm, pinned: {speed:.6f} mm/s (the register: {REGISTER_X4_32_MM_S})")
    assert abs(speed - REGISTER_X4_32_MM_S) <= REGISTER_DIGITS


def test_the_surrogate_arm_built_pinned_is_script_as_reproduction_arm_leaf_for_leaf():
    """With its control, as ``surrogate.py`` plants it: the contact's margin moved by a micrometre is found."""
    assert op.same_as_script_a(op.experiment().build()) == []
    assert op.same_as_script_a(op.experiment(margin_mm=0.1001).build()) != []


def builds_unpinned(path: Path) -> list[str]:
    """Each call in ``path`` that builds a geometry, a body's dynamics or an experiment, other than the ones
    ``operating_point.py`` makes (an experiment, each geometry passed through ``pinned`` and each dynamics through
    ``pinned_dynamics``)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pinned_args = {id(arg) for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id in ("pinned", "pinned_dynamics")
                   for arg in node.args}
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else None)
        if name in COMPARED_WITH or name not in BUILDERS:
            continue
        if path.name == "operating_point.py" and (name == "Experiment" or id(node) in pinned_args):
            continue
        found.append(f"{path.name}:{node.lineno} {name}(...)")
    return found


def test_no_script_of_the_register_builds_a_geometry_or_dynamics_unpinned_or_an_experiment_of_its_own():
    scripts = sorted(SCRIPTS.glob("*.py"))
    assert "operating_point.py" in {path.name for path in scripts} and len(scripts) >= 8
    assert [line for path in scripts for line in builds_unpinned(path)] == []


def test_control_the_experiment_with_the_pin_bypassed_names_the_screw_kind_quasi_uniform_sampling_moved(monkeypatch):
    monkeypatch.setattr(op, "pinned", lambda geometry: geometry)
    found = unfrozen(op.experiment(rung="bem").document())
    assert found == [f"umr.shell: screw {Screw.version}, the register froze 0.2.0"] and Screw.version != "0.2.0"


def test_control_a_script_that_builds_the_shell_unpinned_fails_the_source_scan(tmp_path):
    planted = tmp_path / "operating_point.py"
    source = (SCRIPTS / "operating_point.py").read_text(encoding="utf-8")
    assert source.count("Part(pinned(A.fl9()))") == 1
    planted.write_text(source.replace("Part(pinned(A.fl9()))", "Part(A.fl9())"), encoding="utf-8")
    assert [line.split(" ", 1)[1] for line in builds_unpinned(planted)] == ["fl9(...)"]
    other = tmp_path / "surrogate.py"
    other.write_text("import operating_point as op\nshell = op.A.fl9()\n", encoding="utf-8")
    assert builds_unpinned(other) == ["surrogate.py:2 fl9(...)"]


def test_control_the_experiment_with_the_dynamics_pin_bypassed_names_the_law_the_gyroscopic_fix_moved(monkeypatch):
    monkeypatch.setattr(op, "pinned_dynamics", lambda dynamics: dynamics)
    document = op.experiment(rung="bem").document()
    assert unfrozen_dynamics(document) == ["umr: rigid_inertial 0.3.0, the register froze 0.2.0"]


def test_control_the_transverse_inertia_arm_on_the_current_law_is_off_the_registers_speed():
    """The arm that moved under the version: 3.20747 mm/s on the implicit gyroscopic term, 6e-4 from the register's
    figure, a hundred times its last digit."""
    speed = transverse_speed_mm_s(pinned=False)
    print(f"\nx4.32 arm, dynamics unpinned: {speed:.6f} mm/s")
    assert abs(speed - REGISTER_X4_32_MM_S) > 100 * REGISTER_DIGITS


def test_control_a_script_that_builds_the_body_unpinned_fails_the_source_scan(tmp_path):
    planted = tmp_path / "operating_point.py"
    source = (SCRIPTS / "operating_point.py").read_text(encoding="utf-8")
    assert source.count("dynamics=pinned_dynamics(RigidInertial())") == 1
    planted.write_text(source.replace("dynamics=pinned_dynamics(RigidInertial())", "dynamics=RigidInertial()"),
                       encoding="utf-8")
    assert [line.split(" ", 1)[1] for line in builds_unpinned(planted)] == ["RigidInertial(...)"]
