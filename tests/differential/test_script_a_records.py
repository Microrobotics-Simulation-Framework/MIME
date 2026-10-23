"""Script A's record arm, the gate of dev1's phase 5: materials read from OpenMaterica's records through the seam.

**First, the reproduction arm through the seam is phase 4's arm exactly.** Its values are now read through
``mime.materials``; its derived graph (every node, edge and parameter, bit for bit) is the one phase 4 left, frozen
before phase 5's seam was merged, and in the environment that froze it, so is every array of its 5 s run. Phase 4's
arm as it merged is in ``references/script_a_phase4_reproduction.json``; since phase 4c3 changed its physics (three
declared changes), the arm is held to ``references/script_a_4c3_reproduction.json``, frozen on 4c3's head before it
merged phase 5. The control: with one inline value 1% off, neither holds.

**Then the records, one at a time.** Each switch (water's density, water's viscosity, N45's remanence through two
magnet parts) changes A's steady quantities by what ``script_a_record_switches.json`` predicted before any record arm
existed (``predict_script_a_records.py``: phase 4's emulation run with the switched values), within the tolerances
stated there; since 4c3, what ``corrected_physics_4c3`` predicted on 4c3's physics, declared before the arm ran on it.
A predicted change below its tolerance cannot be told from no change, and is reported as such: on 4c3's physics the
density switch's whole change is (change 2 removed the O(dt) load phase 5 credited it to). The
controls: each switch misses the other switches' predictions, and the reproduction misses each switch's prediction
wherever that prediction is resolvable.

**The last arm differs from the reproduction only by what the report lists**: every parameter of the derived graph
that differs belongs to a node of an entry that reads a value whose line in the report differs, and every such line
says where its value came from.

**What this bounds.** That MIME applies the records' values as the predictions say, through the same dynamics as
the reproduction. It says nothing about whether the records describe A's water and magnets: that is the experiment's
declaration (its acceptances, with their reasons), and the dynamics' own error is phase 4's.
"""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pytest
from script_a import ALL_RECORDS, DENSITY_RECORD, MAGNET_PARTS, VISCOSITY_RECORD, record_arm, reproduction_arm
from script_a_runs import quantities, run_built
from script_a_steady import GATED

from mime import units
from mime.saved._record import environment

HERE = Path(__file__).parent
PREDICTED = json.loads((HERE / "script_a_record_switches.json").read_text())
FROZEN = json.loads((HERE / "references" / "script_a_4c3_reproduction.json").read_text())
TRANSIENT = "transient_speed_0_to_0.5_s_m_s"
QUANTITIES = (*GATED, TRANSIENT)
TOLERANCES = PREDICTED["tolerances"]["values"]
SWITCHES = {"water_density": frozenset({DENSITY_RECORD}), "water_viscosity": frozenset({VISCOSITY_RECORD}),
            "n45_through_parts": frozenset({MAGNET_PARTS})}
#: The predictions on the physics in force: phase 4c3's, declared before the record arm ran on it.
PREDICTION = PREDICTED["corrected_physics_4c3"]["predicted"]
#: Switches whose predicted change no quantity resolves, as the declaration says.
UNRESOLVABLE = frozenset({"water_density"})


def change(name: str, value: float, against: float) -> float:
    return value - against if name == "phase_lag_field_minus_moment_rad" else (value - against) / abs(against)


def array_digest(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(f"{array.dtype.str}{array.shape}".encode() + array.tobytes()).hexdigest()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor()


def this_environment() -> dict[str, str]:
    held = environment()
    return {"python": held["python"], "jax": held["jax"], "jaxlib": held["jaxlib"], "platform": held["platform"],
            "cpu": cpu_model()}


def run_arm(built):
    history = run_built(built)
    return history, quantities(history, 10.0)


@pytest.fixture(scope="module")
def reproduction():
    built = reproduction_arm().build()
    history, found = run_arm(built)
    return built, history, found


@pytest.fixture(scope="module")
def arms():
    return {name: run_arm(record_arm(records).build())[1] for name, records in
            (*SWITCHES.items(), ("all_records", ALL_RECORDS))}


def test_the_reproduction_arm_through_the_seam_is_phase_4s_arm_bit_for_bit(reproduction):
    built, history, found = reproduction
    frozen = FROZEN["runs"]["f64"]
    assert built.record.hashes()["derived"] == frozen["derived_hash"]
    digests = {"umr/pose": array_digest(np.concatenate([history.position, history.orientation], axis=1)),
               "umr/twist": array_digest(np.concatenate([history.velocity, history.angular_velocity], axis=1)),
               "field/flux_density": array_digest(history.field)}
    same_environment = FROZEN.get("environment") == this_environment()
    print(f"\nthe derived graph is phase 4's: {frozen['derived_hash']}\nfrozen in {FROZEN.get('environment')}\n"
          f"running in {this_environment()}\nthe run's arrays, bit for bit: "
          f"{'yes' if digests == frozen['arrays'] else 'no'}")
    if same_environment:
        assert digests == frozen["arrays"]
    for name in QUANTITIES:
        assert abs(change(name, found[name], frozen["steady"][name])) <= 1e-12, name


def test_the_reproduction_control_one_inline_value_one_percent_off_changes_the_graph_and_the_run(reproduction):
    """Water's viscosity 1% off (phase 5 took the density, whose effect on A's speed 4c3's physics leaves below 1e-6:
    the O(dt) load it moved is gone)."""
    built, _, found = reproduction
    experiment = reproduction_arm()
    held = [d for d in experiment._declarations if getattr(d, "path", None) == "water.viscosity"]
    assert len(held) == 1
    experiment._declarations[experiment._declarations.index(held[0])] = type(held[0])(
        "water.viscosity", units.mPa_s(1.01), held[0].reason)
    moved = experiment.build()
    assert moved.record.hashes()["derived"] != built.record.hashes()["derived"]
    _, off = run_arm(moved)
    assert abs(change("swim_speed_axial_m_s", off["swim_speed_axial_m_s"], found["swim_speed_axial_m_s"])) > 1e-6


def rows(measured: dict, base: dict, predicted: dict) -> list[tuple[str, float, float, float, bool, bool]]:
    """Per quantity: the measured change, the predicted one, their difference, whether it holds, and whether the
    predicted change is above ten times the tolerance (resolvable)."""
    found = []
    for name in QUANTITIES:
        moved = change(name, measured[name], base[name])
        miss = moved - predicted[name]
        found.append((name, moved, predicted[name], miss, abs(miss) <= TOLERANCES[name],
                      abs(predicted[name]) > 10 * TOLERANCES[name]))
    return found


def table(title: str, found) -> str:
    lines = [title, f"{'quantity':38s} {'measured':>12s} {'predicted':>12s} {'miss':>10s} {'tolerance':>9s}"]
    lines += [f"{name:38s} {moved:+12.4e} {predicted:+12.4e} {miss:+10.2e} {TOLERANCES[name]:9.0e} "
              f"{'ok' if holds else 'MISS'}{'' if resolvable else ' (below 10x tolerance: cannot discriminate)'}"
              for name, moved, predicted, miss, holds, resolvable in found]
    return "\n".join(lines)


@pytest.mark.parametrize("switch", [*SWITCHES, "all_records"])
def test_each_record_switch_changes_a_as_predicted_before_the_run(switch, reproduction, arms):
    _, _, base = reproduction
    found = rows(arms[switch], base, PREDICTION[switch])
    print("\n" + table(f"{switch}: the change from the reproduction arm, relative (the lag in rad)", found))
    assert [name for name, *_, holds, _ in found if not holds] == []
    if switch in UNRESOLVABLE:
        assert not any(resolvable for *_, resolvable in found), "declared below every tolerance"
        print(f"{switch}: below ten times every tolerance, as declared: no evidence either way")
    else:
        assert any(resolvable for *_, resolvable in found), "no quantity can resolve this switch's predicted change"


@pytest.mark.parametrize("switch", list(SWITCHES))
def test_the_controls_each_switch_misses_the_others_and_the_reproduction_misses_each(switch, reproduction, arms):
    _, _, base = reproduction
    for other in SWITCHES:
        if other != switch:
            found = rows(arms[switch], base, PREDICTION[other])
            assert any(not holds for *_, holds, _ in found), f"{switch} matches {other}'s prediction"
    unmoved = rows(base, base, PREDICTION[switch])
    assert all(not holds for *_, holds, resolvable in unmoved if resolvable)


def changed_values(first, second) -> dict[str, tuple]:
    a = {line.path: line for line in first.report.values}
    b = {line.path: line for line in second.report.values}
    return {path: (a.get(path), b.get(path)) for path in sorted(set(a) | set(b))
            if a.get(path) is None or b.get(path) is None or a[path].value != b[path].value
            or a[path].source != b[path].source}


def test_the_record_arm_differs_from_the_reproduction_only_by_what_its_report_lists(reproduction):
    built = reproduction[0]
    records = record_arm().build()
    listed = changed_values(built, records)
    print("\nvalues whose line differs between the reproduction and the record arm:")
    for path, (before, after) in listed.items():
        print(f"  {path}: {before.value if before else '(none)'} -> {after.value if after else '(none)'}  "
              f"[{after.source if after else ''}]")
    readers = {reader for path, (_, after) in listed.items() if after is not None for reader in after.read_by}
    nodes = {node: plan.owner for plan in records.plans.entries for node in plan.node_names()}
    before = {node["name"]: node["params"] for node in built.record.derived.physics["nodes"]}
    after = {node["name"]: node["params"] for node in records.record.derived.physics["nodes"]}
    assert set(before) == set(after), "the record arm builds other nodes"
    moved = sorted(name for name in after if json.dumps(after[name], sort_keys=True)
                   != json.dumps(before[name], sort_keys=True))
    print(f"nodes whose parameters differ: {moved}, owned by {sorted({nodes[n] for n in moved})}")
    assert moved, "the record arm's graph is the reproduction's: the switches changed nothing"
    assert {nodes[name] for name in moved} <= readers
    for path, (_, after) in listed.items():
        assert after is not None and after.source and after.reason, path
        if after.source.startswith("derived from its parts"):
            assert after.details, f"{path} is derived and lists no inputs"
    edges = [json.dumps(e, sort_keys=True) for e in built.record.derived.physics["edges"]]
    assert edges == [json.dumps(e, sort_keys=True) for e in records.record.derived.physics["edges"]]


def test_the_final_arm_lists_the_records_it_read_with_their_pins_and_verdicts():
    built = record_arm().build()
    text = built.report.text()
    for pin in ("example/water 1.0.0", "example/ndfeb_n45 0.1.0", "sha256:88851e6263f4dba3f41137f07a8a95dba13cb09bf3",
                "sha256:ca2a4e1aed89c12465f3005a686772c642278feb9664ca7969598a6c9ef7850d"):
        assert pin in text, pin
    reads = {(read.material, read.field): read for read in built.plans.materials.reads}
    assert reads[("water", "bulk.density")].status == "met"
    assert reads[("water", "fluid.dynamic_viscosity")].status == "unknown"
    assert reads[("n45", "magnetic.remanence")].bounds == (1.32, 1.37)
    assert built.report.values and "temperature 293.15 K +- 0.5 K" in text


def test_the_example_declares_the_record_arm():
    """``examples/script_a.py``'s ``record_arm``, written as a user writes it, declares this record arm: the same
    things, parts, materials, laws, settings and values (only the reasons' wording differs), so the same graph."""
    import importlib.util

    root = HERE.parents[1]
    spec = importlib.util.spec_from_file_location("example_script_a", root / "examples" / "script_a.py")
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    theirs, ours = example.record_arm(), record_arm()

    def without_reasons(document):
        declared = {path: {k: v for k, v in value.items() if k != "reason"}
                    for path, value in document["declared"].items()}
        return {**document, "declared": declared}

    assert without_reasons(theirs.document()) == without_reasons(ours.document())
    assert theirs.build().record.hashes()["derived"] == ours.build().record.hashes()["derived"]
