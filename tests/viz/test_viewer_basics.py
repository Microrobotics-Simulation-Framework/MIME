"""The browser view's basics, each switched on and off by an argument of ``display``, ``page`` and ``web`` and a flag of
``mime visuals web``: a scale bar, focus on a body and following it, an arrow legend, and A0's motor and magnet drawn
for the view alone.

**What this bounds.** That each switch changes the page or the glTF as stated, read back by parsing (the page's HTML
and its config's JSON, the GLB's JSON and accessors), never by a substring; that the arithmetic the page runs in the
browser (the scale bar's length, the followed position) is right where it is run, in Node, on the page's own text; and
that nothing is written into a saved stage. **What it does not.** That the browser draws it: CI has no WebGL, and the
page was looked at by hand in headless Chrome (``scripts/offline_view_proof.py`` shows the page loads). The scale bar
is true at the point the camera orbits, which is the camera's own geometry (its field of view and distance) and is
tested here as arithmetic, not by a pixel.

**Controls.** A switch wired to nothing (the page made with the argument ignored) fails its test on every run; a
followed target lagged by one frame, a scale bar that ignores the zoom, a shape that stands still, and a write into the
stage, a write into a saved folder's stage or into the asset cache each fail theirs. Each control is a call that must
return a problem, and so fails the test that holds it.

**The page's own script is run too.** Its arithmetic (``_LOGIC``) is run in Node on the page's text, and so is the glue
that feeds it (``_SCRIPT``): against a stand-in for the viewer element and the document, which answer as the viewer's
API is documented to (a camera orbit, a field of view, a height in pixels, a current time, a target to set). That bounds
the page's wiring (which property is read, which argument goes where); it does not bound what a real viewer does with
the target it is given, which is the browser's and was looked at by hand."""

import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import warnings
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
from gltf_cases import accessor, ball, glb_parts, node_by_usd, recorded, with_pose
from pxr import Usd
from script_a0 import a0 as script_a0

import mime.viz._apparatus as apparatus_module
import mime.viz._export as export_module
import mime.viz._web as web_module
from mime import Body, Experiment, Medium, State
from mime.build import ArgumentRefused
from mime.cli import main
from mime.dynamics import RigidInertial
from mime.interactions import Gravity
from mime.saved import dump_document
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import kg_per_m3, m, m_per_s, m_per_s2, mPa_s, ms, rad_per_s, um
from mime.viz import display, gltf, page, web
from mime.viz._apparatus import APPARATUS_ROOT, PLACEHOLDER_FRACTION
from mime.viz._export import length_text
from mime.viz._web import BAR_PIXELS, FOCUS_FACTOR, OUTPUT_WARNING_BYTES

ARROW = "/World/rpm/magnetic/magnetic_torque_umr"
HELIX_LENGTH_M = 7.47e-3
MAGNET_RADIUS_M, MAGNET_LENGTH_M = 17.5e-3, 20e-3


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch) -> Path:
    """The asset cache, with a file in it: a view that rewrote or added to it would show."""
    held = tmp_path / "cache"
    (held / "baked").mkdir(parents=True)
    (held / "baked" / "held.bin").write_bytes(b"a baked mesh, as a view must leave it")
    monkeypatch.setenv("MIME_ASSETS_DIR", str(held))
    return held


# --- the page, read back ---------------------------------------------------------------------------------------------


class Facts(HTMLParser):
    """The elements of a page that carry an id, the state of each toggle in its notes, and the legend's arrows."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.toggles: dict[str, str] = {}
        self.legend: list[str] = []
        self.labels: list[str] = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        held = dict(attrs)
        if "id" in held:
            self.ids.add(held["id"])
        if tag == "li" and "data-toggle" in held:
            self.toggles[held["data-toggle"]] = held["data-state"]
        if tag == "li" and "data-prim" in held:
            self.legend.append(held["data-prim"])


def config_of(text: str) -> dict:
    (held,) = re.findall(r"const config = (\{.*?\});", text)
    return json.loads(held)


def observable(text: str) -> dict[str, object]:
    """What a switch can change in a page: the elements, the config's switches, and the notes' toggles."""
    facts, config = Facts(text), config_of(text)
    return {"scale-bar": "scale-bar" in facts.ids, "legend": "legend" in facts.ids, "follow-box": "follow" in facts.ids,
            "config.scale_bar": config["scale_bar"], "config.legend": config["legend"],
            "config.focus": config["focus"], "config.follow": config["follow"], "toggles": dict(facts.toggles),
            "arrows": tuple(facts.legend)}


def changed(first: str, second: str) -> set[str]:
    one, two = observable(first), observable(second)
    return {key for key in one if one[key] != two[key]}


@pytest.fixture(scope="module")
def ball_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ball") / "run"
    recorded(ball().build(), folder, 5, lambda k, start: with_pose(start, "ball", [0, 1e-6 * k, -2e-6 * k, 1, 0, 0, 0]))
    return folder


def two_balls() -> Experiment:
    """Two steel balls of different sizes, ``ball`` (250 um) and ``far`` (400 um): a view that names the wrong one of
    them is not the view of the right one, which a scene with one body cannot tell."""
    exp = Experiment("two_balls", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.1))
    water = exp.add_medium(Medium("water"))
    exp.declare("water.density", kg_per_m3(998.0), reason="water at 20 degC")
    exp.declare("water.viscosity", mPa_s(1.0), reason="water at 20 degC")
    for name, x, radius in (("ball", 0.0, 250.0), ("far", 1.0, 400.0)):
        held = exp.add_body(Body(name, dynamics=RigidInertial(), initial=State(
            position=m((x, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
            angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(Sphere(radius=um(radius)))}))
        exp.declare(f"{name}.parts.steel.density", kg_per_m3(7850.0), reason="steel")
        exp.attach(Gravity(), body=held, medium=water)
    return exp


#: ``ball``'s position at sample ``k`` in the two-body scene: a different step along each axis, so that an axis swapped
#: is seen; ``far`` stays at one metre on x.
def two_balls_ball(k: int) -> list[float]:
    return [2e-4 * k, 5e-5 * k * k, -1e-4 * k, 1, 0, 0, 0]


@pytest.fixture(scope="module")
def two_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("two") / "run"
    recorded(two_balls().build(), folder, 6,
             lambda k, start: with_pose(with_pose(start, "ball", two_balls_ball(k)), "far", [1.0, 0, 0, 1, 0, 0, 0]))
    return folder


# --- A0 ----------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def a0_built():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        experiment = script_a0()
        return experiment, experiment.build()


def digests(folder: Path) -> dict[str, str]:
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob("*")) if path.is_file()}


#: The rotor frame's pose at each of five samples (x, y, z, then a turn about z of 0.3 rad a sample).
def rotor_pose(k: int) -> list[float]:
    angle = 0.3 * k
    return [0.01 * k, 0.02, 0.15, math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]


def base_pose(start, k: int) -> np.ndarray:
    """The motor's base frame at sample ``k``: where the start has it, moved 4 mm along x and 1 mm along z a sample."""
    pose = np.array(start["rpm.motor"]["frame_poses"][0], dtype=np.float64, copy=True)
    pose[0] += 4e-3 * k
    pose[2] += 1e-3 * k
    return pose


def a0_run(a0_built, folder: Path, *, saved: bool, tesla: float = 0.0) -> Path:
    """A run folder of A0 over five samples: the rotor frame moved by ``rotor_pose`` and the body by a line; the saved
    experiment's document beside the stage where ``saved``. The field is ``tesla`` along y from the second sample."""
    experiment, built = a0_built
    if saved:
        (folder / "experiment").mkdir(parents=True)
        (folder / "experiment" / "experiment.mime.yaml").write_text(dump_document(experiment.document()), encoding="utf-8")
    key = next(e for e in built.binding().entries if e.kind == "vector")

    def state_at(k, start):
        held = with_pose(start, "rpm.motor", np.array(rotor_pose(k)), 1)
        held = with_pose(held, "rpm.motor", base_pose(start, k), 0)
        held = with_pose(held, "umr", np.array([1e-3 * k, 0.0, -1e-3, 1, 0, 0, 0]))
        held[key.node][key.field] = np.zeros(3) if k == 0 else np.array([0.0, tesla, 0.0])
        return held

    return recorded(built, folder, 5, state_at).parent


@pytest.fixture(scope="module")
def declared(a0_built, tmp_path_factory):
    folder = tmp_path_factory.mktemp("declared") / "run"
    return a0_run(a0_built, folder, saved=True)


@pytest.fixture(scope="module")
def saved_a0(a0_built, tmp_path_factory):
    """A0's saved folder (document, lock, scene, descriptions), where the stage ``scene/scene.usda`` is the root layer."""
    _, built = a0_built
    folder = tmp_path_factory.mktemp("saved") / "a0"
    built.save(folder, steps=2)
    return folder


@pytest.fixture(scope="module")
def undeclared(a0_built, tmp_path_factory):
    return a0_run(a0_built, tmp_path_factory.mktemp("undeclared") / "run", saved=False)


@pytest.fixture(scope="module")
def capped(a0_built, tmp_path_factory):
    return a0_run(a0_built, tmp_path_factory.mktemp("capped") / "run", saved=True, tesla=0.010)


def overlay_nodes(result) -> list[str]:
    document, _ = glb_parts(result.glb)
    return sorted(node["extras"]["usd"] for node in document["nodes"]
                  if node.get("extras", {}).get("usd", "").startswith(APPARATUS_ROOT))


def world_nodes(result) -> list[str]:
    document, _ = glb_parts(result.glb)
    return sorted(node["extras"]["usd"] for node in document["nodes"]
                  if node.get("extras", {}).get("usd", "").startswith("/World"))


def tracks_of(result, prim: str) -> dict[str, np.ndarray]:
    document, binary = glb_parts(result.glb)
    index, _ = node_by_usd(document, prim)
    (animation,) = document["animations"]
    return {c["target"]["path"]: accessor(document, binary, animation["samplers"][c["sampler"]]["output"])
            for c in animation["channels"] if c["target"]["node"] == index}


def model_space(result, xyz) -> np.ndarray:
    """A stage point (metres, z up) in the glTF's coordinates, as the viewer's camera sees it: y up, in the glTF unit."""
    x, y, z = np.asarray(xyz, dtype=np.float64).T
    return np.stack([x, z, -y], axis=-1) / result.unit_m


# --- each switch, on and off -------------------------------------------------------------------------------------------


def test_the_scale_bar_is_on_by_default_and_off_removes_exactly_it(ball_run):
    result = gltf(ball_run)
    on, off = page(result), page(result, scale_bar=False)
    assert page(result, scale_bar=True) == on
    assert changed(on, off) == {"scale-bar", "config.scale_bar", "toggles"}
    assert observable(on)["scale-bar"] is True and observable(on)["config.scale_bar"] is True
    assert observable(off)["scale-bar"] is False and observable(off)["config.scale_bar"] is False
    assert observable(on)["toggles"]["scale_bar"] == "on" and observable(off)["toggles"]["scale_bar"] == "off"


def test_the_legend_is_on_by_default_and_off_removes_exactly_it(declared):
    result = gltf(declared)
    on, off = page(result), page(result, legend=False)
    assert changed(on, off) == {"legend", "config.legend", "toggles", "arrows"}
    assert observable(on)["arrows"] == (ARROW,) and observable(off)["arrows"] == ()
    assert observable(on)["config.legend"] is True and observable(off)["config.legend"] is False


def test_the_apparatus_shapes_are_on_by_default_and_off_leaves_the_glb_without_them(declared):
    on, off = gltf(declared), gltf(declared, apparatus_shapes=False)
    assert gltf(declared, apparatus_shapes=True).glb == on.glb
    assert overlay_nodes(on) == [f"{APPARATUS_ROOT}", f"{APPARATUS_ROOT}/rpm_motor_housing_placeholder",
                                 f"{APPARATUS_ROOT}/rpm_motor_housing_placeholder/shape",
                                 f"{APPARATUS_ROOT}/rpm_rotor_magnet", f"{APPARATUS_ROOT}/rpm_rotor_magnet/shape"]
    assert overlay_nodes(off) == [] and off.shapes == [] and off.apparatus_shapes is False
    assert world_nodes(on) == world_nodes(off), "the stage's own prims are the same with the shapes on and off"
    assert Facts(page(on)).toggles["apparatus_shapes"] == "on" and Facts(page(off)).toggles["apparatus_shapes"] == "off"
    said = glb_parts(on.glb)[0]["asset"]["extras"]["mime"]["apparatus_shapes"]
    assert sorted(shape["name"] for shape in said) == ["rpm.motor", "rpm.rotor_magnet"]
    assert glb_parts(off.glb)[0]["asset"]["extras"]["mime"]["apparatus_shapes"] == []


def test_focus_and_follow_are_off_until_a_body_is_named(ball_run):
    result = gltf(ball_run)
    plain = page(result)
    assert observable(plain)["config.focus"] is None and observable(plain)["config.follow"] is None
    focused, followed = page(result, focus="ball"), page(result, follow="ball")
    assert changed(plain, focused) == {"config.focus", "toggles"}
    assert changed(plain, followed) == {"config.follow", "toggles", "follow-box"}
    assert observable(focused)["toggles"]["focus"] == "on" and observable(followed)["toggles"]["follow"] == "on"
    assert observable(plain)["toggles"]["focus"] == "off" and observable(plain)["toggles"]["follow"] == "off"
    track = result.bodies["ball"]
    focus = config_of(focused)["focus"]
    assert focus["name"] == "ball" and focus["target"] == pytest.approx(track.positions[0], rel=1e-8, abs=1e-12)
    assert focus["radius"] == pytest.approx(FOCUS_FACTOR * track.extent) and track.extent == pytest.approx(2e-5 / result.unit_m)


@pytest.mark.parametrize("name", ["scale_bar", "legend"])
def test_control_a_switch_wired_to_nothing_changes_nothing_and_fails(declared, name, monkeypatch):
    """The page made with the argument ignored is the page with it on: the test above would see no change."""
    result = gltf(declared)
    expected = {"scale_bar": {"scale-bar", "config.scale_bar", "toggles"},
                "legend": {"legend", "config.legend", "toggles", "arrows"}}[name]
    original = web_module.page
    monkeypatch.setattr(web_module, "page", lambda result, **kw: original(result, **{**kw, name: True}))
    assert changed(web_module.page(result), web_module.page(result, **{name: False})) != expected


def test_control_an_apparatus_switch_wired_to_nothing_leaves_the_shapes_in(declared, monkeypatch):
    original = web_module.gltf
    monkeypatch.setattr(web_module, "gltf", lambda target, **kw: original(target, **{**kw, "apparatus_shapes": True}))
    assert overlay_nodes(web_module.gltf(declared, apparatus_shapes=False)) != []


def test_the_same_switches_are_arguments_of_display_and_web(ball_run, declared, tmp_path):
    shown = display(ball_run, scale_bar=False, legend=False, focus="ball", follow="ball", apparatus_shapes=False)
    assert shown.html == page(shown.result, cdn=True, scale_bar=False, legend=False, focus="ball", follow="ball")
    written, result = web(declared, tmp_path / "a0.html", scale_bar=False, apparatus_shapes=False)
    assert written.read_text(encoding="utf-8") == page(result, scale_bar=False) and overlay_nodes(result) == []
    assert observable(written.read_text(encoding="utf-8"))["scale-bar"] is False


def test_display_and_web_draw_the_shapes_and_the_bar_and_the_legend_by_default(declared, tmp_path):
    """The defaults are the arguments' own, not the tests' choice: called with none, each is on."""
    shown = display(declared)
    written, result = web(declared, tmp_path / "a0.html")
    for result_of, text in ((shown.result, shown.html), (result, written.read_text(encoding="utf-8"))):
        assert overlay_nodes(result_of) == overlay_nodes(gltf(declared)) != [] and result_of.apparatus_shapes is True
        assert Facts(text).toggles == {"scale_bar": "on", "legend": "on", "apparatus_shapes": "on", "focus": "off", "follow": "off"}
        assert {"scale-bar", "legend"} <= Facts(text).ids
        assert observable(text)["config.focus"] is None and observable(text)["config.follow"] is None


def test_the_notes_say_what_is_on_and_what_each_switch_does(declared):
    text = page(gltf(declared), focus="umr", follow="umr")
    assert Facts(text).toggles == {name: "on" for name in ("scale_bar", "legend", "apparatus_shapes", "focus", "follow")}
    for said in ("scale_bar=False or --no-scale-bar", "legend=False or --no-legend", "apparatus_shapes=False or --no-apparatus-shapes",
                 "focus=None", "follow=None"):
        assert said in text


def test_focus_and_follow_each_name_their_own_body(two_run):
    """With two bodies, the one named is the one framed and the one followed, and naming both as different bodies keeps
    them apart: a view that took the first body, or the other argument's, is not the view of the right one."""
    result = gltf(two_run)
    ball_at = model_space(result, np.array([two_balls_ball(k)[:3] for k in range(6)]))
    far_at = model_space(result, np.array([[1.0, 0.0, 0.0]] * 6))
    unit = result.unit_m
    assert result.bodies["ball"].extent == pytest.approx(500e-6 / unit) and result.bodies["far"].extent == pytest.approx(800e-6 / unit)
    for focus, follow, framed, followed in (("ball", "far", ball_at, far_at), ("far", "ball", far_at, ball_at),
                                            ("far", "far", far_at, far_at), ("ball", "ball", ball_at, ball_at)):
        config = config_of(page(result, focus=focus, follow=follow))
        assert config["focus"]["name"] == focus and config["follow"]["name"] == follow
        assert config["focus"]["target"] == pytest.approx(framed[0], rel=1e-8, abs=1e-12)
        assert config["focus"]["radius"] == pytest.approx(FOCUS_FACTOR * result.bodies[focus].extent)
        assert follow_problems(np.asarray(config["follow"]["positions"]), followed) == []
    assert config_of(page(result, focus="far"))["follow"] is None and config_of(page(result, follow="far"))["focus"] is None


def test_the_refusal_names_the_nearest_of_two_bodies(two_run):
    with pytest.raises(ArgumentRefused) as refused:
        page(gltf(two_run), follow="fa")
    (found,) = refused.value.refusals
    assert "Did you mean 'far'?" in found.message and "The candidates are: ball, far." in found.message
    assert "follow='ball'" in found.action


def test_the_refusal_for_a_scene_with_no_bodies_gives_no_example_name(ball_run):
    nobody = dataclasses.replace(gltf(ball_run), bodies={})
    with pytest.raises(ArgumentRefused) as refused:
        page(nobody, focus="umr")
    (found,) = refused.value.refusals
    assert "no bodies" in found.action and "'umr'" not in found.action


# --- the refusal ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("argument", ["focus", "follow"])
def test_a_body_the_scene_does_not_hold_is_refused_with_the_nearest_name(ball_run, argument):
    with pytest.raises(ArgumentRefused) as refused:
        page(gltf(ball_run), **{argument: "bal"})
    (found,) = refused.value.refusals
    assert found.code == "visuals.unknown-body" and f"{argument}='bal'" in found.message
    assert "Did you mean 'ball'?" in found.message and "The candidates are: ball." in found.message
    with pytest.raises(ArgumentRefused, match="visuals.unknown-body"):
        display(ball_run, **{argument: "balls"})


def test_the_command_refuses_an_unknown_body_and_writes_no_page(ball_run, tmp_path, capsys):
    out = tmp_path / "ball.html"
    assert main(["visuals", "web", str(ball_run), "--out", str(out), "--follow", "bal"]) == 2
    assert "visuals.unknown-body" in capsys.readouterr().err and not out.exists()


def test_control_a_known_name_is_not_refused(ball_run):
    page(gltf(ball_run), focus="ball", follow="ball")


# --- follow ----------------------------------------------------------------------------------------------------------------


def node_program(logic: str, body: str) -> str:
    return f"{logic}\nconsole.log(JSON.stringify({body}));"


@pytest.fixture(scope="module")
def node():
    found = shutil.which("node")
    if found is None:
        if os.environ.get("CI"):
            pytest.fail("Node is on every CI runner: the page's arithmetic is run in it")
        pytest.skip("Node is not installed here, so the page's arithmetic is not run (it is run on CI)")
    return found


def run_node(node: str, program: str):
    done = subprocess.run([node, "--input-type=module"], input=program, capture_output=True, text=True, timeout=60,
                          check=True)  # on stdin: a long recording's program is more than an argument may be
    return json.loads(done.stdout)


def follow_targets(node: str, text: str, logic: str | None = None) -> np.ndarray:
    """What the page's own code puts the camera's target at, at each sampled display time of its config."""
    follow = config_of(text)["follow"]
    program = node_program(logic or web_module._LOGIC,
                           f"{json.dumps(follow['times'])}.map((t) => followAt({json.dumps(follow['times'])}, "
                           f"{json.dumps(follow['positions'])}, t))")
    return np.asarray(run_node(node, program))


def follow_problems(targets: np.ndarray, expected: np.ndarray) -> list[str]:
    """Where the camera's target is not at the body's recorded position, by frame."""
    wrong = [f"frame {k}: {t} != {e}" for k, (t, e) in enumerate(zip(targets, expected))
             if not np.allclose(t, e, rtol=1e-7, atol=1e-12)]
    return [f"{len(targets)} frames, not {len(expected)}", *wrong] if len(targets) != len(expected) else wrong


def test_the_camera_follows_the_bodys_recorded_positions_at_the_sampled_frames(ball_run, node):
    result = gltf(ball_run)
    text = page(result, follow="ball")
    recorded_m = np.array([[0.0, 1e-6 * k, -2e-6 * k] for k in range(5)])
    expected = model_space(result, recorded_m)
    follow = config_of(text)["follow"]
    assert len(follow["positions"]) == 5 and follow["name"] == "ball"
    assert follow["times"] == pytest.approx(np.arange(5) * 1e-4 * result.slow)
    assert follow_problems(np.asarray(follow["positions"]), expected) == []
    assert follow_problems(follow_targets(node, text), expected) == [], "the page's own code, at each sampled time"
    document, binary = glb_parts(result.glb)
    index, _ = node_by_usd(document, "/World/ball")
    (animation,) = document["animations"]
    (channel,) = [c for c in animation["channels"] if c["target"]["node"] == index and c["target"]["path"] == "translation"]
    held = accessor(document, binary, animation["samplers"][channel["sampler"]]["output"])
    times = accessor(document, binary, animation["samplers"][channel["sampler"]]["input"])
    assert follow["times"] == pytest.approx(times, rel=1e-6)
    assert document["nodes"][0]["scale"][0] == pytest.approx(1.0 / result.unit_m), "the root scales metres to the unit"
    assert follow_problems(np.asarray(follow["positions"]), model_space(result, held)) == [], \
        "and the GLB's own animation of the body, in the viewer's coordinates, is the same"


def test_control_a_target_lagged_by_one_frame_fails(ball_run, node):
    result = gltf(ball_run)
    expected = model_space(result, np.array([[0.0, 1e-6 * k, -2e-6 * k] for k in range(5)]))
    lagged = dataclasses.replace(result.bodies["ball"], positions=(result.bodies["ball"].positions[0],
                                                                    *result.bodies["ball"].positions[:-1]))
    text = page(dataclasses.replace(result, bodies={"ball": lagged}), follow="ball")
    problems = follow_problems(follow_targets(node, text), expected)
    assert [p.split(":")[0] for p in problems] == [f"frame {k}" for k in range(1, 5)]


def test_control_the_logic_with_its_interpolation_dropped_fails(ball_run, node):
    result = gltf(ball_run)
    expected = model_space(result, np.array([[0.0, 1e-6 * k, -2e-6 * k] for k in range(5)]))
    text = page(result, follow="ball")
    planted = web_module._LOGIC.replace("const weight = (t - times[low]) / (times[high] - times[low]);", "const weight = 1;")
    assert planted != web_module._LOGIC
    assert follow_problems(follow_targets(node, text, planted), expected) != []


def test_between_samples_the_target_is_the_straight_line_between_them(ball_run, node):
    result = gltf(ball_run)
    follow = config_of(page(result, follow="ball"))["follow"]
    times, positions = follow["times"], np.asarray(follow["positions"])
    middle = (times[1] + times[2]) / 2
    program = node_program(web_module._LOGIC, f"[followAt({json.dumps(times)}, {json.dumps(follow['positions'])}, {middle}), "
                           f"followAt({json.dumps(times)}, {json.dumps(follow['positions'])}, -1), "
                           f"followAt({json.dumps(times)}, {json.dumps(follow['positions'])}, 1e9)]")
    halfway, before, after = (np.asarray(v) for v in run_node(node, program))
    assert halfway == pytest.approx((positions[1] + positions[2]) / 2) and before == pytest.approx(positions[0])
    assert after == pytest.approx(positions[-1])


def test_a_still_scene_is_followed_at_its_one_position(tmp_path):
    folder = tmp_path / "still"
    built = ball().build()
    (folder / "experiment" / "scene").mkdir(parents=True)
    (folder / "experiment" / "scene" / "scene.usda").write_text(built.stage_text, encoding="utf-8")
    result = gltf(folder)
    follow = config_of(page(result, follow="ball"))["follow"]
    assert follow["times"] == [0.0] and len(follow["positions"]) == 1
    assert "follow" not in Facts(page(result, follow="ball")).ids, "no recording: no checkbox to release it"


LONG = 1500


def long_wobble(k: int) -> list[float]:
    """A body wobbling about the origin with a period of 40 samples, 20 um across on a 20 um body: a helical swimmer's
    motion, which a track sampled every third sample or so does not follow."""
    angle = 2.0 * math.pi * k / 40.0
    return [0.0, 2e-5 * math.sin(angle), 2e-5 * math.cos(angle), 1, 0, 0, 0]


@pytest.fixture(scope="module")
def long_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("long") / "run"
    recorded(ball().build(), folder, LONG, lambda k, start: with_pose(start, "ball", long_wobble(k)))
    return folder


def long_expected(result) -> np.ndarray:
    return model_space(result, np.array([long_wobble(k)[:3] for k in range(LONG)]))


def test_the_camera_follows_every_sample_of_a_recording_longer_than_the_boxs_sampling(long_run, node):
    """The motion's box is read at no more than ``BOUNDS_SAMPLES`` (512) time codes; the camera must not be: at every
    frame the animation draws, its target is the body's recorded position."""
    result = gltf(long_run)
    assert result.samples == LONG > export_module.BOUNDS_SAMPLES
    text = page(result, follow="ball")
    follow = config_of(text)["follow"]
    expected = long_expected(result)
    assert len(follow["positions"]) == len(follow["times"]) == LONG
    assert follow_problems(np.asarray(follow["positions"]), expected) == []
    assert follow_problems(follow_targets(node, text), expected) == [], "the page's own code, at each sampled time"
    document, binary = glb_parts(result.glb)
    index, _ = node_by_usd(document, "/World/ball")
    (animation,) = document["animations"]
    (channel,) = [c for c in animation["channels"] if c["target"]["node"] == index and c["target"]["path"] == "translation"]
    sampler = animation["samplers"][channel["sampler"]]
    assert follow["times"] == pytest.approx(accessor(document, binary, sampler["input"]), rel=1e-6, abs=1e-12)
    assert follow_problems(np.asarray(follow["positions"]), model_space(result, accessor(document, binary, sampler["output"]))) == [], \
        "and the GLB's own animation of the body, which keeps every sample, is the same track"


def test_control_a_track_thinned_to_the_boxs_samples_fails(long_run, monkeypatch):
    """The track the page had: 512 evenly spaced time codes. It is off the body at the frames it skips."""
    def thinned(stage):
        codes = apparatus_module.time_codes(stage)
        return [float(v) for v in np.linspace(codes[0], codes[-1], export_module.BOUNDS_SAMPLES)]

    monkeypatch.setattr(export_module, "time_codes", thinned)
    result = gltf(long_run)
    follow = config_of(page(result, follow="ball"))["follow"]
    problems = follow_problems(np.asarray(follow["positions"]), long_expected(result))
    assert problems and problems[0].startswith(f"{export_module.BOUNDS_SAMPLES} frames, not {LONG}")


def test_the_long_recordings_page_stays_under_the_notebooks_output_limit(long_run):
    assert len(display(long_run, follow="ball").html.encode("utf-8")) < OUTPUT_WARNING_BYTES


# --- the scale bar -----------------------------------------------------------------------------------------------------------


FOV, HEIGHT = 45.0, 600.0


def bar(node: str, unit_m: float, radius: float, logic: str | None = None) -> dict:
    return run_node(node, node_program(logic or web_module._LOGIC,
                                       f"scaleBar({unit_m!r}, {radius!r}, {FOV}, {HEIGHT}, {BAR_PIXELS})"))


def per_pixel(unit_m: float, radius: float) -> float:
    return unit_m * 2.0 * radius * math.tan(math.radians(FOV) / 2.0) / HEIGHT


def bar_problems(node: str, unit_m: float, radius: float, logic: str | None = None) -> list[str]:
    """The bar against the view's units: its length over its pixels is what a pixel spans, it is a round length, and
    it is near the target width."""
    held, mpp = bar(node, unit_m, radius, logic), per_pixel(unit_m, radius)
    problems: list[str] = []
    if not held["metres"] / held["pixels"] == pytest.approx(mpp, rel=1e-9):
        problems.append(f"{held['metres']} m over {held['pixels']} px is not {mpp} m/px")
    mantissa = held["metres"] / 10.0 ** math.floor(math.log10(held["metres"]) + 1e-9)
    if not any(mantissa == pytest.approx(m, rel=1e-9) for m in (1.0, 2.0, 5.0)):
        problems.append(f"{held['metres']} m is not 1, 2 or 5 times a power of ten")
    if not 0.4 * BAR_PIXELS - 1e-6 < held["pixels"] <= BAR_PIXELS + 1e-6:
        problems.append(f"{held['pixels']} px is far from {BAR_PIXELS}")
    if held["label"] != expected_label(held["metres"]):
        problems.append(f"label {held['label']!r} is not {expected_label(held['metres'])!r}")
    return problems


def expected_label(metres: float) -> str:
    for factor, unit in ((1.0, "m"), (1e-3, "mm"), (1e-6, "µm"), (1e-9, "nm"), (1e-12, "pm")):
        if metres >= factor * 0.999999 or unit == "pm":
            return f"{float(f'{metres / factor:.6g}'):g} {unit}"
    raise AssertionError


def test_the_scale_bar_matches_the_views_units_at_two_zooms(ball_run, node):
    result = gltf(ball_run)
    unit_m, far = result.unit_m, 6.0
    near = far / 10.0
    assert bar_problems(node, unit_m, far) == [] and bar_problems(node, unit_m, near) == []
    wide, close = bar(node, unit_m, far), bar(node, unit_m, near)
    assert wide["metres"] == pytest.approx(10.0 * close["metres"], rel=1e-9), "ten times nearer, a tenth the length"
    assert wide["pixels"] == pytest.approx(close["pixels"], rel=1e-9), "and the same width on the screen"
    assert wide["label"] != close["label"]
    assert config_of(page(result))["unit_m"] == unit_m and config_of(page(result))["bar_pixels"] == BAR_PIXELS


def test_the_scale_bar_is_a_round_si_length_from_nanometres_to_metres(node):
    radii = [10.0 ** (k / 7.0) for k in range(-50, 70)]
    program = node_program(web_module._LOGIC, f"{json.dumps(radii)}.map((r) => scaleBar(1e-3, r, {FOV}, {HEIGHT}, {BAR_PIXELS}))")
    held_all = run_node(node, program)
    assert held_all[0]["metres"] < 1e-9 and held_all[-1]["metres"] > 1.0, "from nanometres to metres"
    for radius, held in zip(radii, held_all):
        mpp = per_pixel(1e-3, radius)
        assert held["metres"] / held["pixels"] == pytest.approx(mpp, rel=1e-9)
        assert 0.4 * BAR_PIXELS - 1e-6 < held["pixels"] <= BAR_PIXELS + 1e-6, (radius, held)
        if held["metres"] < 1e6:
            assert held["label"] == expected_label(held["metres"]), (radius, held)


def test_control_a_scale_bar_that_ignores_the_zoom_fails(ball_run, node):
    result = gltf(ball_run)
    planted = web_module._LOGIC.replace("unitMetres * 2 * radius *", "unitMetres * 2 * 1 *")
    assert planted != web_module._LOGIC
    assert bar_problems(node, result.unit_m, 0.6, planted) != [] or bar_problems(node, result.unit_m, 60.0, planted) != []


def test_control_a_scale_bar_in_the_wrong_unit_fails(node):
    planted = web_module._LOGIC.replace('[1e-3, "mm"]', '[1e-2, "mm"]')
    assert planted != web_module._LOGIC
    assert any(bar_problems(node, 0.1, 10.0 ** (k / 3.0), planted) != [] for k in range(-30, 30))


# --- the page's own script, against a stand-in for the viewer ----------------------------------------------------------------------

VIEWER_FOV, VIEWER_HEIGHT, VIEWER_WIDTH, VIEWER_RADIUS = 30.0, 480.0, 900.0, 0.37


def script_of(text: str) -> str:
    (held,) = re.findall(r'<script type="module">(.*?)</script>', text, re.S)
    return held


def run_page_script(node: str, text: str, times: list[float], *, checked: bool = True, radius: float = VIEWER_RADIUS) -> dict:
    """The page's own script (its logic and its glue, as the page carries them) run against a stand-in for the document
    and the viewer element: it has the elements the page has, the viewer answers the camera's orbit and field of view, a
    height and a width in pixels that differ, and the current time the test sets; the load event is fired, then one
    frame for each of ``times``. Returns what the script set the camera's target and the scale bar to at each."""
    ids = sorted(Facts(text).ids - {"view"})
    stand_in = f"""
const __listeners = {{}}; const __frames = []; let __jumps = 0;
const __element = (id) => ({{ id, style: {{}}, textContent: "", value: "", max: "", checked: {json.dumps(checked)},
  scrollHeight: 0, addEventListener() {{}}, setPointerCapture() {{}}, getBoundingClientRect: () => ({{ height: 0 }}) }});
const __els = Object.fromEntries({json.dumps(ids)}.map((id) => [id, __element(id)]));
const __view = {{ clientHeight: {VIEWER_HEIGHT}, clientWidth: {VIEWER_WIDTH}, currentTime: 0, paused: false, duration: 8,
  availableAnimations: [], cameraTarget: null, cameraOrbit: null, minCameraOrbit: null,
  addEventListener(name, handler) {{ (__listeners[name] ??= []).push(handler); }},
  getCameraOrbit: () => ({{ theta: 0.1, phi: 1.2, radius: {radius!r} }}), getFieldOfView: () => {VIEWER_FOV},
  jumpCameraToGoal() {{ __jumps += 1; }} }};
globalThis.document = {{ getElementById: (id) => id === "view" ? __view : (__els[id] ?? null), body: {{ dataset: {{}} }} }};
globalThis.window = {{ innerHeight: 800 }};
globalThis.requestAnimationFrame = (frame) => {{ __frames.push(frame); }};
"""
    driver = f"""
for (const handler of __listeners.load ?? []) handler();
const __loaded = {{ target: __view.cameraTarget, orbit: __view.cameraOrbit, jumps: __jumps }};
const __steps = [];
for (const t of {json.dumps(times)}) {{
  __view.currentTime = t;
  __frames.splice(0).forEach((frame) => frame());
  const line = __els["scale-line"], label = __els["scale-label"];
  __steps.push({{ t, target: __view.cameraTarget, width: line ? line.style.width : null, label: label ? label.textContent : null }});
}}
console.log(JSON.stringify({{ loaded: __loaded, steps: __steps, listened: Object.keys(__listeners) }}));
"""
    return run_node(node, stand_in + script_of(text) + driver)


def metres_of(target: str) -> np.ndarray:
    return np.array([float(v.removesuffix("m")) for v in target.split(" ")])


def test_the_pages_glue_puts_the_camera_on_the_followed_body_at_the_time_the_viewer_is_at(two_run, node):
    result = gltf(two_run)
    text = page(result, follow="ball")
    follow = config_of(text)["follow"]
    times, positions = np.asarray(follow["times"]), np.asarray(follow["positions"])
    asked = [0.0, float(times[1]), 0.5 * float(times[1] + times[2]), float(times[3]) + 0.1 * float(times[4] - times[3]),
             float(times[-1]), float(times[-1]) + 5.0]
    held = run_page_script(node, text, asked)
    assert "load" in held["listened"]
    for step in held["steps"]:
        wanted = np.array([np.interp(step["t"], times, positions[:, axis]) for axis in range(3)])
        assert metres_of(step["target"]) == pytest.approx(wanted, rel=1e-7, abs=1e-12), step
    ball_at = model_space(result, np.array([two_balls_ball(k)[:3] for k in range(6)]))
    assert metres_of(held["steps"][1]["target"]) == pytest.approx(ball_at[1], rel=1e-7, abs=1e-12), "the viewer's time is the animation's"
    assert not np.allclose(ball_at[1], ball_at[2]) and len({round(float(v), 12) for v in ball_at[1]}) == 3, "three different axes"


def test_the_pages_glue_leaves_the_camera_where_the_user_put_it_when_follow_is_released_or_off(two_run, node):
    text = page(gltf(two_run), follow="ball")
    assert run_page_script(node, text, [0.5, 1.0], checked=False)["steps"][0]["target"] is None
    assert run_page_script(node, page(gltf(two_run)), [0.5, 1.0])["steps"][0]["target"] is None


def test_the_pages_glue_frames_the_focused_body_at_the_start(two_run, node):
    result = gltf(two_run)
    held = run_page_script(node, page(result, focus="far"), [0.0])["loaded"]
    assert metres_of(held["target"]) == pytest.approx(model_space(result, np.array([[1.0, 0.0, 0.0]]))[0], abs=1e-9)
    radius = float(held["orbit"].split(" ")[2].removesuffix("m"))
    assert radius == pytest.approx(FOCUS_FACTOR * result.bodies["far"].extent) and held["jumps"] >= 1
    assert run_page_script(node, page(result), [0.0])["loaded"]["target"] is None, "and without focus it sets none"


def test_the_pages_glue_draws_the_scale_bar_for_the_viewers_height_and_field_of_view(two_run, node):
    """The bar is the one the logic gives for the orbit's radius, the viewer's field of view and its *height* in pixels
    (its width is another number here, and the field of view is the viewer's 30 degrees, not the logic tests' 45)."""
    result = gltf(two_run)
    held = run_page_script(node, page(result), [0.0])["steps"][0]
    mpp = result.unit_m * 2.0 * VIEWER_RADIUS * math.tan(math.radians(VIEWER_FOV) / 2.0) / VIEWER_HEIGHT
    metres = 10.0 ** math.floor(math.log10(mpp * BAR_PIXELS))
    metres *= 5 if mpp * BAR_PIXELS / metres >= 5 else 2 if mpp * BAR_PIXELS / metres >= 2 else 1
    assert float(held["width"].removesuffix("px")) == pytest.approx(metres / mpp, rel=1e-9)
    assert held["label"] == expected_label(metres)
    nearer = run_page_script(node, page(result), [0.0], radius=VIEWER_RADIUS / 10.0)["steps"][0]
    assert float(nearer["width"].removesuffix("px")) != pytest.approx(float(held["width"].removesuffix("px")), rel=1e-3) \
        or nearer["label"] != held["label"], "and it follows the zoom"
    off = run_page_script(node, page(result, scale_bar=False), [0.0])["steps"][0]
    assert off["width"] is None and off["label"] is None


@pytest.mark.parametrize("planted", [
    ("view.clientHeight", "view.clientWidth"), ("view.currentTime);", "view.currentTime + 1);"),
    ('view.cameraTarget = at.map', "view.cameraTarget = [at[0], at[2], at[1]].map"), ("view.getFieldOfView()", "45"),
    ("config.unit_m, view", "config.unit_m * 10, view")], ids=["width", "time", "axes", "fov", "unit"])
def test_control_the_pages_glue_with_a_planted_fault_fails(two_run, node, planted):
    """Each is a mutant of ``_SCRIPT`` the stand-in must see: the run page's answers differ from the right ones."""
    result = gltf(two_run)
    text = page(result, follow="ball")
    assert planted[0] in text
    right = run_page_script(node, text, [0.5 * float(config_of(text)["follow"]["times"][2])])
    wrong = run_page_script(node, text.replace(*planted), [0.5 * float(config_of(text)["follow"]["times"][2])])
    assert wrong["steps"] != right["steps"]


# --- A0's motor and magnet -----------------------------------------------------------------------------------------------------


def test_a0s_shapes_are_drawn_at_the_declared_size_and_follow_the_rotor_frame(declared):
    result = gltf(declared)
    by_name = {shape.name: shape for shape in result.shapes}
    magnet = by_name["rpm.rotor_magnet"]
    assert magnet.kind == "cylinder" and magnet.source == "declared" and magnet.size_m == (MAGNET_RADIUS_M, MAGNET_LENGTH_M)
    document, binary = glb_parts(result.glb)
    index, _ = node_by_usd(document, f"{APPARATUS_ROOT}/rpm_rotor_magnet/shape")
    node = document["nodes"][index]
    mesh = document["meshes"][node["mesh"]]["primitives"][0]
    points = accessor(document, binary, mesh["attributes"]["POSITION"])  # the stage's metres: the root node scales them
    assert points[:, 2].max() - points[:, 2].min() == pytest.approx(MAGNET_LENGTH_M, rel=1e-4)
    assert np.linalg.norm(points[:, :2], axis=1).max() == pytest.approx(MAGNET_RADIUS_M, rel=1e-4)
    x, y, z, w = node["rotation"]
    along = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    assert along == pytest.approx([1.0, 0.0, 0.0], abs=1e-6), "its axis is the part's declared x"
    moved = tracks_of(result, f"{APPARATUS_ROOT}/rpm_rotor_magnet")
    recorded_m = np.array([rotor_pose(k)[:3] for k in range(5)])
    assert moved["translation"] == pytest.approx(recorded_m, rel=1e-6, abs=1e-9), "the frame's world pose, in the stage's metres"
    turn = moved["rotation"]
    assert len(turn) == 5 and not np.allclose(turn[0], turn[-1]), "it turns with the rotor"


def test_control_a_shape_that_stands_still_fails(declared, monkeypatch):
    monkeypatch.setattr(apparatus_module, "time_codes", lambda stage: [])
    result = gltf(declared)
    assert "translation" not in tracks_of(result, f"{APPARATUS_ROOT}/rpm_rotor_magnet")


def test_the_motor_is_a_marked_placeholder_on_its_axis(declared):
    result = gltf(declared)
    housing = next(shape for shape in result.shapes if shape.name == "rpm.motor")
    assert housing.placeholder
    assert housing.size_m == (0.5 * 35e-3, 35e-3), "radius half and length equal to the magnet's largest extent"
    notes = [line for line in result.lines() if "rpm.motor is drawn as a placeholder" in line]
    assert len(notes) == 1 and "which side of the rotor" in notes[0] and "the stage holds no such shape" in notes[0]
    assert all("rotor_magnet is drawn as a placeholder" not in line for line in result.lines()), "the magnet's is declared"
    assert any(path.endswith("housing_placeholder") for path in overlay_nodes(result))


def test_the_motors_housing_follows_the_base_frame_and_ends_in_the_rotors_plane(declared, a0_built):
    """The housing is drawn at the motor's *base* frame, which the recording moves 4 mm a sample along x and 1 mm along z
    (the rotor, which the magnet follows, moves otherwise), and its cylinder is offset half its length back along the
    axis from that frame (the axis is z, as the document declares)."""
    _, built = a0_built
    start = {node: {name: np.asarray(value) for name, value in fields.items()} for node, fields in built.current_state().items()}
    result = gltf(declared)
    housing = f"{APPARATUS_ROOT}/rpm_motor_housing_placeholder"
    moved = tracks_of(result, housing)
    base = np.array([base_pose(start, k) for k in range(5)])
    assert moved["translation"] == pytest.approx(base[:, :3], rel=1e-6, abs=1e-9), "the base frame's world position, each sample"
    assert not np.allclose(moved["translation"][0], moved["translation"][-1]), "and it moves"
    w, x, y, z = base[0, 3:]
    turns = moved["rotation"] if "rotation" in moved else None
    held = np.array([x, y, z, w])
    assert turns is None or all(abs(float(np.dot(row, held))) == pytest.approx(1.0, abs=1e-6) for row in turns), "the base frame's turn"
    rotor = tracks_of(result, f"{APPARATUS_ROOT}/rpm_rotor_magnet")["translation"]
    assert not np.allclose(moved["translation"], rotor, atol=1e-3), "not the rotor's"
    document, _ = glb_parts(result.glb)
    _, shape = node_by_usd(document, f"{housing}/shape")
    assert shape["translation"] == pytest.approx([0.0, 0.0, -0.5 * 35e-3], abs=1e-9), "half its length back along the axis"
    assert shape.get("rotation", [0.0, 0.0, 0.0, 1.0]) == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-9), "its cylinder along z"


def test_without_a_saved_experiment_the_magnet_is_a_marked_placeholder_by_a_stated_rule(undeclared):
    result = gltf(undeclared)
    magnet = next(shape for shape in result.shapes if shape.name == "rpm.rotor_magnet")
    assert magnet.placeholder and "no saved experiment" in magnet.rule and "1/40" in magnet.rule
    assert PLACEHOLDER_FRACTION == 1.0 / 40.0
    assert magnet.size_m[1] == pytest.approx(PLACEHOLDER_FRACTION * result.across_m, rel=1e-6), "exactly a fortieth, as said"
    assert magnet.size_m[0] == pytest.approx(magnet.size_m[1] / 2)
    assert f"{APPARATUS_ROOT}/rpm_rotor_magnet_placeholder" in overlay_nodes(result)
    said = [line for line in result.lines() if line.startswith("rpm.rotor_magnet is drawn as a placeholder")]
    assert len(said) == 1 and "no saved experiment" in said[0]
    assert any("rotor_magnet is drawn as a placeholder" in p.replace("<", "") for p in
               re.findall(r"<p class=\"muted\">([^<]*)</p>", page(result))), "and the page's notes say so"


def test_a_planted_declared_size_in_another_unit_is_not_guessed(a0_built, tmp_path):
    experiment, _ = a0_built
    run = tmp_path / "run"
    folder = a0_run(a0_built, run, saved=True)
    document = run / "experiment" / "experiment.mime.yaml"
    text = document.read_text(encoding="utf-8")
    assert text.count("unit: m") > 2
    document.write_text(text.replace("value: 0.0175\n              unit: m", "value: 17.5\n              unit: mm"), encoding="utf-8")
    magnet = next(shape for shape in gltf(folder).shapes if shape.name == "rpm.rotor_magnet")
    assert magnet.placeholder, "a size in a unit the view does not read is a placeholder, never a guess"


# --- nothing of it is written into a stage ---------------------------------------------------------------------------------------


def state_of(folder: Path) -> dict[str, tuple]:
    """Everything in ``folder``: each directory, and each file's size, modification time and sha256."""
    if not folder.exists():
        return {}
    return {str(path.relative_to(folder)): ("directory",) if path.is_dir() else
            (path.stat().st_size, path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(folder.rglob("*"))}


def view_every_way(target: Path, out: Path) -> list:
    """``target`` through ``gltf``, ``display`` and ``web``, every switch on and a body followed and framed."""
    every = {"scale_bar": True, "legend": True, "apparatus_shapes": True}
    shapes = [gltf(target, apparatus_shapes=True).shapes]
    shown = display(target, follow="umr", focus="umr", **every)
    _, result = web(target, out / f"{target.parent.name}_{target.name}.html", follow="umr", focus="umr", **every)
    return [*shapes, shown.result.shapes, result.shapes]


def watched(saved_a0: Path, declared: Path, cache: Path) -> dict[str, Path]:
    return {"saved folder": saved_a0, "run folder": declared, "asset cache": cache}


def targets_of(saved_a0: Path, declared: Path) -> list[Path]:
    """The four ways to name A0: its saved folder, a run folder, the recording, and the bare stage (in a saved folder, where
    the stage is the root layer a careless draw would edit; in a run folder it is a sublayer of the recording)."""
    return [saved_a0, declared, declared / "recording.usdc", saved_a0 / "scene" / "scene.usda"]


def test_a0s_saved_folder_run_folder_and_cache_are_byte_for_byte_the_same_after_a_view(saved_a0, declared, cache, a0_built, tmp_path):
    """The headline guarantee: nothing of the shapes is written into a stage, a recording, a folder or the cache. Each of
    the four ways to name A0 is viewed through ``gltf``, ``display`` and ``web`` with every switch on, and every file and
    directory of both folders and of the cache is the same after (its size, its modification time and its hash)."""
    _, built = a0_built
    before = {name: state_of(folder) for name, folder in watched(saved_a0, declared, cache).items()}
    assert before["saved folder"] and before["run folder"] and before["asset cache"], "each holds files, or the comparison is empty"
    for target in targets_of(saved_a0, declared):
        for shapes in view_every_way(target, tmp_path):
            assert [shape.name for shape in shapes] == ["rpm.rotor_magnet", "rpm.motor"], f"the shapes were drawn for {target}"
    after = {name: state_of(folder) for name, folder in watched(saved_a0, declared, cache).items()}
    for name in before:
        assert after[name] == before[name], f"the {name} changed"
    stage = saved_a0 / "scene" / "scene.usda"
    assert hashlib.sha256(stage.read_bytes()).hexdigest() == hashlib.sha256(built.stage_text.encode("utf-8")).hexdigest(), \
        "the saved stage is the built stage, byte for byte"
    opened = Usd.Stage.Open(str(stage))
    assert [p.GetName() for p in opened.GetPseudoRoot().GetChildren()] == ["World"], "the stage has the one root"


def planted_in_saved_stage(saved_a0: Path, declared: Path, cache: Path):
    stage = saved_a0 / "scene" / "scene.usda"
    return lambda: stage.open("ab").write(b"\n# planted\n")


def planted_in_run_stage(saved_a0: Path, declared: Path, cache: Path):
    stage = declared / "experiment" / "scene" / "scene.usda"
    return lambda: stage.open("ab").write(b"\n# planted\n")


def planted_in_cache(saved_a0: Path, declared: Path, cache: Path):
    return lambda: (cache / "baked" / "planted.bin").write_bytes(b"planted")


def planted_over_cache_file(saved_a0: Path, declared: Path, cache: Path):
    return lambda: (cache / "baked" / "held.bin").write_bytes(b"a baked mesh, as a view must leave it")  # same bytes, new time


def planted_directory_in_cache(saved_a0: Path, declared: Path, cache: Path):
    return lambda: (cache / "planted").mkdir()


@pytest.mark.parametrize("plant", [planted_in_saved_stage, planted_in_run_stage, planted_in_cache, planted_over_cache_file,
                                   planted_directory_in_cache],
                         ids=["saved stage", "run stage", "cache file", "cache file rewritten", "cache directory"])
def test_control_a_write_during_a_view_into_the_saved_folder_the_run_folder_or_the_cache_is_seen(plant, saved_a0, declared, cache, tmp_path, monkeypatch):
    """A write made while the view is drawn (into either folder's stage, or into the cache) changes the state the test above
    compares, on every run: the folders are put back after."""
    kept = {name: {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}
            for name, folder in watched(saved_a0, declared, cache).items()}
    before = {name: state_of(folder) for name, folder in watched(saved_a0, declared, cache).items()}
    write = plant(saved_a0, declared, cache)
    original = export_module.draw

    def writing(stage, source, scene_m):
        found = original(stage, source, scene_m)
        write()
        return found

    monkeypatch.setattr(export_module, "draw", writing)
    try:
        gltf(saved_a0)
        after = {name: state_of(folder) for name, folder in watched(saved_a0, declared, cache).items()}
        assert after != before
    finally:
        monkeypatch.undo()
        for name, files in kept.items():
            for path, data in files.items():
                path.write_bytes(data)
        for extra in (cache / "planted", cache / "baked" / "planted.bin"):
            if extra.is_dir():
                extra.rmdir()
            elif extra.exists():
                extra.unlink()


def test_control_shapes_planted_into_world_are_seen(declared, monkeypatch):
    """The test of ``/World`` above compares the stage's own prims with the shapes on and off: a shape written below
    ``/World`` would add a node there."""
    original = apparatus_module.APPARATUS_ROOT
    import mime.viz._export as export_module

    real = export_module.draw

    def inside(stage, source, scene_m):
        monkeypatch.setattr(apparatus_module, "APPARATUS_ROOT", "/World/ApparatusView")
        try:
            return real(stage, source, scene_m)
        finally:
            monkeypatch.setattr(apparatus_module, "APPARATUS_ROOT", original)

    monkeypatch.setattr(export_module, "draw", inside)
    assert world_nodes(gltf(declared)) != world_nodes(gltf(declared, apparatus_shapes=False))


# --- the legend ----------------------------------------------------------------------------------------------------------------


def legend_text(text: str) -> str:
    (held,) = re.findall(r'<li data-prim="[^"]*"><span class="swatch"[^>]*></span>([^<]*)</li>', text)
    return held.replace("&#x27;", "'")


def test_the_legend_says_what_the_arrow_is_its_colour_and_its_scale(declared):
    result = gltf(declared)
    text = page(result)
    stage = Usd.Stage.Open(str(declared / "experiment" / "scene" / "scene.usda"))
    prim = stage.GetPrimAtPath(ARROW)
    per_tesla = prim.GetAttribute("mime:vector:metres_per_unit").Get()
    colour = prim.GetChild("arrow").GetAttribute("primvars:displayColor").Get()[0]
    line = legend_text(text)
    assert "rpm.magnetic.magnetic_torque_umr: magnetic flux density (T)" in line
    assert f"drawn {length_text(per_tesla)} per T" in line
    assert f"never longer than {length_text(2 * HELIX_LENGTH_M)}" in line, "the body's cap, as the stage's body says it"
    assert f"(a vector of {2 * HELIX_LENGTH_M / per_tesla:.3g} T or more is drawn at that length" in line, "the field that reaches the cap"
    assert "capped in this view" not in line, "at A0's own field it is not shortened"
    (swatch,) = re.findall(r'class="swatch" style="background: rgb\((\d+), (\d+), (\d+)\)"', text)
    assert tuple(int(v) for v in swatch) == tuple(round(float(v) * 255) for v in colour)


def test_the_legend_says_when_the_arrow_cap_shortened_the_arrow(capped):
    result = gltf(capped)
    (held,) = result.capped.values()
    line = legend_text(page(result))
    assert f"capped in this view: its longest is drawn {length_text(held.drawn_m)}, not {length_text(held.would_be_m)}" in line
    assert held.would_be_m > held.drawn_m > 0.0


def test_control_a_legend_with_the_wrong_scale_fails(declared, monkeypatch):
    result = gltf(declared)
    wrong = dataclasses.replace(result, arrows=[dataclasses.replace(key, metres_per_unit=key.metres_per_unit * 10.0)
                                                for key in result.arrows])
    stage = Usd.Stage.Open(str(declared / "experiment" / "scene" / "scene.usda"))
    per_tesla = stage.GetPrimAtPath(ARROW).GetAttribute("mime:vector:metres_per_unit").Get()
    assert f"drawn {length_text(per_tesla)} per T" in legend_text(page(result))
    assert f"drawn {length_text(per_tesla)} per T" not in legend_text(page(wrong))


def test_a_scene_with_no_arrows_has_no_legend_and_says_so(ball_run):
    text = page(gltf(ball_run))
    assert "legend" not in Facts(text).ids and config_of(text)["legend"] is False
    assert Facts(text).toggles["legend"] == "on" and "this scene draws no arrows" in text


# --- the command -----------------------------------------------------------------------------------------------------------------


def test_the_command_has_a_flag_for_each_switch(declared, tmp_path):
    out = tmp_path / "a0.html"
    assert main(["visuals", "web", str(declared), "--out", str(out)]) == 0
    default = out.read_text(encoding="utf-8")
    assert default == page(gltf(declared))
    flags = {"--no-scale-bar": "scale-bar", "--no-legend": "legend"}
    for flag, element in flags.items():
        assert main(["visuals", "web", str(declared), "--out", str(out), flag]) == 0
        assert changed(default, out.read_text(encoding="utf-8")) >= {element}
        assert element not in Facts(out.read_text(encoding="utf-8")).ids
    assert main(["visuals", "web", str(declared), "--out", str(out), "--no-apparatus-shapes"]) == 0
    assert Facts(out.read_text(encoding="utf-8")).toggles["apparatus_shapes"] == "off"
    assert main(["visuals", "web", str(declared), "--out", str(out), "--focus", "umr", "--follow", "umr"]) == 0
    held = config_of(out.read_text(encoding="utf-8"))
    assert held["focus"]["name"] == "umr" and held["follow"]["name"] == "umr"


def test_a_view_of_a0_with_everything_on_is_under_the_notebooks_output_limit(declared):
    shown = display(declared, follow="umr", focus="umr")
    assert len(shown.html.encode("utf-8")) < OUTPUT_WARNING_BYTES
