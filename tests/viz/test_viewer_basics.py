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
stage each fail theirs. Each control is a call that must return a problem, and so fails the test that holds it."""

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
import mime.viz._web as web_module
from mime.build import ArgumentRefused
from mime.cli import main
from mime.saved import dump_document
from mime.viz import display, gltf, page, web
from mime.viz._apparatus import APPARATUS_ROOT
from mime.viz._export import length_text
from mime.viz._web import BAR_PIXELS, FOCUS_FACTOR, OUTPUT_WARNING_BYTES

ARROW = "/World/rpm/magnetic/magnetic_torque_umr"
HELIX_LENGTH_M = 7.47e-3
MAGNET_RADIUS_M, MAGNET_LENGTH_M = 17.5e-3, 20e-3


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))


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
        held = with_pose(held, "umr", np.array([1e-3 * k, 0.0, -1e-3, 1, 0, 0, 0]))
        held[key.node][key.field] = np.zeros(3) if k == 0 else np.array([0.0, tesla, 0.0])
        return held

    return recorded(built, folder, 5, state_at).parent


@pytest.fixture(scope="module")
def declared(a0_built, tmp_path_factory):
    folder = tmp_path_factory.mktemp("declared") / "run"
    return a0_run(a0_built, folder, saved=True)


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


def test_the_notes_say_what_is_on_and_what_each_switch_does(declared):
    text = page(gltf(declared), focus="umr", follow="umr")
    assert Facts(text).toggles == {name: "on" for name in ("scale_bar", "legend", "apparatus_shapes", "focus", "follow")}
    for said in ("scale_bar=False or --no-scale-bar", "legend=False or --no-legend", "apparatus_shapes=False or --no-apparatus-shapes",
                 "focus=None", "follow=None"):
        assert said in text


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
    done = subprocess.run([node, "--input-type=module", "-e", program], capture_output=True, text=True, timeout=60,
                          check=True)
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
    return [f"frame {k}: {t} != {e}" for k, (t, e) in enumerate(zip(targets, expected))
            if not np.allclose(t, e, rtol=1e-7, atol=1e-12)]


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
    monkeypatch.setattr(apparatus_module, "_time_codes", lambda stage: [])
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


def test_without_a_saved_experiment_the_magnet_is_a_marked_placeholder_by_a_stated_rule(undeclared):
    result = gltf(undeclared)
    magnet = next(shape for shape in result.shapes if shape.name == "rpm.rotor_magnet")
    assert magnet.placeholder and "no saved experiment" in magnet.rule and "1/40" in magnet.rule
    assert 0.0 < magnet.size_m[1] < result.across_m / 20.0 and magnet.size_m[0] == pytest.approx(magnet.size_m[1] / 2)
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


def unchanged(folder: Path, before: dict[str, str]) -> bool:
    return digests(folder) == before


def test_a0s_saved_stage_and_its_folder_are_byte_for_byte_the_same_after_a_view(declared, a0_built, tmp_path):
    _, built = a0_built
    before, stage_hash = digests(declared), hashlib.sha256(built.stage_text.encode("utf-8")).hexdigest()
    result = gltf(declared)
    page(result, follow="umr", focus="umr")
    display(declared, follow="umr")
    web(declared, tmp_path / "a0.html")
    assert result.shapes and unchanged(declared, before)
    stage = declared / "experiment" / "scene" / "scene.usda"
    assert hashlib.sha256(stage.read_bytes()).hexdigest() == hashlib.sha256(built.stage_text.encode("utf-8")).hexdigest() == stage_hash
    opened = Usd.Stage.Open(str(stage))
    assert [p.GetName() for p in opened.GetPseudoRoot().GetChildren()] == ["World"], "the stage has the one root"


def test_control_a_planted_write_into_the_stage_is_seen(declared, monkeypatch):
    stage = declared / "experiment" / "scene" / "scene.usda"
    original_bytes, before = stage.read_bytes(), digests(declared)
    original = apparatus_module.draw

    def writing(stage_opened, source, scene_m):
        found = original(stage_opened, source, scene_m)
        with stage.open("ab") as held:
            held.write(b"\n# planted\n")
        return found

    import mime.viz._export as export_module

    monkeypatch.setattr(export_module, "draw", writing)
    try:
        gltf(declared)
        assert not unchanged(declared, before)
    finally:
        stage.write_bytes(original_bytes)
    assert unchanged(declared, before)


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
