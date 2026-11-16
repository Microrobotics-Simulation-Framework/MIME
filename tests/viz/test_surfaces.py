"""A deforming surface carrying fields, recorded and viewed: ``toy.indented_sheet`` (a sheet indented by a dimple whose
depth grows at a known rate) run with its stage recorded, its recording read back, its GLB read by an independent
reader, and its page's picker run in Node.

**What this bounds.** That every sample's points and fields in ``recording.usdc`` are the node's values at that step as
float32 bits (points over the stage's metres per unit), in float32 and in float64, on a stage in metres and one in
millimetres; and where the surface is drawn decimated, each sample is its clusters' means (to one float32 unit in the
last place, the double-precision sum's rounding) and a field declared ``max`` its clusters' maxima exactly. That the
same declarations give the same bytes for the stage, the recording, the GLB and the page in two processes. That the
recording's time codes are its samples' indices at ``1 / (every dt)`` a second. That the GLB holds one primitive with a
morph target for each drawn sample of each field, each with a POSITION and a four-component COLOR_0, a minimum and a
maximum on every POSITION accessor, an animation for each field, and at most ``MORPH_TARGETS`` targets. That the page
carries the field legend and picker only where a surface's field is drawn, and states its own size; and that the picker
keeps the view's time and its play state, though the viewer sets its time to zero when the animation changes (as
model-viewer 4.3.1 does, measured in the spike). **What it does not.** That a browser draws it: CI has no WebGL. The
spike drew it in headless Chrome (SwiftShader) and the Khronos validator (2.0.0-dev.3.10) found no error or warning;
the validator is a local check (Node and its package are not in CI's lanes), run on this branch's GLBs before merge.

**Controls.** Each fails on every run (the inputs are deterministic): a field written one sample late, and the points;
the wall clock in the recording's layer data; the rate doubled and halved; a target's POSITION without its minimum and
maximum, and a COLOR_0 of three components; the picker without its line that puts the time back."""

import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import jax
import numpy as np
import pygltflib
import pytest
from pxr import Sdf, Usd

import mime.testing.toys as toys
import mime.viz._web as web_module
from mime.build._surfaces import clustered
from mime.experiment import Experiment
from mime.results import RunResult
from mime.scene import Scene, Z
from mime.things import Body, State
from mime.units import m, m_per_s, m_per_s2, s
from mime.viz import _recording, gltf, page
from mime.viz._gltf import read_glb
from mime.viz._gltf._converter import MORPH_TARGETS

DT = 0.01
SURFACE = "/World/sheet/sheet"


def sheet_experiment(units: str = "m", **declared) -> Experiment:
    experiment = Experiment("sheet", scene=Scene.new(up=Z, units=units), gravity=m_per_s2(0.0), dt=s(DT))
    experiment.add_body(Body("sheet", dynamics=toys.ToyIndentedSheet(speed=m_per_s(5e-4), **declared),
                             initial=State(position=m((0.0, 0.0, 0.0)))))
    return experiment


def run_sheet(folder: Path, *, x64: bool, samples: int, every: int = 1, units: str = "m", **declared):
    """The sheet run to ``folder``, its stage sampled every ``every`` steps, its history kept; and its state at the
    start."""
    with jax.enable_x64(x64), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = sheet_experiment(units, **declared).build()
        start = {node: {name: np.asarray(value) for name, value in fields.items()}
                 for node, fields in built.current_state().items()}
        built.run_to(folder, s(samples * every * DT), stage_every=s(every * DT), history=True)
    return built, start


def recorded(folder: Path):
    layer = Sdf.Layer.FindOrOpen(str(folder / "recording.usdc"))
    stage = Usd.Stage.Open(str(folder / "recording.usdc"))
    return layer, stage


def node_history(folder: Path) -> tuple[str, dict[str, np.ndarray]]:
    history = RunResult.read(folder).history
    (node,) = [name for name, fields in history.items() if "points" in fields]
    return node, {name: np.asarray(value) for name, value in history[node].items()}


def bits(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def readback_mismatches(folder: Path, start: dict) -> dict[str, list[int]]:
    """The time codes whose points or field values in the recording are not the node's at that step, as float32 bits:
    code 0 against the state the run began from, every later code against the history at its step."""
    layer, stage = recorded(folder)
    steps = list(layer.customLayerData["mime:recording"]["steps"])
    meters = float(layer.pseudoRoot.GetInfo("metersPerUnit"))
    node, history = node_history(folder)
    prim = stage.GetPrimAtPath(SURFACE)
    found: dict[str, list[int]] = {"points": [], "depth": [], "rate": []}
    for code, step in enumerate(steps):
        held = {name: np.asarray(start[node][name]) if step == 0 else history[name][step - 1] for name in found}
        want = {"points": (held["points"].astype(np.float64) / meters).astype(np.float32),
                "depth": held["depth"].astype(np.float32), "rate": held["rate"].astype(np.float32)}
        got = {"points": prim.GetAttribute("points").Get(Usd.TimeCode(code)),
               "depth": prim.GetAttribute("primvars:mime:depth").Get(Usd.TimeCode(code)),
               "rate": prim.GetAttribute("primvars:mime:rate").Get(Usd.TimeCode(code))}
        for name in found:
            if not np.array_equal(bits(got[name]).reshape(-1), bits(want[name]).reshape(-1)):
                found[name].append(code)
    return found


#: float32 in metres every step, float64 in millimetres every third step.
ARMS = [(False, "m", 1), (True, "mm", 3)]


@pytest.mark.parametrize(("x64", "units", "every"), ARMS, ids=["float32-m", "float64-mm"])
def test_every_sample_reads_back_as_the_nodes_values(tmp_path, x64, units, every):
    _, start = run_sheet(tmp_path / "run", x64=x64, samples=12, every=every, units=units, count=9)
    _, history = node_history(tmp_path / "run")
    assert history["points"].dtype == (np.float64 if x64 else np.float32)
    assert readback_mismatches(tmp_path / "run", start) == {"points": [], "depth": [], "rate": []}
    layer, _ = recorded(tmp_path / "run")
    assert len(layer.customLayerData["mime:recording"]["steps"]) == 13 and float(history["depth"].max()) > 0.0


@pytest.mark.parametrize("planted", ["depth", "points"])
def test_control_a_sample_written_one_late_is_caught(tmp_path, monkeypatch, planted):
    real = _recording.StageSamples.stacked

    def late(self, key):
        held = real(self, key)
        return np.concatenate([held[:1], held[:-1]]) if key[1] == planted and len(held) > 1 else held

    monkeypatch.setattr(_recording.StageSamples, "stacked", late)
    _, start = run_sheet(tmp_path / "run", x64=False, samples=12, count=9)
    found = readback_mismatches(tmp_path / "run", start)
    assert len(found[planted]) == 12, found  # every code after the first


def test_a_decimated_sample_is_its_clusters_means_and_a_max_field_its_maxima(tmp_path):
    built, start = run_sheet(tmp_path / "run", x64=False, samples=3, count=129, peak=True)
    (points_entry,) = [e for e in built.binding().entries if e.kind == "points"]
    assert points_entry.clusters is not None
    clusters = np.asarray(points_entry.clusters)
    rest, faces = toys.sheet(1e-3, 129)
    assert np.array_equal(clusters, clustered(rest, faces).clusters)
    members = [np.flatnonzero(clusters == k) for k in range(int(points_entry.count or 0))]
    layer, stage = recorded(tmp_path / "run")
    steps = list(layer.customLayerData["mime:recording"]["steps"])
    node, history = node_history(tmp_path / "run")
    prim = stage.GetPrimAtPath(SURFACE)
    for code, step in enumerate(steps):
        held = {name: np.asarray(start[node][name]) if step == 0 else history[name][step - 1]
                for name in ("points", "depth", "rate")}
        means = np.array([[math.fsum(held["points"][index, axis].astype(np.float64)) / len(index) for axis in range(3)]
                          for index in members]).astype(np.float32)
        got = np.asarray(prim.GetAttribute("points").Get(Usd.TimeCode(code)), dtype=np.float32)
        assert int(np.max(np.abs(bits(got).astype(np.int64) - bits(means).astype(np.int64)))) <= 1, code
        peaks = np.array([held["depth"][index].max() for index in members]).astype(np.float32)
        assert np.array_equal(bits(prim.GetAttribute("primvars:mime:depth").Get(Usd.TimeCode(code))), bits(peaks))
        rates = np.array([math.fsum(held["rate"][index].astype(np.float64)) / len(index) for index in members])
        drawn = np.asarray(prim.GetAttribute("primvars:mime:rate").Get(Usd.TimeCode(code)), dtype=np.float32)
        assert int(np.max(np.abs(bits(drawn).astype(np.int64) - bits(rates).astype(np.int64)))) <= 1, code


# --- the time base ----------------------------------------------------------------------------------------------------


def time_base_problems(layer, dt: float, every: int) -> list[str]:
    steps = list(layer.customLayerData["mime:recording"]["steps"])
    rate = float(layer.timeCodesPerSecond)
    found = []
    if float(layer.framesPerSecond) != rate:
        found.append("framesPerSecond is not timeCodesPerSecond")
    if sorted(layer.ListAllTimeSamples()) != [float(code) for code in range(len(steps))]:
        found.append("the time samples are not at 0 .. n - 1")
    if not math.isclose(rate, 1.0 / (every * dt), rel_tol=1e-12):
        found.append(f"the rate {rate} is not 1 / (every dt)")
    found += [f"code {code} is {code / rate} s, not step {step}'s" for code, step in enumerate(steps)
              if not math.isclose(code / rate, (step - steps[0]) * dt, rel_tol=1e-9, abs_tol=1e-12)][:1]
    return found


@pytest.fixture(scope="module")
def every_third(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sheet") / "run"
    run_sheet(folder, x64=False, samples=8, every=3, count=9)
    return folder


def test_the_time_codes_are_the_samples_at_one_over_every_dt(every_third):
    layer, _ = recorded(every_third)
    assert time_base_problems(layer, DT, 3) == []


@pytest.mark.parametrize("factor", [2.0, 0.5])
def test_control_a_rate_off_by_a_factor_is_caught(every_third, factor):
    layer, _ = recorded(every_third)
    copied = Sdf.Layer.CreateAnonymous(".usda")
    copied.TransferContent(layer)
    copied.timeCodesPerSecond = copied.framesPerSecond = layer.timeCodesPerSecond * factor
    assert time_base_problems(copied, DT, 3) != []


# --- the GLB ----------------------------------------------------------------------------------------------------------


def glb_problems(document: dict, morphs: list[dict]) -> list[str]:
    """Where the GLB's surfaces are not as the converter's record says: a primitive's targets, each with a POSITION and
    a four-component COLOR_0, a minimum and a maximum on every POSITION accessor, one animation per field."""
    found = []
    accessors = document["accessors"]
    primitives = [p for mesh in document["meshes"] for p in mesh["primitives"] if "targets" in p]
    if len(primitives) != len(morphs):
        found.append(f"{len(primitives)} morphed primitives, not {len(morphs)}")
    for primitive, morph in zip(primitives, morphs, strict=False):
        expected = morph["drawn"] * max(1, len(morph["fields"]))
        if len(primitive["targets"]) != expected or expected > MORPH_TARGETS:
            found.append(f"{len(primitive['targets'])} targets, not {expected} (at most {MORPH_TARGETS})")
        positions = [primitive["attributes"]["POSITION"], *(t.get("POSITION") for t in primitive["targets"])]
        colours = [primitive["attributes"].get("COLOR_0"), *(t.get("COLOR_0") for t in primitive["targets"])]
        if None in positions or (morph["fields"] and None in colours):
            found.append("a target without its POSITION or its COLOR_0")
        found += [f"POSITION accessor {k} has no min or max" for k in positions
                  if k is not None and not {"min", "max"} <= set(accessors[k])]
        found += [f"COLOR_0 accessor {k} is {accessors[k]['type']}, not VEC4" for k in colours
                  if k is not None and accessors[k]["type"] != "VEC4"]
    names = [animation["name"] for animation in document.get("animations", [])]
    fields = [field["name"] for field in morphs[0]["fields"]] if morphs else []
    if fields and names != ["run", *(f"run:{name}" for name in fields[1:])]:
        found.append(f"animations {names}")
    return found


@pytest.fixture(scope="module")
def viewed(every_third):
    return gltf(every_third)


def test_the_glb_holds_a_morph_target_per_sample_and_field(viewed):
    document, _ = read_glb(viewed.glb)
    assert glb_problems(document, viewed.morphs) == []
    (morph,) = viewed.morphs
    assert morph["vertices"] == 81 and morph["samples"] == morph["drawn"] == 9 and morph["targets"] == 18
    assert [field["name"] for field in morph["fields"]] == ["depth", "rate"] and viewed.animations == ["run", "run:rate"]
    independent = pygltflib.GLTF2.load_from_bytes(viewed.glb)
    (primitive,) = [p for mesh in independent.meshes for p in mesh.primitives if p.targets]
    assert len(primitive.targets) == 18 and len(independent.animations) == 2


@pytest.mark.parametrize("planted", ["minmax", "vec3"])
def test_control_a_planted_fault_in_the_glb_is_caught(viewed, planted):
    document, _ = read_glb(viewed.glb)
    held = copy.deepcopy(document)
    target = [p for mesh in held["meshes"] for p in mesh["primitives"] if "targets" in p][0]["targets"][1]
    if planted == "minmax":
        accessor = held["accessors"][target["POSITION"]]
        del accessor["min"], accessor["max"]
    else:
        held["accessors"][target["COLOR_0"]]["type"] = "VEC3"
    assert glb_problems(held, viewed.morphs) != []


def test_a_recording_of_more_samples_than_a_view_holds_is_drawn_at_the_cap(tmp_path):
    run_sheet(tmp_path / "run", x64=False, samples=140, count=3)
    result = gltf(tmp_path / "run")
    (morph,) = result.morphs
    assert morph["samples"] == 141 and morph["drawn"] == MORPH_TARGETS // 2 and morph["targets"] == MORPH_TARGETS
    assert any(f"drawn at {MORPH_TARGETS // 2} of its 141 recorded samples" in line for line in result.lines())


# --- the page ---------------------------------------------------------------------------------------------------------


def config_of(text: str) -> dict:
    (held,) = re.findall(r"const config = (\{.*?\});\n", text, re.S)
    return json.loads(held)


def test_the_page_carries_the_field_legend_and_states_its_size(viewed):
    text = page(viewed)
    keys = config_of(text)["fields"]
    assert [key["name"] for key in keys] == ["depth", "rate"] and [key["animation"] for key in keys] == ["run", "run:rate"]
    assert 'id="fields"' in text and 'id="field"' in text and 'animation-crossfade-duration="0"' in text
    (said,) = re.findall(r'id="surface-size">([^<]*)</p>', text)
    assert "sheet/sheet has 81 vertices drawn at 9 samples, 729 vertex-frames" in said
    assert said.endswith(f"This page is {len(text.encode('utf-8')) / 1e6:.2f} MB.")


def test_a_page_with_no_surface_carries_none_of_its_parts(tmp_path):
    experiment = Experiment("ball", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(DT))
    experiment.add_body(Body("ball", dynamics=toys.ToyBody(), initial=State(position=m((0.0, 0.0, 0.0)),
                                                                          velocity=m_per_s((1e-3, 0.0, 0.0)))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        experiment.build().run_to(tmp_path / "run", s(0.05), stage_every=s(DT))
    text = page(gltf(tmp_path / "run"))
    assert "fields" not in config_of(text)
    for part in ('id="fields"', "animation-crossfade-duration", "surface-size", "#fields", "picker"):
        assert part not in text, part


# --- determinism, in two processes ------------------------------------------------------------------------------------

WRITER = """
import hashlib, json, os, sys, time, warnings
from pathlib import Path
sys.path.insert(0, {tests!r})
import mime.testing.toys as toys
from mime.viz import _recording, gltf, page
from test_surfaces import run_sheet
if os.environ.get("MIME_PLANT_CLOCK") == "1":
    real = _recording.recording_layer
    def clocked(samples, settings, **kw):
        layer = real(samples, settings, **kw)
        layer.customLayerData = {{**dict(layer.customLayerData), "written_at": time.time()}}
        return layer
    _recording.recording_layer = clocked
folder = Path(sys.argv[1])
run_sheet(folder, x64=False, samples=6, count=9)
result = gltf(folder)
digest = lambda data: hashlib.sha256(data).hexdigest()
print(json.dumps({{"stage": digest((folder / "experiment" / "scene" / "scene.usda").read_bytes()),
                  "recording": digest((folder / "recording.usdc").read_bytes()), "glb": digest(result.glb),
                  "page": digest(page(result).encode())}}))
"""


def writer(folder: Path, plant: bool = False) -> subprocess.Popen[str]:
    """A process of its own that writes the sheet's run, its GLB and its page, and prints their digests."""
    environment = {**os.environ, "JAX_PLATFORMS": "cpu", "MIME_PLANT_CLOCK": "1" if plant else "0"}
    environment.pop("JAX_ENABLE_X64", None)
    return subprocess.Popen([sys.executable, "-c", WRITER.format(tests=str(Path(__file__).parent)), str(folder)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)


def digests(process: subprocess.Popen[str]) -> dict[str, str]:
    out, err = process.communicate(timeout=600)
    assert process.returncode == 0, err[-2000:]
    return json.loads(out.strip().splitlines()[-1])


def test_the_same_declarations_give_the_same_bytes_in_two_processes(tmp_path):
    running = [writer(tmp_path / "a"), writer(tmp_path / "b"), writer(tmp_path / "c", plant=True)]
    first, second, planted = (digests(process) for process in running)
    assert first == second
    assert planted["recording"] != first["recording"], "the control: the wall clock in the layer's data is caught"
    assert planted["stage"] == first["stage"]


# --- the picker, in Node ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def node():
    found = shutil.which("node")
    if found is None:
        if os.environ.get("CI"):
            pytest.fail("Node is on every CI runner: the page's picker is run in it")
        pytest.skip("Node is not installed here, so the page's picker is not run (it is run on CI)")
    return found


def script_of(text: str) -> str:
    (held,) = re.findall(r'<script type="module">(.*?)</script>', text, re.S)
    return held


def pick(node: str, text: str, at: float, paused: bool, index: int) -> dict:
    """The page's own script against a stand-in viewer that, as model-viewer 4.3.1 does, sets its time to zero when its
    animation changes: the picker switched to field ``index`` with the view at ``at``, paused or playing."""
    stand_in = f"""
const __handlers = {{}};
const __element = (id) => ({{ id, style: {{}}, textContent: "", value: "", max: "", checked: true, selectedIndex: 0,
  scrollHeight: 0, addEventListener(name, handler) {{ (__handlers[id + ":" + name] ??= []).push(handler); }},
  setPointerCapture() {{}}, getBoundingClientRect: () => ({{ height: 0 }}) }});
const __els = Object.fromEntries(["field", "field-low", "field-high", "field-said", "play"].map((id) => [id, __element(id)]));
let __animation = "run";
const __view = {{ clientHeight: 480, clientWidth: 900, currentTime: {at!r}, paused: {json.dumps(paused)}, duration: 8,
  availableAnimations: [], updateComplete: Promise.resolve(true),
  get animationName() {{ return __animation; }},
  set animationName(name) {{ __animation = name; this.currentTime = 0; }},
  play() {{ this.paused = false; }}, pause() {{ this.paused = true; }},
  addEventListener() {{}}, getCameraOrbit: () => ({{ theta: 0, phi: 1, radius: 1 }}), getFieldOfView: () => 30,
  jumpCameraToGoal() {{}} }};
globalThis.document = {{ getElementById: (id) => id === "view" ? __view : (__els[id] ?? null), body: {{ dataset: {{}} }} }};
globalThis.window = {{ innerHeight: 800 }};
globalThis.requestAnimationFrame = () => {{}};
"""
    driver = f"""
__els.field.selectedIndex = {index};
for (const handler of __handlers["field:change"] ?? []) await handler();
console.log(JSON.stringify({{ time: __view.currentTime, paused: __view.paused, animation: __animation,
  field: document.body.dataset.field, low: __els["field-low"].textContent }}));
"""
    done = subprocess.run([node, "--input-type=module"], input=stand_in + script_of(text) + driver, capture_output=True,
                          text=True, timeout=60, check=True)
    return json.loads(done.stdout)


def test_the_picker_keeps_the_time_and_the_play_state(viewed, node):
    text = page(viewed)
    keys = config_of(text)["fields"]
    paused = pick(node, text, 0.4, True, 1)
    assert paused == {"time": 0.4, "paused": True, "animation": "run:rate", "field": "rate", "low": keys[1]["low_text"]}
    playing = pick(node, text, 0.25, False, 0)
    assert playing["time"] == 0.25 and playing["paused"] is False and playing["animation"] == "run"


def test_control_the_picker_without_its_time_restore_loses_the_time(viewed, node):
    text = page(viewed)
    line = "  view.currentTime = at;\n"
    assert text.count(line) == 1 and line in web_module._FIELDS_SCRIPT
    assert pick(node, text.replace(line, ""), 0.4, True, 1)["time"] == 0

