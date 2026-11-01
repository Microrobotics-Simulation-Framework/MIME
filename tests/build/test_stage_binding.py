"""The stage binding, the graph's topology and ``run_to``'s hook, on A0's apparatus (``chain_cases.kinematic_rpm``):
every prim the binding names is in the stage, every field it moves is in the state at its shape, and each chain frame's
prim, placed in the world, stands where the state puts the frame at the start; the topology is the built graph in
MIME's names, as plain JSON; a saved run streams through its hook. The controls: a binding with one prim, one field or
one row planted wrong is caught, each by its own line."""

import dataclasses
import json
import warnings

import numpy as np
import pytest
from chain_cases import kinematic_rpm
from live_cases import built as rig
from live_cases import steered
from pxr import Sdf, Usd, UsdGeom

from mime.build import StageBinding
from mime.units import ms


@pytest.fixture(scope="module")
def built():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return kinematic_rpm().build()


def prims_of(text: str) -> list[str]:
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(text)
    stage = Usd.Stage.Open(layer)
    return [str(prim.GetPath()) for prim in stage.Traverse()]


def test_every_bound_prim_is_in_the_stage_and_every_field_in_the_state(built):
    binding = built.binding()
    names = [entry.name for entry in binding.entries]
    assert names[0] == "umr" and sum(name.startswith("rpm.arm.") for name in names) == 27
    assert {"rpm.motor.base", "rpm.motor.rotor", "rpm.arm.link_6"} <= set(names)
    assert [entry.kind for entry in binding.entries].count("vector") == 1
    assert binding.problems(prims_of(built.stage_text), built.current_state()) == []
    assert json.loads(json.dumps(binding.record()))["format"] == "mime.binding/1"


@pytest.mark.parametrize("planted", ["prim", "field", "row"])
def test_a_planted_wrong_mapping_is_caught(built, planted):
    binding = built.binding()
    entry = next(e for e in binding.entries if e.name == "rpm.arm.link_3")
    wrong = {"prim": dataclasses.replace(entry, prim="/World/rpm/arm/link_33"),
             "field": dataclasses.replace(entry, field="frame_pose"),
             "row": dataclasses.replace(entry, index=27)}[planted]
    held = StageBinding(tuple(wrong if e is entry else e for e in binding.entries))
    found = held.problems(prims_of(built.stage_text), built.current_state())
    assert len(found) == 1 and found[0].startswith("rpm.arm.link_3:")


def test_each_frames_prim_stands_in_the_world_where_the_state_puts_it_at_the_start(built):
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(built.stage_text)
    stage = Usd.Stage.Open(layer)
    cache = UsdGeom.XformCache()
    state = built.current_state()
    worst = 0.0
    for entry in built.binding().entries:
        if entry.kind != "pose" or entry.index is None:
            continue
        world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(entry.prim)).ExtractTranslation()
        expected = np.asarray(state[entry.node][entry.field][entry.index], dtype=np.float64)[:3]
        worst = max(worst, float(np.max(np.abs(np.asarray(world) - expected))))
        assert UsdGeom.Xformable(stage.GetPrimAtPath(entry.prim)).GetResetXformStack()
    # The frames' poses at the start in single precision against the stage's in double: within float32's rounding of
    # the arm's reach.
    assert worst <= 1e-6


def test_the_topology_is_the_built_graph_in_mimes_names(built):
    topology = built.topology()
    assert json.loads(json.dumps(topology)) == topology and topology["format"] == "mime.topology/1"
    names = [node["name"] for node in topology["nodes"]]
    assert names == list(built.graph.node_names)
    arm = next(node for node in topology["nodes"] if node["name"] == "rpm.arm")
    assert arm["entry"]["name"] == "arm" and arm["entry"]["kind"] == "dynamics" and arm["thing"] == "rpm"
    edge = topology["edges"][0]
    assert {"source_node", "source_field", "target_node", "target_field", "reads"} <= set(edge)
    controlled = steered().build().topology()["controller"]
    assert controlled["mode"] == "callback" and controlled["live"]["offset"]["units"] == "1"


def test_a_saved_run_streams_through_its_hook(tmp_path):
    seen = []
    written = rig().run_to(tmp_path / "run", ms(30), hook=lambda chunk: seen.append(chunk.steps))
    assert seen == list(range(2, 31, 2)) and written.steps == 30
    assert (tmp_path / "run" / "topology.json").is_file() and written.recording is None
