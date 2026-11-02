"""The stage writer emits one root prim, ``/World``, which is the ``defaultPrim``: every other root prim is reserved for
MIME's overlays (a ghost of another drag rung at ``/Compare/<rung>/<body>``, a measured track), so no name a thing is
given inside ``/World`` can collide with one. Over a scene written Python-first, one with a referenced asset, A0's and
script A's. The control: the same stage with a second root prim added to it is found by the check every time."""

from pathlib import Path

import pytest
from pxr import Sdf
from robot_asset import bound_robot
from script_a import script_a
from script_a0 import a0
from test_stage import tube_and_body

from mime import Scene
from mime.scene import STAGE_FILE, Z

SCENES = {
    "python-first": lambda: tube_and_body(Scene.new(up=Z, units="m")),
    "referenced asset": lambda: bound_robot(),
    "script A0": lambda: a0(),
    "script A": lambda: script_a(),
}


def root_problems(path: Path) -> list[str]:
    """Every way the layer at ``path`` has a root prim other than ``/World``, or a ``defaultPrim`` that is not it."""
    layer = Sdf.Layer.FindOrOpen(str(path))
    found = [f"root prim {prim.path}" for prim in layer.rootPrims if prim.name != "World"]
    if [prim.name for prim in layer.rootPrims].count("World") != 1:
        found.append("no single /World")
    if layer.defaultPrim != "World":
        found.append(f"defaultPrim is {layer.defaultPrim!r}")
    return found


@pytest.mark.parametrize("name", SCENES)
def test_the_stage_writer_emits_only_world_as_a_root_prim(name, tmp_path):
    SCENES[name]().build().save(tmp_path / "saved")
    stage = tmp_path / "saved" / STAGE_FILE
    assert root_problems(stage) == []
    # The control: the same stage with a root prim planted beside /World (where an overlay would sit) is found.
    layer = Sdf.Layer.FindOrOpen(str(stage))
    Sdf.CreatePrimInLayer(layer, Sdf.Path("/Compare/mrs/umr"))
    planted = tmp_path / "planted.usda"
    layer.Export(str(planted))
    assert root_problems(planted) == ["root prim /Compare"]
