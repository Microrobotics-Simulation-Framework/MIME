"""Things bound to prims of a referenced robot asset, and the ``mime:`` schema's design data on its part prims.

Two names per thing: a body bound to a prim takes the prim's last element as its name unless it is given one, and a
name or a prim held twice is refused. The magnet's direction is read from its part's prim, turned into the body's
frame by the transforms between them, and directs the body's moment; declared in the experiment as well, it is
refused. Each check has its control: a direction written differently in the asset moves the moment with it."""

import math

import numpy as np
import pytest
from robot_asset import bound_robot, robot_file, write_robot

from mime import Body, PermanentMagnet, State
from mime.build import BuildRefused
from mime.dynamics import RigidInertial
from mime.scene import asset, sha256_hex
from mime.things import PrimPath, Structure
from mime.units import A_m2, m, m_per_s, mm, rad_per_s


def codes(experiment) -> list[str]:
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    print(f"\n{refused.value}")
    return refused.value.codes()


def message(experiment, code: str) -> str:
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    (found,) = [r for r in refused.value.refusals if r.code == code]
    return f"{found.message} | {found.action}"


def moment(experiment) -> tuple[float, ...]:
    (value,) = [v for v in experiment.plan().values if v.path.endswith(".magnetic_moment")]
    return tuple(value.value)


# -- two names per thing --------------------------------------------------------------------------------------------

def test_a_body_bound_to_a_prim_is_named_after_its_leaf_and_both_names_are_saved():
    experiment = bound_robot()
    document = experiment.document()
    assert set(document["things"]) == {"UMR", "coils", "tube", "water"}
    assert document["things"]["UMR"]["prim"] == "/World/UMR"
    assert document["things"]["coils"]["prim"] == "/World/coils"
    plans = experiment.plan()
    assert plans.scene.prims == {"UMR": "/World/UMR", "coils": "/World/coils", "tube": "/World/tube",
                                 "water": "/World/water"}
    assert plans.scene.bound == {"UMR"}


def test_a_given_name_is_kept_beside_the_prim():
    plans = bound_robot(body_name="umr").plan()
    assert plans.scene.prims["umr"] == "/World/UMR" and "magnetic.torque:umr" in [p.owner for p in plans.entries]


def body_at(prim: PrimPath, name: str = "") -> Body:
    return Body(name, prim=prim, dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))))


def test_a_name_defaulting_to_a_leaf_another_thing_holds_is_refused_naming_both_prims(tmp_path):
    experiment = bound_robot()
    scene = experiment._settings.scene
    scene.reference(asset(write_robot(tmp_path / "spare.usda")), at="/World/Spare")
    experiment.add_body(body_at(scene["/World/Spare"], name="UMR"))
    text = message(experiment, "thing.duplicate")
    assert "two things are named 'UMR': a body at /World/UMR and a body at /World/Spare" in text


def test_a_body_binds_the_prim_its_asset_is_referenced_at_not_one_of_its_parts():
    experiment = bound_robot()
    experiment.add_body(body_at(experiment._settings.scene["/World/UMR/Magnets"], name="magnet"))
    assert "binds /World/UMR/Magnets, a prim inside the asset referenced at /World/UMR" in message(
        experiment, "thing.prim-inside-asset")


def test_two_things_bound_to_one_prim_are_refused():
    experiment = bound_robot()
    experiment.add_body(body_at(experiment._settings.scene["/World/UMR"], name="again"))
    assert "'UMR' and 'again' both bind /World/UMR" in message(experiment, "thing.duplicate-prim")


def test_a_prim_no_asset_brings_in_is_refused_with_the_prims_that_exist():
    experiment = bound_robot()
    experiment.add_body(body_at(PrimPath("/World/UMR/Magnet"), name="lost"))
    text = message(experiment, "thing.prim-missing")
    assert "binds /World/UMR/Magnet, which no referenced asset brings into the scene" in text
    assert "Did you mean '/World/UMR/Magnets'" in text


def test_a_thing_written_python_first_where_an_asset_already_is_is_refused():
    experiment = bound_robot(body_name="umr")
    experiment.add_structure(Structure("UMR"))
    assert "structure 'UMR' is written at /World/UMR, which a referenced asset already brings in" in message(
        experiment, "thing.prim-taken")


# -- the mime: schema -----------------------------------------------------------------------------------------------

def test_the_direction_is_read_from_the_asset_and_turned_into_the_bodys_frame():
    plans = bound_robot().plan()
    (magnet,) = plans.scene.magnets
    assert (magnet.part, magnet.in_part) == ("/World/UMR/Magnets", (1.0, 0.0, 0.0))
    assert np.allclose(magnet.in_body, (0.0, 1.0, 0.0), rtol=0, atol=4 * np.finfo(float).eps)
    assert magnet.source == "the MimeMagnetAPI schema on /World/UMR/Magnets"
    assert np.allclose(moment(bound_robot()), (0.0, 1.68e-3, 0.0), rtol=0, atol=1e-18)
    line = next(line for line in bound_robot().build().report.orientation if line.thing == "UMR")
    assert "magnet Magnets along (1, 0, 0) in its part's frame" in line.text()


@pytest.mark.parametrize(("direction", "turn", "expected"), [((1.0, 0.0, 0.0), 0.0, (1.0, 0.0, 0.0)),
                                                              ((0.0, 0.0, 1.0), 90.0, (0.0, 0.0, 1.0)),
                                                              ((0.6, 0.8, 0.0), 90.0, (-0.8, 0.6, 0.0))])
def test_the_moment_follows_the_direction_the_asset_writes(tmp_path, direction, turn, expected):
    robot = write_robot(tmp_path / "robot.usda", direction=direction, turn_degrees=turn)
    found = moment(bound_robot(robot=robot))
    assert np.allclose(found, 1.68e-3 * np.asarray(expected), rtol=0, atol=1e-18)


def test_a_direction_declared_twice_is_refused():
    twice = bound_robot(magnet=PermanentMagnet(direction=(1.0, 0.0, 0.0)))
    text = message(twice, "magnet.direction-twice")
    assert "mime:magnet:direction = (1, 0, 0) on its prim /World/UMR/Magnets, and direction=(1.0, 0.0, 0.0)" in text
    vector = bound_robot(moment=A_m2((0.0, 1.68e-3, 0.0)))
    assert "is a vector, and its direction is the magnet Magnets's direction" in message(
        vector, "magnet.direction-twice")


def test_a_part_without_design_data_takes_its_direction_from_the_experiment(tmp_path):
    robot = write_robot(tmp_path / "plain.usda", direction=None, schema=False)
    assert "magnet.direction-missing" in codes(bound_robot(robot=robot))
    declared = bound_robot(robot=robot, magnet=PermanentMagnet(direction=(1.0, 0.0, 0.0)))
    (magnet,) = declared.plan().scene.magnets
    assert magnet.source == "declared in the experiment"
    assert np.allclose(moment(declared), (0.0, 1.68e-3, 0.0), rtol=0, atol=1e-18)


@pytest.mark.parametrize(("options", "code"), [
    ({"schema": False}, "magnet.schema-not-applied"),
    ({"direction": None}, "magnet.schema-incomplete"),
    ({"direction": (2.0, 0.0, 0.0)}, "magnet.direction"),
])
def test_design_data_that_is_not_the_schemas_is_refused(tmp_path, options, code):
    assert code in codes(bound_robot(robot=write_robot(tmp_path / "robot.usda", **options)))


def test_a_magnet_on_a_body_with_no_prim_is_refused():
    experiment = bound_robot()
    experiment.add_body(Body("loose", dynamics=RigidInertial(), magnets={"Magnets": PermanentMagnet()},
                             initial=State(position=mm((0.0, 0.0, 0.0)))))
    assert "magnet.no-prim" in codes(experiment)


# -- the asset itself -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("options", "code"), [
    ({"default_prim": False}, "scene.asset-default-prim"),
    ({"meters_per_unit": 0.001}, "scene.asset-units"),
    ({"meters_per_unit": None}, "scene.asset-units"),
])
def test_an_asset_the_scene_cannot_read_as_written_is_refused(tmp_path, options, code):
    assert code in codes(bound_robot(robot=write_robot(tmp_path / "robot.usda", **options)))


def test_an_asset_pinned_by_its_hash_is_refused_once_its_bytes_change(tmp_path):
    robot = write_robot(tmp_path / "robot.usda")
    pin = sha256_hex(robot.read_bytes())
    experiment = bound_robot(robot=asset(robot, sha256=pin))
    assert experiment.plan().scene.layers[0].sha256 == pin
    robot.write_text(robot.read_text().replace("(1, 0, 0)", "(0, 1, 0)"))
    assert "is pinned at sha256" in message(experiment, "scene.asset-changed")


@pytest.mark.parametrize("at", ["/Robots/UMR", "/World/Robots/UMR", "World/UMR"])
def test_a_reference_must_be_a_prim_directly_below_world(at):
    assert "scene.reference-path" in codes(bound_robot(at=at))


def test_the_saved_folder_holds_the_assets_bytes_under_the_stage(tmp_path):
    lock = bound_robot().build().save(tmp_path)
    (layer, digest), = lock["stage"]["layers"].items()
    assert layer == "scene/layers/fl9_robot.usda"
    assert sha256_hex((tmp_path / layer).read_bytes()) == digest == sha256_hex(robot_file().read_bytes())
    assert "@./layers/fl9_robot.usda@" in (tmp_path / "scene" / "scene.usda").read_text()
    assert math.isclose(lock["resolved"]["scene"]["magnets"]["UMR"]["Magnets"]["in_body"][1], 1.0)
