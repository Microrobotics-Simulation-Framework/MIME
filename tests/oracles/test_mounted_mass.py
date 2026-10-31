"""Mounted mass on a dynamic arm (phase 7a4, MIME-ANO-122): a rigid payload grafted onto its link as a welded body,
against the link's inertial edited by hand. Float64 throughout.

- **Graft equals hand edit**: on the AR4 and on a C-arm-like tree with a sliding joint (``descriptions/c_arm.urdf``),
  a payload drawn on any link that moves (its mass, centre of mass and inertia, its placement's offset and turn) and
  welded on by ``graft`` gives the mass matrix and the gravity terms of the description whose ``<inertial>`` of that
  link is replaced by hand, in the file, with the two combined (the parallel-axis theorem written out here, not the
  library's), at drawn joint positions, and the same rigid bodies' inertials (the articulation's masses, centres of
  mass and inertias, of which the mass matrix and the gravity terms are made). The mass matrix and the gravity terms
  alone cannot see every misplaced payload: a payload moved along its link's only axis, or on a link a slide alone
  moves, changes neither, so the comparison of the inertials is part of the oracle, and it sees every one. Controls,
  each failing on every draw: the payload grafted at the wrong offset (moved by a twentieth of the tree's reach), and
  the payload not turned into the link's frame (its placement's turn dropped; the turn drawn at 0.3 rad or more, and
  the payload's principal moments drawn apart, so that a turn moves its tensor).
- **Through the build**: a part with a declared density on a link of A0's dynamic arm, a part on a dynamic motor's
  stator, and the motor's rotor with a part on it, each reach the arm's node as ``graft`` places them: the node's
  inertials are the grafted tree's; a part with no mass changes nothing (the first check, before this phase: the
  magnet's mass did not reach the arm, its material refused and no mass declarable).
- **Joint commands by kind**: the C-arm's slide takes a force in newtons (``joint_forces``) and its revolute joints
  torques (``joint_torques``), and on the kinematic rung displacements in metres (``joint_displacements``) and angles.
"""

import math
import warnings
import xml.etree.ElementTree as ElementTree

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import AR4, AR4_HOME, C_ARM, MAGNET_DENSITY, add_body, experiment, massive_rpm
from hypothesis import given, settings
from hypothesis import strategies as st

import multibody_cases as cases
from mime.core.multibody import ROOT, Inertial, Payload, RigidTransform, articulate, graft, gravity_forces, \
    mass_matrix, read_urdf
from mime.drives import Constant
from mime.dynamics import Arm
from mime.scene import asset
from mime.things import Apparatus
from mime.units import N, N_m, m, rad, s


PROPERTY = settings(deadline=None, max_examples=100)
EPS = float(np.finfo(np.float64).eps)
ROUNDINGS = 256
GRAVITY = 9.81 * np.array([0.3, -0.5, -0.8]) / np.linalg.norm([0.3, -0.5, -0.8])


def read(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_urdf(path).tree


#: Each tree with its reach (m) and the mass of its links' order (kg), which scale the payloads drawn on it.
TREES = {"ar4": (AR4, 0.75, 0.3), "c_arm": (C_ARM, 1.5, 20.0)}
HELD = {name: read(path) for name, (path, _, _) in TREES.items()}


def moving_links(tree):
    """The links a joint moves: every body but those welded to the root."""
    a = tree.articulation
    return tuple(name for name, body in zip(a.frame_names, a.frame_bodies, strict=True)
                 if body != ROOT and name in {b.name for b in tree.bodies})


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


@st.composite
def payloads(draw):
    """A tree, a link it moves, and a payload of the tree's scale on it: a box of drawn mass and half-sizes apart from
    one another (its principal moments apart), its centre of mass off its own origin and its principal axes turned in
    its own frame, placed on the link at a drawn offset and turned by at least 0.3 rad; with joint positions inside
    the limits and a direction to move it by."""
    name = draw(st.sampled_from(sorted(TREES)))
    _, reach, scale = TREES[name]
    tree = HELD[name]
    link = draw(st.sampled_from(moving_links(tree)))
    mass = scale * draw(st.floats(0.01, 1.0))
    size = 0.05 * reach * draw(st.floats(0.2, 1.0))
    halves = (size, size * (1.0 + draw(st.floats(0.3, 1.0))), size * (2.4 + draw(st.floats(0.3, 1.0))))
    centre = tuple(0.05 * reach * draw(st.floats(-1.0, 1.0)) for _ in range(3))
    principal = cases.turned(tuple(draw(st.floats(0.1, 1.0)) * draw(st.sampled_from((-1.0, 1.0))) for _ in range(3)),
                             draw(st.floats(0.0, math.pi)))
    inertial = cases.box(mass, *(2.0 * h for h in halves), centre, principal)
    offset = tuple(0.1 * reach * draw(st.floats(-1.0, 1.0)) for _ in range(3))
    axis = tuple(draw(st.floats(0.1, 1.0)) * draw(st.sampled_from((-1.0, 1.0))) for _ in range(3))
    turn = cases.turned(axis, draw(st.floats(0.3, 2.8)))
    move = unit([draw(st.floats(0.1, 1.0)) * draw(st.sampled_from((-1.0, 1.0))) for _ in range(3)])
    return name, link, Payload("mounted:payload", link, RigidTransform(offset, turn), inertial), \
        draw(cases.positions(tree)), move


def edited_by_hand(name, link, payload: Payload):
    """The description with ``link``'s ``<inertial>`` replaced, in the file, by the link's own and the payload's
    combined: each centre of mass in the link's frame, each tensor turned into its axes and moved to the common
    centre of mass by the parallel-axis theorem."""
    path, _, _ = TREES[name]
    tree = HELD[name]
    rotation = payload.placement.matrix()
    parts = [(payload.inertial.mass, rotation @ np.asarray(payload.inertial.centre_of_mass)
              + np.asarray(payload.placement.translation), rotation @ np.asarray(payload.inertial.inertia) @ rotation.T)]
    own = tree.body(link).inertial
    if own is not None:
        parts.append((own.mass, np.asarray(own.centre_of_mass), np.asarray(own.inertia)))
    mass = sum(p[0] for p in parts)
    centre = sum(p[0] * p[1] for p in parts) / mass
    tensor = sum(p[2] + p[0] * (np.dot(p[1] - centre, p[1] - centre) * np.eye(3)
                                - np.outer(p[1] - centre, p[1] - centre)) for p in parts)
    root = ElementTree.parse(path).getroot()
    element = next(e for e in root.iter("link") if e.get("name") == link)
    for old in element.findall("inertial"):
        element.remove(old)
    inertial = ElementTree.SubElement(element, "inertial")
    ElementTree.SubElement(inertial, "origin", xyz=" ".join(repr(float(c)) for c in centre), rpy="0 0 0")
    ElementTree.SubElement(inertial, "mass", value=repr(float(mass)))
    ElementTree.SubElement(inertial, "inertia", **{key: repr(float(tensor[i, j])) for key, (i, j) in {
        "ixx": (0, 0), "ixy": (0, 1), "ixz": (0, 2), "iyy": (1, 1), "iyz": (1, 2), "izz": (2, 2)}.items()})
    return ElementTree.tostring(root, encoding="unicode")


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return tmp_path_factory.mktemp("edited")


@jax.jit
def terms(articulation, q):
    return mass_matrix(articulation, q), gravity_forces(articulation, q, jnp.asarray(GRAVITY))


def misfits(grafted, by_hand, q) -> dict[str, float]:
    """How far the grafted tree is from the one edited by hand, each against its own scale's rounding (1 at the
    bound): the mass matrix, the gravity terms, and the rigid bodies' masses, first moments and inertias."""
    a, b = grafted, by_hand
    (mass_a, weight_a), (mass_b, weight_b) = terms(a, jnp.asarray(q)), terms(b, jnp.asarray(q))
    reach = float(np.max(np.abs(b.origin_translations))) + float(np.max(np.abs(b.centres_of_mass)))
    total = float(np.sum(b.masses))
    bound = ROUNDINGS * EPS
    return {
        "mass matrix": float(np.max(np.abs(mass_a - mass_b))) / (bound * float(np.max(np.abs(mass_b)))),
        "gravity terms": float(np.max(np.abs(weight_a - weight_b))) / (bound * total * 9.81 * reach),
        "masses": float(np.max(np.abs(a.masses - b.masses))) / (bound * total),
        "first moments": float(np.max(np.abs(a.masses[:, None] * a.centres_of_mass
                                             - b.masses[:, None] * b.centres_of_mass))) / (bound * total * reach),
        "inertias": float(np.max(np.abs(a.inertias - b.inertias))) / (bound * float(np.max(np.abs(b.inertias)))),
    }


def compare(folder, drawn, planted=None) -> dict[str, float]:
    name, link, payload, q, move = drawn
    path = folder / f"{name}.urdf"
    path.write_text(edited_by_hand(name, link, payload))
    by_hand = read(path).articulation
    placed = planted(payload, move, TREES[name][1]) if planted is not None else payload
    tree = HELD[name]
    return misfits(articulate(graft(tree, [placed]), frames=tree.frame_names), by_hand, q)


@PROPERTY
@given(payloads())
def test_a_payload_grafted_on_any_link_is_its_inertial_edited_by_hand(folder, drawn):
    found = compare(folder, drawn)
    assert max(found.values()) <= 1.0, found


def at_the_wrong_offset(payload, move, reach):
    shifted = np.asarray(payload.placement.translation) + 0.05 * reach * move
    return Payload(payload.name, payload.frame, RigidTransform(tuple(shifted), payload.placement.rotation),
                   payload.inertial)


def not_turned(payload, move, reach):
    return Payload(payload.name, payload.frame, RigidTransform(payload.placement.translation), payload.inertial)


@PROPERTY
@given(payloads())
def test_control_a_payload_at_the_wrong_offset_fails_on_every_draw(folder, drawn):
    found = compare(folder, drawn, at_the_wrong_offset)
    assert max(found.values()) > 100.0, found


@PROPERTY
@given(payloads())
def test_control_a_payload_not_turned_into_its_links_frame_fails_on_every_draw(folder, drawn):
    found = compare(folder, drawn, not_turned)
    assert max(found.values()) > 100.0, found


def test_a_payload_on_a_body_welded_to_the_root_moves_nothing():
    tree = HELD["ar4"]
    still = Payload("mounted:payload", "base_link", RigidTransform((0.1, 0.0, 0.0)), Inertial(2.0))
    q = np.asarray(AR4_HOME)
    found = misfits(articulate(graft(tree, [still]), frames=tree.frame_names), tree.articulation, q)
    assert max(found.values()) <= 1.0, found


# -- through the build ---------------------------------------------------------------------------------------------

def built_arm(experiment_):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = experiment_.build()
    return built, built.graph.get_node("rpm.arm").constants()


def arrays_of(constants):
    return {key: np.asarray(constants[key]) for key in ("masses", "centres_of_mass", "inertias")}


def test_through_the_build_each_mass_reaches_the_arm_as_graft_places_it():
    """A camera on link 3, a stator part on the motor's base, and the magnet on the rotor (turned to be symmetric
    about the spin axis): the arm's node holds the tree with the three grafted where the walk places them."""
    built, held = built_arm(massive_rpm("grafted", magnet_axis=(0.0, 0.0, 1.0), extra=True))
    tree = read(AR4)
    walked = {payload.name: payload for plan in built.plans.entries for payload in plan.mounted}
    assert sorted(walked) == ["mounted:rpm.camera", "mounted:rpm.motor.rotor", "mounted:rpm.stator"]
    assert walked["mounted:rpm.camera"].frame == "link_3"
    assert walked["mounted:rpm.stator"].frame == walked["mounted:rpm.motor.rotor"].frame == "link_6"
    expected = inertials_of(articulate(graft(tree, list(walked.values())), frames=tree.frame_names))
    for key, value in arrays_of(held).items():
        assert np.allclose(value, expected[key], rtol=0.0, atol=ROUNDINGS * EPS * float(np.max(np.abs(expected[key])))), key
    bare = inertials_of(tree.articulation)
    assert not np.allclose(arrays_of(held)["masses"], bare["masses"])
    lines = built.report.text()
    assert "mounted masses, each with what carries it:" in lines and "welded on rpm.arm.link_3" in lines


def inertials_of(articulation):
    """An articulation's inertials as a dynamic arm's node holds them, flattened."""
    return {"masses": np.ravel(articulation.masses), "centres_of_mass": np.ravel(articulation.centres_of_mass),
            "inertias": np.ravel(articulation.inertias)}


def test_a_part_without_mass_changes_nothing_and_the_first_check_held_before():
    """The magnet with no density declared: the arm's inertials are its description's alone, as before this phase,
    when an apparatus part's mass could not reach the arm at all."""
    _, held = built_arm(massive_rpm("massless", density=None))
    bare = inertials_of(read(AR4).articulation)
    for key, value in arrays_of(held).items():
        assert np.array_equal(value, bare[key]), key


# -- joint commands by kind ------------------------------------------------------------------------------------------

def c_arm(rung: str, commands):
    exp = experiment(f"c_arm_{rung}")
    held = (0.1, 0.2, -0.3)
    segment = Arm.Dynamic(description=asset(C_ARM), initial_positions=rad(held)) if rung == "dynamic" else \
        Arm.Kinematic(description=asset(C_ARM), initial_positions=rad(held))
    exp.add_apparatus(Apparatus("rig", chain={"arm": segment}, commands=commands))
    add_body(exp)
    return exp


def test_a_sliding_joint_takes_a_force_in_newtons_and_a_turning_one_a_torque():
    tree = read(C_ARM)
    q = np.asarray((0.1, 0.2, -0.3))
    weight = np.asarray(gravity_forces(tree, q, np.asarray((0.0, 0.0, -9.80665))))
    exp = c_arm("dynamic", {"arm.joint_forces": Constant(N((float(weight[0]),))),
                            "arm.joint_torques": Constant(N_m(tuple(float(t) for t in weight[1:])))})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = exp.build()
    history = built.run(s(0.02), history=True).history
    moved = np.max(np.abs(history["rig.arm"]["joint_positions"] - q))
    assert moved < 1e-9, moved
    spec = built.graph.get_node("rig.arm").boundary_input_spec()
    assert spec["joint_forces"].shape == (1,) and spec["joint_torques"].shape == (2,)
    assert spec["joint_forces"].expected_units == "N" and spec["joint_torques"].expected_units == "N m"


def test_control_a_force_given_in_newton_metres_or_all_joints_as_torques_is_refused():
    from mime.build import BuildRefused

    for commands in ({"arm.joint_forces": Constant(N_m((1.0,))), "arm.joint_torques": Constant(N_m((0.0, 0.0)))},
                     {"arm.joint_torques": Constant(N_m((0.0, 0.0, 0.0)))}):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(BuildRefused) as caught:
                c_arm("dynamic", commands).build()
        assert caught.value.refusals, commands


def test_the_kinematic_slide_takes_a_displacement_in_metres():
    exp = c_arm("kinematic", {"arm.joint_displacements": Constant(m((0.25,))),
                              "arm.joint_positions": Constant(rad((0.4, -0.2)))})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = exp.build()
    history = built.run(s(0.003), history=True).history
    assert np.allclose(history["rig.arm"]["joint_positions"][-1], (0.25, 0.4, -0.2), rtol=0.0, atol=1e-12)


def test_the_magnets_density_gives_its_mass():
    _, held = built_arm(massive_rpm("dense", magnet_axis=(0.0, 0.0, 1.0)))
    bare = inertials_of(read(AR4).articulation)
    added = float(np.sum(np.asarray(held["masses"])) - np.sum(bare["masses"]))
    expected = MAGNET_DENSITY * math.pi * 17.5e-3 ** 2 * 20e-3
    assert abs(added - expected) <= 1e-12 * expected, (added, expected)


def material_part(record, name: str):
    """A camera of water's density on link 3 of A0's dynamic arm, its density read from its material's record at 20
    degrees C, as script A reads water's (``tests/script_a.py``)."""
    from script_a import CONDITIONS, WATER_RECORD

    from mime.experiment import Experiment
    from mime.materials import Record, packaged_record
    from mime.scene import Cylinder, Scene, Z
    from mime.things import Part
    from mime.units import ms, standard_gravity

    exp = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(), dt=ms(1),
                     conditions=CONDITIONS)
    material = WATER_RECORD if record == "openmaterica_example_water" else Record(record, packaged_record(record))
    exp.add_apparatus(Apparatus("rpm", chain={"arm": Arm.Dynamic(description=asset(AR4),
                                                                 initial_positions=rad(AR4_HOME))},
                                parts={"camera": Part(Cylinder(radius=m(0.015), length=m(0.04), axis=(0.0, 1.0, 0.0)),
                                                      material=material, mount="arm.link_3")},
                                commands={"arm.joint_torques": Constant(N_m((0.0,) * 6))}))
    add_body(exp)
    return exp


def test_a_parts_density_is_read_from_its_materials_record_and_a_record_without_one_is_refused():
    from mime.build import BuildRefused

    _, held = built_arm(material_part("openmaterica_example_water", "watery"))
    added = float(np.sum(np.asarray(held["masses"])) - np.sum(inertials_of(read(AR4).articulation)["masses"]))
    volume = math.pi * 0.015 ** 2 * 0.04
    assert 0.99 * 998.0 * volume <= added <= 1.0 * 1000.0 * volume, added
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(BuildRefused) as caught:
            material_part("openmaterica_example_ndfeb_n45", "no_density").plan()
    found = [r for r in caught.value.refusals if r.code == "record.not-met"]
    assert found and "holds no section bulk" in found[0].message, [(r.code, r.message) for r in caught.value.refusals]
