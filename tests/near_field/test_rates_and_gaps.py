"""The near field's two refusals of a configuration its joint system does not describe, and its flag at the validity
line's edge.

**Members at different rates** (``drag.near-field-rates``): the joint node steps at the base step and checks the
lagged gain at that step; a body whose dynamics steps slower takes the others' loads over its own longer step, whose
amplification the check does not see. The refuting review's case: the triangle of MIME's default spheres a tenth of a
radius apart, each body's relaxation time the base step's (1 ms), the bodies stepping every second: the node's gain
read 0.877 and the bodies' lagged step amplified by 1.0146 a step. Refused at the plan, at a body's rate of 10 ms and
of 1 s; the control, the same triangle at the base step, plans.

**Bodies that overlap** (``drag.near-field-overlap``): a point of one body behind another's surface, by the tangent
plane at the other's nearest force point, deeper than a quarter of a cell (the estimate's resolution). Refused at the
build for two spheres interpenetrating by half a radius, and for two coincident ones, whose message is the overlap's,
not the discretisation's; a step that brings two bodies into overlap stops the run there. The controls: two spheres a
twentieth of a radius apart at a coarse spacing (where the tangent plane's sag is largest) read no overlap. **Near contact**: a gap below a twentieth of the smaller body's equivalent radius raises ``near_contact``;
at a third of a radius it does not."""

import math

import numpy as np
import pytest
from near_field_cases import NODE, RADIUS, VISCOSITY, bodies, joint_node, mrs, precision, spheres, start_inputs, stepped

from mime.build import BuildRefused
from mime.interactions import Drag
from mime.nodes.hydro.near_field import CONTACT_GAP, JointDragNode
from mime.scene import Sphere
from mime.units import m, ms

COARSE = 0.6e-3


def triangle(rate=None):
    """The review's triangle: MIME's default spheres a tenth of a radius apart, each body's relaxation time the base
    step's."""
    mass = 6.0 * math.pi * VISCOSITY * RADIUS * 1e-3
    return bodies("rates", spheres(0.1 * RADIUS, 3), Drag.MRS(wall_kind="free_space"), mass=mass,
                  inertia=0.4 * mass * RADIUS * RADIUS, rate=rate)


@pytest.mark.parametrize("rate_ms", [10.0, 1000.0])
def test_a_near_field_whose_bodies_step_slower_than_the_base_step_is_refused_at_the_plan(rate_ms):
    with pytest.raises(BuildRefused) as refused:
        triangle(ms(rate_ms)).plan()
    found = [r for r in refused.value.refusals if r.code == "drag.near-field-rates"]
    print(f"\n{found[0].message if found else refused.value.codes()}")
    assert found and refused.value.codes() == ["drag.near-field-rates"]
    message = found[0].message
    assert all(f"'s{i}' every {round(rate_ms)} steps" in message for i in range(3)) and "every 1" in message


def test_control_the_same_triangle_at_the_base_step_plans():
    triangle().plan()


def overlapping(centres: float, spacing: float = COARSE):
    placed = {"s0": (Sphere(radius=m(RADIUS)), (0.0, 0.0, 0.0)), "s1": (Sphere(radius=m(RADIUS)), (centres, 0.0, 0.0))}
    return bodies("overlap", placed, mrs(spacing))


@pytest.mark.parametrize(("centres", "depth"), [(1.5e-3, 0.5e-3), (0.0, 1e-3)], ids=["half a radius", "coincident"])
def test_two_bodies_that_overlap_are_refused_at_the_build_naming_the_pair_and_the_overlap(centres, depth):
    with precision("float64"):
        with pytest.raises(BuildRefused) as refused:
            overlapping(centres).build()
    print(f"\n{refused.value}")
    assert refused.value.codes() == ["drag.near-field-overlap"]
    message = refused.value.refusals[0].message
    assert "s0 and s1 overlap" in message and "behind s1's surface" in message
    stated = float(message.split("lies ")[1].split(" mm")[0]) * 1e-3
    assert stated == pytest.approx(depth, rel=0.1), (stated, depth)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_step_that_brings_two_bodies_into_overlap_stops_the_run_there(dtype):
    with precision(dtype):
        built = bodies("closing", spheres(0.3e-3, 2), mrs(COARSE)).build()
        node = joint_node(built)
        inputs = start_inputs(built)
        clean = stepped(built, inputs)
        moved = dict(inputs)
        moved["pose_1"] = np.asarray(inputs["pose_1"]).copy()
        moved["pose_1"][0] -= 0.8e-3
        out = stepped(built, moved)
        problem = node.stop_problem(out)
    print(f"\n{dtype}: unmoved least gap {float(clean['least_gap']) * 1e3:.4g} mm, stopped {float(clean['stopped_at'])}; "
          f"moved {float(out['least_gap']) * 1e3:.4g} mm, why {float(out['stopped_why'])}; {problem}")
    assert float(clean["stopped_at"]) == 0.0 and float(clean["overlap"]) == 0.0
    assert float(out["stopped_at"]) == 1.0 and float(out["overlap"]) == 1.0
    assert all(np.all(np.isnan(out[f"{field}_{i}"])) for field in ("resistance", "background_load") for i in range(2))
    assert problem is not None and problem.code == "drag.near-field-overlap"


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_control_two_spheres_a_twentieth_of_a_radius_apart_at_a_coarse_spacing_do_not_overlap(dtype):
    """Where the tangent plane's sag is largest (a coarse spacing, a cell more than half a radius): the step at a gap of
    a twentieth of a radius reads no overlap, whatever else its coarse discretisation does there."""
    with precision(dtype):
        built = bodies("close", spheres(0.3e-3, 2), mrs(COARSE)).build()
        inputs = dict(start_inputs(built))
        inputs["pose_1"] = np.asarray(inputs["pose_1"]).copy()
        inputs["pose_1"][0] -= 0.3e-3 - 0.05 * RADIUS
        out = stepped(built, inputs)
    print(f"\n{dtype}: least gap estimated {float(out['least_gap']) * 1e3:.4g} mm against 0.05 mm, overlap "
          f"{float(out['overlap'])}, stopped {float(out['stopped_at'])} (why {float(out['stopped_why'])})")
    assert float(out["overlap"]) == 0.0 and float(out["stopped_why"]) != 4.0


@pytest.mark.parametrize(("gap", "raised"), [(0.02, 1.0), (0.3, 0.0)])
def test_near_contact_is_raised_below_a_twentieth_of_a_radius_and_not_at_a_third(gap, raised):
    with precision("float64"):
        out = stepped(bodies("contact", spheres(gap * RADIUS, 2), mrs(COARSE)).build())
    print(f"\nh/a {gap}: least gap {float(out['least_gap']) * 1e3:.4g} mm (edge {CONTACT_GAP * RADIUS * 1e3:g} mm), "
          f"near_contact {float(out['near_contact'])}")
    assert float(out["near_contact"]) == raised and float(out["stopped_at"]) == 0.0


def test_the_overlap_is_judged_before_the_cluster_s_definiteness():
    """Coincident bodies make the cluster's grand resistance singular, which the step judges indefinite too: the stop
    it records is the overlap's (``stopped_why`` 4), so its message names the overlap."""
    with precision("float64"):
        built = bodies("apart", spheres(0.5e-3, 2), mrs(COARSE)).build()
        node = joint_node(built)
        inputs = dict(start_inputs(built))
        inputs["pose_1"] = np.asarray(inputs["pose_0"]).copy()
        out = stepped(built, inputs)
    assert isinstance(node, JointDragNode) and float(out["stopped_why"]) == 4.0
    assert node.stop_problem(out).code == "drag.near-field-overlap"
    assert NODE == "near_field:water"
