"""Gate 1, one body against dev1 (C1, 4; W3's first step (d)): a lone mrs body keeps dev1's per-body node, so no released
program builds the joint node, and the joint node of that one member gives dev1's resistance to its solve's rounding,
in free space and in the tube, in both dtypes. Its line is of each entry over its row's and column's diagonal scale:
1e-12 in float64 (in free space dev1 holds its 6 x 6 and turns it where the joint node solves its turned matrix; the
difference is the LU's rounding on a conditioned matrix), 1e-5 in float32. The control: the member's radius one part in
a million up in float64 (a thousand in float32), which moves the joint node's resistance far past the line, so the
comparison sees a change of that size."""

import math

import jax.numpy as jnp
import numpy as np
import pytest
from near_field_cases import IDENTITY, entrywise, precision, single

from mime import Body, Experiment, Medium, State, Structure
from mime.build._maddening import parameter_leaves
from mime.dynamics import RigidInertial
from mime.interactions import Drag
from mime.nodes.hydro.mrs import MRSDragNode
from mime.scene import Scene, Sphere, Tube, Z
from mime.things import Part
from mime.units import kg, kg_m2, mm, mPa_s, m_per_s, m_per_s2, ms, rad_per_s

LINES = {"float64": 1e-12, "float32": 1e-5}
#: The control's radius change in each dtype: far above the line's rounding, far below the radius.
PLANTS = {"float64": 1e-6, "float32": 1e-3}
#: A pose off the origin, turned 30 degrees about y: its quaternion unit to float64's rounding, as a body's step keeps it
#: (a quaternion off unit length by 8e-8 makes a rotation that is not one, and the two nodes turn by it differently).
POSE = (0.2e-3, -0.1e-3, 0.3e-3, math.cos(math.pi / 12.0), 0.0, math.sin(math.pi / 12.0), 0.0)


def lone(tube: bool) -> Experiment:
    experiment = Experiment("lone", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
    if tube:
        experiment.add_structure(Structure("tube", geometry=Tube(axis=(0.0, 0.0, 1.0), inner_radius=mm(5.0),
                                                                 wall=mm(0.5), length=mm(200.0))))
        experiment.add_medium(Medium("water", fills="tube.lumen"))
    else:
        experiment.add_medium(Medium("water"))
    experiment.declare("water.viscosity", mPa_s(1.0), reason="water")
    experiment.add_body(Body("ball", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=mm(1.0)))},
                             initial=State(position=mm((0.0, 0.0, 0.0)), orientation=IDENTITY,
                                           velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0,) * 3))))
    experiment.declare("ball.mass", kg(4.19e-6), reason="a millimetre sphere")
    experiment.declare("ball.inertia", kg_m2((1.7e-12,) * 3), reason="a millimetre sphere")
    wall = "tube" if tube else "free_space"
    experiment.attach(Drag.MRS(spacing=mm(0.6), blob=0.5, wall_kind=wall), body="ball", medium="water")
    return experiment


def pair(built, scale: float = 1.0, **settings) -> tuple[np.ndarray, np.ndarray]:
    """dev1's node's resistance at ``POSE`` and the one-member joint node's (built with ``settings``), its radius scaled
    by ``scale``."""
    node = built.graph.get_node("drag:ball")
    assert isinstance(node, MRSDragNode)
    leaves = parameter_leaves(built.graph)["drag:ball"]
    dtype = leaves["viscosity"].dtype
    pose = jnp.asarray(POSE, dtype=dtype)
    dev1 = node.update(node.initial_state(), {"pose": pose}, 1e-3, params=leaves)
    joint = single(node, **settings)
    params = joint.params_pytree()
    params["m0_shape_radius"] = params["m0_shape_radius"] * scale
    out = joint.update(joint.initial_state(), {"pose_0": pose, "twist_0": jnp.zeros(6, dtype=dtype),
                                               "mass_matrix_0": jnp.eye(6, dtype=dtype)}, 1e-3, params=params)
    return np.asarray(dev1["resistance"], dtype=np.float64), np.asarray(out["resistance_0"], dtype=np.float64)


@pytest.mark.parametrize("tube", [False, True], ids=["free_space", "tube"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_one_member_of_the_joint_node_gives_dev1s_resistance_to_its_rounding(dtype, tube):
    with precision(dtype):
        built = lone(tube).build()
        assert sorted(built.graph.node_names) == ["ball", "drag:ball"], "a lone body keeps its own node"
        dev1, joint = pair(built)
        _, moved = pair(built, 1.0 + PLANTS[dtype])
    found, control = entrywise(joint, dev1), entrywise(moved, dev1)
    print(f"\n{dtype}, {'tube' if tube else 'free space'}: the joint node against dev1 {found:.3g} (line "
          f"{LINES[dtype]:g}); the radius {PLANTS[dtype]:g} up {control:.3g}")
    assert found <= LINES[dtype]
    assert control > 10.0 * LINES[dtype], "the control: the planted radius is seen"


#: Gate 1 on the iterative route (the one member past a takeover line of zero), each entry over its diagonal scale: the
#: solve's tolerance carried through the member's conditioned block (float64's 1e-10, float32's 1e-5). In free space the
#: member's preconditioner is its own block's inverse and one iteration solves it; in the tube the block holds the wall's
#: image too, which the preconditioner leaves out.
ITERATIVE_LINES = {"float64": 1e-8, "float32": 5e-5}


@pytest.mark.parametrize("tube", [False, True], ids=["free_space", "tube"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_on_the_iterative_route_one_member_gives_dev1s_resistance_to_the_solves_tolerance(dtype, tube):
    with precision(dtype):
        built = lone(tube).build()
        dev1, joint = pair(built, dense_unknowns=0)
        _, moved = pair(built, 1.0 + PLANTS[dtype], dense_unknowns=0)
    found, control = entrywise(joint, dev1), entrywise(moved, dev1)
    print(f"\n{dtype}, {'tube' if tube else 'free space'}, GMRES: the joint node against dev1 {found:.3g} (line "
          f"{ITERATIVE_LINES[dtype]:g}); the radius {PLANTS[dtype]:g} up {control:.3g}")
    assert found <= ITERATIVE_LINES[dtype]
    assert control > 10.0 * ITERATIVE_LINES[dtype], "the control: the planted radius is seen"
