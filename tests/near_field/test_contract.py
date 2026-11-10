"""The near field's contract held by a second body model of another block size (C1, 1: several geometries and body
models in one medium are blocks of different sizes): a bead line (``bead_line.BeadLine``, test-only) beside MIME's MRS
sphere in one joint system, through the same ``joint_matrix`` and ``grand_resistance`` the joint node calls. Its
grand resistance is symmetric to the discretisation's error and positive definite; its blocks are each model's size;
and far apart each body's own 6 x 6 is its resistance alone, the coupling falling as the distance. The control: the
line's cross-kernel dropped (the joint system uncoupled), which leaves the sphere's coupled resistance its alone one
even close by, so the coupling's size at close range is the kernel's, not an artefact of the assembly."""

import jax.numpy as jnp
import numpy as np
from bead_line import BeadLine
from near_field_cases import NODE, bodies, joint_node, mrs, precision, spheres, start_inputs

from mime.build._maddening import parameter_leaves
from mime.nodes.hydro.mrs._walls import wall_correction_named
from mime.nodes.hydro.near_field import blocks_of, grand_resistance, joint_matrix


def system(distance: float, coupled: bool = True) -> tuple[np.ndarray, list[int]]:
    """The sphere (from a built joint node's first member) and a bead line ``distance`` beyond it along x, the grand
    resistance and each body's block size."""
    built = bodies("contract", spheres(0.5e-3, 2), mrs(0.4e-3)).build()
    node = joint_node(built)
    p = node.constants(parameter_leaves(built.graph)[NODE])
    sphere = node._operators[0]  # noqa: SLF001 - the MRS body model the node holds
    line = BeadLine(21, 3e-3, 0.2e-3)
    pose = jnp.asarray(start_inputs(built)["pose_0"])
    beyond = pose.at[0].add(distance)
    held = [node._held_value(0, p), None]  # noqa: SLF001
    joint = joint_matrix([sphere, line], [node._leaves(0, p), {}], held, jnp.stack([pose, beyond]),  # noqa: SLF001
                         p["viscosity"], wall_correction_named("free_space", "0.1.0"), {}, 3.0, coupled=coupled)
    grand = np.asarray(grand_resistance(joint), dtype=np.float64)
    return grand, [3 * placed.count for placed in joint.placed]


def test_an_mrs_sphere_and_a_bead_line_couple_as_blocks_of_different_sizes():
    with precision("float64"):
        close, sizes = system(2.5e-3)
        far, _ = system(1.0)
        alone, _ = system(1.0, coupled=False)
    assert sizes[0] != sizes[1] and sizes[1] == 63
    symmetric = 0.5 * (close + close.T)
    scale = np.sqrt(np.abs(np.diag(symmetric)))
    asymmetry = np.linalg.norm((close - close.T) / np.outer(scale, scale)) / np.linalg.norm(symmetric / np.outer(scale, scale))
    least = float(np.linalg.eigvalsh(symmetric / np.outer(scale, scale)).min())
    own_close, off_close = blocks_of(jnp.asarray(symmetric), 2)
    own_far, off_far = blocks_of(jnp.asarray(0.5 * (far + far.T)), 2)
    own_alone, _ = blocks_of(jnp.asarray(0.5 * (alone + alone.T)), 2)
    near = float(np.max(np.abs(np.asarray(own_close[0] - own_alone[0])) / np.abs(np.asarray(own_alone[0])).max()))
    distant = float(np.max(np.abs(np.asarray(own_far[0] - own_alone[0])) / np.abs(np.asarray(own_alone[0])).max()))
    print(f"\nblocks {sizes}; asymmetry {asymmetry:.3g}, least scaled eigenvalue {least:.3g}; the sphere's own 6 x 6 "
          f"moved by the line {near:.3g} at 2.5 mm, {distant:.3g} at 1 m")
    assert asymmetry < 1e-2 and least > 0.0
    assert near > 1e-3, "the control: close by, the line's kernel moves the sphere's own resistance"
    assert distant < 1e-5, "far apart, each body's own 6 x 6 is its resistance alone"
