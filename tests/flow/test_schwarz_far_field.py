"""Gate 10: a rotating helix's far field through the grid. The coupling spike's R6 helix (``rft_cases.R6_HELIX``, ``D``
= 1 mm its tube's diameter: 2.5 turns of 0.6 D, radius 0.15 D, filament 0.03 D) as an MRS body in free space, rotating
at 1 rad/s about its axis and swimming free along it (its free-space columns, the axial force zero); its force
distribution scattered by the exchange kernel into a large free box (16 D at D/h 8, the steady limit, Hasimoto's term
for its net force's images, the Schwarz partner's own transforms: ``own_response``, ``image_terms``), the box's
velocity gathered at probe points against the distribution's own regularised Stokeslets summed directly.

**The case** (R5's set-up): in the plane through the helix's centre across its axis, at 0.75, 1, 1.5 and 2 D from it,
the median relative miss over twelve points within ``FAR_LINE``, where the far field is the torque's rotlet and the
stresslet; on the axis at 2 D within ``AXIS_LINE``. The helix's place within a cell and its turn about its axis are
drawn from ``DETECTION_SEED``. **Control:** the net force alone scattered (``MIME_PLANT_SCHWARZ=net``, at the centre:
the rotlet and the stresslet dropped), which misses by about the whole field."""

import math

import jax.numpy as jnp
import numpy as np
from rft_cases import R6_HELIX
from schwarz_cases import draw, plant, schwarz_body

from mime.interactions import Drag
from mime.nodes.flow.fvm._exchange import gather, own_box, scatter
from mime.nodes.flow.fvm._grid import MACGrid
from mime.nodes.flow.fvm._schwarz import image_terms, own_response
from mime.units import m

DIAMETER = 1e-3
SIDE = 128
SPACING = 16.0 * DIAMETER / SIDE
FORCE_SPACING = 2e-5
MU, RHO = 1e-3, 1e3
#: The lines: the plane's medians and the axis's at 2 D. Near the helix the gathered velocity is the kernel's average of a
#: field curving over four to six cells [inference: gate 7's measured mechanism, about (h/d)^2 / 2 at a distance d from
#: the nearest of the helix], about 2 % at 0.75 D; R5's line of Stokeslets read 0.6 to 0.9 % in the same box.
FAR_LINE, AXIS_LINE = 0.03, 0.05
DISTANCES = (0.75, 1.0, 1.5, 2.0)


def helix_member():
    from near_field_cases import bodies  # noqa: PLC0415

    experiment = bodies("schwarz_helix", {"s0": (R6_HELIX, (0.0, 0.0, 0.0))},
                        Drag.MRS(wall_kind="free_space", spacing=m(FORCE_SPACING)))
    return experiment.build().graph.get_node("drag:s0")


def stokeslets(targets: np.ndarray, points: np.ndarray, forces: np.ndarray, eps_squared: float) -> np.ndarray:
    """The regularised Stokeslets' velocity at ``targets`` (the MRS kernel's form, blob ``eps``)."""
    apart = targets[:, None, :] - points[None, :, :]
    squared = np.sum(apart * apart, -1)
    weight = 1.0 / (8.0 * math.pi * MU * (squared + eps_squared) ** 1.5)
    return (np.einsum("pn,nk->pk", weight * (squared + 2.0 * eps_squared), forces)
            + np.einsum("pn,pnk->pk", weight * np.einsum("pnk,nk->pn", apart, forces), apart))


def misses() -> dict[str, float]:
    rng = draw()
    member = helix_member()
    turn = rng.uniform(0.0, 2.0 * math.pi)
    centre = np.full(3, 0.5 * SIDE * SPACING) + rng.uniform(0.0, 1.0, 3) * SPACING
    pose = jnp.asarray([*centre, math.cos(0.5 * turn), math.sin(0.5 * turn), 0.0, 0.0])
    body = schwarz_body(member, pose, MU)
    placed = body.placed
    count = placed.count
    columns = np.linalg.solve(np.asarray(body.matrix), np.asarray(placed.motions()))
    area = np.asarray(placed.area)[:, None]
    translating, rotating = (area * columns[:, k].reshape(count, 3) for k in (0, 3))
    swim = -rotating[:, 0].sum() / translating[:, 0].sum()
    forces = swim * translating + rotating
    points = np.asarray(placed.force)
    scattered, at = (np.sum(forces, axis=0)[None, :], centre[None, :]) if plant() == "net" else (forces, points)
    grid = MACGrid((SIDE,) * 3, SPACING)
    box = own_box(SIDE, SPACING, jnp.float64)
    held = [grid.exchange_stencil(k, jnp.asarray(at), 1.0) for k in range(3)]
    field = own_response(box, tuple(scatter(grid, k, jnp.asarray(scattered[:, k]), held[k]) for k in range(3)), 0.0,
                         MU / RHO, RHO)
    now, _ = image_terms(SIDE, SPACING, 0.0, MU / RHO, MU)
    across = np.array([[0.0, math.cos(t), math.sin(t)] for t in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)])
    found = {"swim_over_omega_d": float(swim / DIAMETER)}
    for where, unit in (("plane", across), ("axis", np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))):
        for distance in DISTANCES if where == "plane" else (2.0,):
            targets = centre + distance * DIAMETER * unit
            stencil = [grid.exchange_stencil(k, jnp.asarray(targets), 1.0) for k in range(3)]
            read = np.stack([np.asarray(gather(grid, k, field[k], stencil[k])) for k in range(3)], axis=1)
            read = read + float(now) * np.sum(scattered, axis=0)[None, :]
            wanted = stokeslets(targets, points, forces, float(placed.eps_squared))
            miss = np.linalg.norm(read - wanted, axis=1) / np.linalg.norm(wanted, axis=1)
            found[f"{where}_{distance}"] = float(np.median(miss))
    return found


def test_a_rotating_helixs_scattered_distribution_carries_its_far_field(float64):
    found = misses()
    print(f"F5 far field: {found}")
    assert all(found[f"plane_{d}"] <= FAR_LINE for d in DISTANCES), found
    assert found["axis_2.0"] <= AXIS_LINE, found
