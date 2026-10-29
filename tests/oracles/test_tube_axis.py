"""The tube's read near its axis (F9) and its readable range from the table's own provenance, on a small table of the
unit tube built by the table's builder (``tube_cases.builder_table``) with its radii from the axis, as the full table
holds them, read through the tube's correction in a tube moved off the origin, along z (where a sample on the axis is
exactly on it) and turned (where it is within the frame's rounding of it). The series is taken at the table's
truncation (its sources reach 0.9 of the radius, so the series is given a source there).

**Near the axis** a sample's angle is undefined, and the rule (``mime.nodes.hydro.bem._tube``) reads each pair with
the target's rows in the source's frame and blends each pair's read with its reciprocal's, the sample nearer the axis
the target. The tests, at a sample ``rho`` of ``0``, ``1e-8``, ``1e-3``, ``0.01`` and ``0.02`` of the radius from the
axis, as the source of a pair, its target and both: every value and every derivative (reverse and forward) is finite;
the Cartesian block of every pair equals the series (``tube_fourier_bessel.image``) within the table's read accuracy,
measured on the same table at samples off the axis; and a sample moved through the axis moves the block continuously,
within ``1e-6`` of its largest entry for a step of ``1e-9`` of the radius either side. The table's knots against the
series bound the truncation's share of every miss (within ``1e-6``). Controls, each failing on every run: today's rule
planted (a sample's angle from its own position, ``nonzero_norm``) gives NaN on the axis; the blend dropped (the read
with ``t`` the target alone) is discontinuous where a source crosses the axis, by more than ``1e-6``.

**The readable range** is the file's: its ``readable_rho`` where it states one, which narrows its knots' and never
widens them, and its knots' otherwise; a read past it is refused (NaN, not accepted), as a read past the knots is. A
file is written in the 2026-05-21 form with and without the key: control, the same knots read with a different stated
range accept a different set of samples, so the range is the file's and not the code's."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tube_fourier_bessel as fb
from jax import Array
from tube_cases import builder_data, builder_table

import mime.nodes.hydro.bem._tube as tube_module
from mime.core import switch
from mime.nodes.hydro.bem import frame_across, tube_correction, tube_table_from

#: The table's knots, in the tube's radius: radii from the axis, the angle on ``[0, pi]``, the axial offset.
RADII = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
ANGLES = tuple(float(a) for a in np.linspace(0.0, math.pi, 13))
OFFSETS = tuple(float(z) for z in np.linspace(0.0, 0.6, 7))
#: The tubes the samples are placed in, each moved off the origin, in SI, at a viscosity that is not 1: one along z,
#: in which a sample on the axis is exactly on it (its frame holds only zeros and ones), and one turned, in which it is
#: within the frame's rounding of it.
TUBE, MU = 3e-3, 2e-3
CENTRE = np.array([1e-3, -2e-3, 0.5e-3])
TURNED = np.array([0.3, -0.4, 0.86]) / np.linalg.norm([0.3, -0.4, 0.86])
FRAMES = {"along z": np.asarray(frame_across((0.0, 0.0, 1.0))).reshape(3, 3),
          "turned": np.asarray(frame_across(TURNED)).reshape(3, 3)}
#: The truncation's source radius: the table's sources reach 0.9 of the radius, and the series' truncation is set by
#: its largest source, so the series is taken with one there, its column unread.
REACH = (0.9, 0.0, 0.0)
#: The distances from the axis the rule is tested at, in the tube's radius.
NEAR = (0.0, 1e-8, 1e-3, 0.01, 0.02)
#: Samples off the axis, in the unit tube's frame, and the direction a sample is moved through the axis along.
OFF = (0.27 * math.cos(2.1), 0.27 * math.sin(2.1), -0.13)
OTHER = (0.36 * math.cos(-0.7), 0.36 * math.sin(-0.7), 0.22)
THROUGH = np.array([math.cos(0.9), math.sin(0.9), 0.0])


@pytest.fixture(scope="module")
def table():
    return builder_table(RADII, ANGLES, OFFSETS).arrays(jnp.float64)


def near(rho: float, angle: float, height: float) -> list[float]:
    return [rho * math.cos(angle), rho * math.sin(angle), height]


def configurations(rho: float) -> dict[str, np.ndarray]:
    """Three samples in the unit tube's frame: one ``rho`` from the axis with two off it (the near one the source of
    some pairs and the target of others), and two that near with one off it (the pair between them both near)."""
    return {"one near": np.array([near(rho, 0.4, 0.1), OFF, OTHER]),
            "both near": np.array([near(rho, 0.4, 0.1), near(0.7 * rho, 2.5, -0.05), OFF])}


def placed(local: np.ndarray, frame: np.ndarray) -> Array:
    return jnp.asarray(CENTRE + (np.asarray(local) * TUBE) @ frame)


def matrix_at(points: Array, frame: Array, table) -> Array:
    return tube_correction(points, {"centre": tuple(CENTRE), "frame": frame.ravel(), "radius": TUBE, "table": table},
                           MU).matrix


@jax.jit
def derivatives(points: Array, frame: Array, table) -> tuple[Array, Array]:
    """The matrix's derivatives in the samples' positions, reverse (of a weighted sum of its entries) and forward."""
    count = 3 * points.shape[0]
    weights = jnp.arange(1.0, 1.0 + count ** 2).reshape(count, count)
    reverse = jax.grad(lambda at: jnp.sum(matrix_at(at, frame, table) * weights))(points)
    return reverse, jax.jacfwd(lambda at: matrix_at(at, frame, table))(points)


def blocks(local: np.ndarray, table, frame: np.ndarray = FRAMES["along z"]) -> tuple[np.ndarray, bool]:
    """``(t, s, 3, 3)``: the tube's correction at samples given in the unit tube's frame, placed in the tube of
    ``frame``, Cartesian in the unit tube's frame at viscosity 1, as the series gives it; and whether every read was
    accepted."""
    found = tube_correction(placed(local, frame), {"centre": tuple(CENTRE), "frame": tuple(frame.ravel()),
                                                   "radius": TUBE, "table": table}, MU)
    count = len(local)
    matrix = np.asarray(found.matrix).reshape(count, 3, count, 3).transpose(0, 2, 1, 3)
    return np.einsum("ai,tsij,bj->tsab", frame, matrix, frame) * TUBE * MU, bool(found.accepted)


def series(targets: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """The series at the table's truncation (``REACH``)."""
    return fb.image(targets, np.vstack([sources, [REACH]]), 1.0, 1.0)[:, :len(sources)]


def misses(local: np.ndarray, table, frame: np.ndarray = FRAMES["along z"]) -> np.ndarray:
    """Each pair's block against the series at the same samples, relative to the pair's largest component."""
    found, accepted = blocks(local, table, frame)
    assert accepted
    image = series(np.asarray(local), np.asarray(local))
    count = len(local)
    return np.array([np.max(np.abs(found[t, s] - image[t, s])) / np.max(np.abs(image[t, s]))
                     for t in range(count) for s in range(count)])


@pytest.fixture(scope="module")
def read_accuracy(table) -> float:
    """The table's read accuracy, measured: the worst miss over every pair of twelve samples off the axis, drawn once
    (a fixed seed) inside its readable range, from 0.05 to 0.45 of the radius and within 0.25 of it along the axis."""
    rng = np.random.default_rng(1)
    drawn = [near(rng.uniform(0.05, 0.45), rng.uniform(0.0, 2 * math.pi), rng.uniform(-0.25, 0.25)) for _ in range(12)]
    return float(np.max(misses(np.array(drawn), table)))


def test_the_tables_knots_are_the_series():
    """The builder's table at its knots against the series, so that a miss of the read below is the read's, not the
    truncation's: a source on the axis and one off it, targets on the axis and off it."""
    data = builder_data(RADII, ANGLES, OFFSETS)
    worst = 0.0
    for source in (0, 5):
        for target, angle, offset in ((0, 3, 2), (5, 3, 2), (2, 7, 4)):
            point = np.array([near(RADII[target], ANGLES[angle], OFFSETS[offset])])
            image = series(point, np.array([[RADII[source], 0.0, 0.0]]))[0, 0]
            cylindrical = fb.to_cylindrical(np.array([ANGLES[angle]]))[0] @ image
            held = data[target, source, angle, offset]
            worst = max(worst, float(np.max(np.abs(held - cylindrical)) / np.max(np.abs(cylindrical))))
    print(f"\nthe table at its knots against the series: worst {worst:.1e}")
    assert worst <= 1e-6


def test_on_the_axis_the_table_holds_one_block_at_every_angle(table):
    """The rule's premise, in the built table: at ``rho_t = 0`` the block, its rows in the source's frame, is the same
    at every angle, to rounding of its largest entry."""
    values = np.asarray(table.values).reshape(len(RADII), len(RADII), -1, len(OFFSETS) + 2, 3, 3)
    spread = max(float(np.max(np.abs(values[0, j] - values[0, j, :1])) / np.max(np.abs(values[0, j])))
                 for j in range(len(RADII)))
    print(f"\nat the axis, the block's spread over the angles: {spread:.1e} of its largest entry")
    assert spread <= 1e-12


@pytest.mark.parametrize("tube", sorted(FRAMES))
@pytest.mark.parametrize("rho", NEAR)
def test_near_the_axis_every_value_and_derivative_is_finite(table, rho, tube):
    frame = FRAMES[tube]
    for name, local in configurations(rho).items():
        found, accepted = blocks(local, table, frame)
        reverse, forward = (np.asarray(d) for d in derivatives(placed(local, frame), jnp.asarray(frame), table))
        assert accepted, name
        assert np.all(np.isfinite(found)), name
        assert np.all(np.isfinite(reverse)) and np.all(np.isfinite(forward)), name


@pytest.mark.parametrize("tube", sorted(FRAMES))
@pytest.mark.parametrize("rho", NEAR)
def test_near_the_axis_the_block_is_the_series_within_the_tables_read_accuracy(table, read_accuracy, rho, tube):
    worst = {name: float(np.max(misses(local, table, FRAMES[tube]))) for name, local in configurations(rho).items()}
    print(f"\nat {rho} of the radius from the axis: worst {worst}; the table's read accuracy off the axis "
          f"{read_accuracy:.2e}")
    assert max(worst.values()) <= read_accuracy


def jump_through_the_axis(table, step: float) -> float:
    """The largest change of the block, over its largest entry, with a sample on the axis (paired with one off it)
    moved ``step`` of the radius either way through it, along a direction across it: the sample the target of one pair
    and the source of the other."""
    held = np.array([OFF, [0.0, 0.0, 0.1]])
    on, _ = blocks(held, table)
    changes = []
    for sign in (1.0, -1.0):
        moved = held.copy()
        moved[1] += sign * step * THROUGH
        found, _ = blocks(moved, table)
        changes.append(np.max(np.abs(found - on)) / np.max(np.abs(on)))
    return float(np.max(changes))  # NaN wherever a block is, so that it fails every bound


def test_a_sample_moved_through_the_axis_moves_the_block_continuously(table):
    found = {step: jump_through_the_axis(table, step) for step in (1e-3, 1e-6, 1e-9)}
    print(f"\nthe block's change, a sample moved through the axis: {found}")
    assert found[1e-9] <= 1e-6 and found[1e-9] < found[1e-6] < found[1e-3]


def todays_rule(across):
    """A sample's direction from its own position, as the read took it before: undefined on the axis."""
    distance = switch.nonzero_norm(across, axis=-1)
    return jnp.sum(across * across, axis=-1), distance, across / distance[:, None]


def test_control_todays_rule_on_the_axis_is_not_a_number(table, monkeypatch):
    monkeypatch.setattr(tube_module, "across_the_axis", todays_rule)
    found, _ = blocks(configurations(0.0)["one near"], table)
    print(f"\ntoday's rule on the axis: {np.sum(~np.isfinite(found))} of {found.size} entries not finite")
    assert not np.all(np.isfinite(found))
    assert not jump_through_the_axis(table, 1e-9) <= 1e-6


def test_control_the_blend_dropped_is_discontinuous_where_a_source_crosses_the_axis(table, monkeypatch):
    monkeypatch.setattr(tube_module, "target_weight", lambda squared: jnp.ones((squared.shape[0], squared.shape[0])))
    found = jump_through_the_axis(table, 1e-9)
    print(f"\nthe blend dropped: the block's change {found:.2e} for a step of 1e-9 through the axis")
    assert found > 1e-6


# -- the readable range ---------------------------------------------------------------------------------------------

def written(path, stated: tuple[float, float] | None) -> str:
    """The table written in the 2026-05-21 form, built at a tube of radius 2 (its lengths twice the unit tube's), with
    ``readable_rho`` where ``stated``."""
    held = {"R_cyl": np.float64(2.0), "rho_grid": 2.0 * np.asarray(RADII), "dphi_grid": np.asarray(ANGLES),
            "dz_grid": 2.0 * np.asarray(OFFSETS), "data": builder_data(RADII, ANGLES, OFFSETS) / 2.0}
    if stated is not None:
        held["readable_rho"] = np.asarray(stated)
    np.savez(path, **held)
    return str(path)


def accepted_at(path: str, rho: float) -> tuple[bool, bool]:
    """Whether a pair of a sample ``rho`` from the axis and one off it is accepted, and whether its block is finite."""
    found, accepted = blocks(np.array([near(rho, 0.4, 0.1), OFF]), tube_table_from(path).arrays(jnp.float64))
    return accepted, bool(np.all(np.isfinite(found)))


def test_the_readable_range_is_the_files_and_a_read_past_it_is_refused(tmp_path):
    """Stated to 0.8 (0.4 of the radius, in the file's lengths): a sample at 0.38 is read, one at 0.42 is refused
    (NaN, not accepted), though the knots reach 0.5; with no range stated, the knots' range reads both, and a sample
    past the knots, at 0.55, is refused."""
    narrowed, own = written(tmp_path / "narrowed.npz", (0.0, 0.8)), written(tmp_path / "own.npz", None)
    held = {name: tube_table_from(path).axes[0].readable_range for name, path in (("narrowed", narrowed), ("own", own))}
    print(f"\nthe radii's readable range: {held}")
    assert held == {"narrowed": (0.0, 0.4), "own": (0.0, 0.5)}
    assert accepted_at(narrowed, 0.38) == (True, True) and accepted_at(narrowed, 0.42) == (False, False)
    assert accepted_at(own, 0.42) == (True, True) and accepted_at(own, 0.55) == (False, False)


def test_a_stated_range_narrows_the_knots_and_never_widens_them(tmp_path):
    path = written(tmp_path / "wide.npz", (0.0, 1.6))
    assert tube_table_from(path).axes[0].readable_range == (0.0, 0.5)
    assert accepted_at(path, 0.55) == (False, False)
