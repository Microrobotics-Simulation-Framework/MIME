"""The CG's stagnation stop is reachable before its cap (``mime.nodes.flow.fvm._solve.restart_fits``): a stagnation
restarts the CG only where another whole window fits before the cap, otherwise it stops there, flagged.

The cause (F2b 4a, a float32 tube tree 73 by 49 by 109 at the default 1e-5): the window (66 checks, 1320 iterations)
and three restarts need 5280 iterations of a 5000 cap, so a plateau at the solve's rounding floor ran every solve to
the cap, flagged there and not stagnated. A window cut to fit the four of them in the cap still missed it, because the
plateau sets in some hundreds of iterations after the start and the stop needs every window after that.

Its gates (``predictions_cg_stagnation_reach.json``): a float32 solve whose true residual plateaus above its tolerance
(its tolerance below float32's rounding) is stopped as stagnated, before the cap, at the tree's window, at a small
size's, and under a small cap; a solve that restarts at a stall and converges is bit for bit what it was. The control
for the plateau gates is the restart always taken (``MIME_PLANT_STAGNATION=old_window``), whose rate
``.github/probe/detection_rate.py`` measures; each draw is seeded by ``DETECTION_SEED``."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fvm_cases import in_dtype
from test_fvm_residual_replacement import STALLING, StenosisSolve

import mime.nodes.flow.fvm._solve as solve_module
from mime.nodes.flow.fvm import Stopping, guarded_cg
from mime.nodes.flow.fvm._solve import DEFAULT_CAP, REPORT_FIELDS, stagnation_window

ITERATIONS, AT_CAP, STAGNATED = (REPORT_FIELDS.index(name) for name in ("iterations", "at_cap", "stagnated"))
TREE, SMALL = (73, 49, 109), (8, 8, 8)
POINTS, DRAWS = 48, 12


def _plant() -> str:
    return os.environ.get("MIME_PLANT_STAGNATION", "")


def _seed() -> int:
    return int(os.environ.get("DETECTION_SEED", "0"))


def _plateau(draw: int, stopping: Stopping) -> np.ndarray:
    """The report of a float32 CG on a periodic Laplacian of ``POINTS`` points a component (shifted by 1e-3, three
    components, the identity preconditioner) at a tolerance below float32's rounding: it converges in some tens of
    iterations and then its true residual sits at the rounding floor, a plateau above the tolerance."""
    rng = np.random.default_rng(1000 * _seed() + draw)

    def lap(u: jax.Array) -> jax.Array:
        return 2.0 * u - jnp.roll(u, 1, 0) - jnp.roll(u, -1, 0) + 1e-3 * u

    def inner(a: tuple[jax.Array, ...], c: tuple[jax.Array, ...]) -> jax.Array:
        return sum(jnp.sum(p * q) for p, q in zip(a, c, strict=True))

    b = tuple(jnp.asarray(rng.standard_normal(POINTS), jnp.float32) for _ in range(3))
    zero = tuple(jnp.zeros_like(c) for c in b)
    with in_dtype("float32"):
        _, report = jax.jit(lambda rhs: guarded_cg(lambda x: tuple(lap(c) for c in x), rhs, lambda r: r, zero,
                                                   stopping, inner))(b)
    return np.asarray(report)


def _stopped_stagnated_before_the_cap(cells: tuple[int, ...], cap: int, draw: int) -> bool:
    stopping = Stopping(tolerance=jnp.float32(1e-12), cap=cap, window=stagnation_window(cells, 20))
    report = _plateau(draw, stopping)
    return bool(report[STAGNATED] == 1.0 and report[AT_CAP] == 0.0 and report[ITERATIONS] < cap)


@pytest.fixture(autouse=True)
def planted_old_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control: the restart taken whether or not another window fits, as before the fix."""
    if _plant() == "old_window":
        monkeypatch.setattr(solve_module, "restart_fits", lambda count, stopping: jnp.asarray(True))


@pytest.mark.parametrize("draw", range(DRAWS))
def test_a_plateau_stops_stagnated_before_the_cap_at_the_tree_size(draw: int) -> None:
    """The tube tree's window (66 checks) under the default cap of 5000: the stop is reached, flagged, with the cap
    unreached. Before the fix every draw ran to the cap."""
    assert stagnation_window(TREE, 20) * 20 * (solve_module.STALL_RESTARTS + 1) > DEFAULT_CAP  # the premise
    assert _stopped_stagnated_before_the_cap(TREE, DEFAULT_CAP, draw)


@pytest.mark.parametrize("draw", range(DRAWS))
def test_a_plateau_stops_stagnated_before_the_cap_at_a_small_size(draw: int) -> None:
    """A small box's window (5 checks) fits its restarts in the cap many times over: the stop was reachable, and is
    the same. Not a discriminating gate (the control also stops here); it shows the rule moves nothing where the
    window already fit."""
    assert stagnation_window(SMALL, 20) * 20 * (solve_module.STALL_RESTARTS + 1) <= DEFAULT_CAP
    assert _stopped_stagnated_before_the_cap(SMALL, DEFAULT_CAP, draw)


@pytest.mark.parametrize("draw", range(DRAWS))
def test_a_plateau_stops_stagnated_before_a_small_cap_at_the_tree_size(draw: int) -> None:
    """A cap of 1000 holds less than one tube-tree window (1320 iterations): no stagnation can be found in it, and the
    solve ends at the cap, flagged there: a stop that cannot be reached is not claimed. At a cap of 2000 one window
    fits and a second does not: the stop is the first stagnation found, with no restart."""
    assert not _stopped_stagnated_before_the_cap(TREE, 1000, draw)
    assert _stopped_stagnated_before_the_cap(TREE, 2000, draw)


def test_a_stagnation_with_a_window_to_spare_restarts_and_one_without_stops() -> None:
    """The seam itself, over a window of 66 checks every 20 iterations and a cap of 5000: a stagnation found at 1340,
    2660 or 3680 restarts (a further window ends at 2660, 3980, 5000), one found at 3700 or later stops."""
    stopping = Stopping(tolerance=1e-5, window=66)
    fits = {count: bool(solve_module.restart_fits(jnp.asarray(count), stopping)) for count in (1340, 2660, 3680, 3700)}
    assert fits == {1340: True, 2660: True, 3680: True, 3700: False}


@pytest.mark.parametrize("shape", STALLING[:2])
def test_a_solve_that_restarts_at_a_stall_and_converges_is_bit_for_bit_unchanged(shape: tuple[float, ...]) -> None:
    """A float32 stenosis solve that stalls without its restart and converges with it (the control of
    ``test_fvm_residual_replacement``), run with the restart always taken and with the rule: the same answer and the
    same report, every bit. The stall is found early in its cap, so the rule has no say."""
    with in_dtype("float32"):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(solve_module, "restart_fits", lambda count, stopping: jnp.asarray(True))
            before, before_report = StenosisSolve("float32", 8)(shape)
        after, after_report = StenosisSolve("float32", 8)(shape)
    assert before_report[STAGNATED] == 0.0 and before_report[ITERATIONS] > 0
    assert np.array_equal(before_report, after_report)
    assert all(np.array_equal(np.asarray(a), np.asarray(c)) for a, c in zip(before, after, strict=True))
