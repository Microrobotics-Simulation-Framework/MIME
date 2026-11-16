"""Spike harness: does the exact allocator's tie window make a closed loop chatter between faces?

A target moves slowly along a path through each tie region; the allocator is called once a tick (lax.scan, one call a
tick, in the dtype's own arithmetic); the counter then reads the command jumps between consecutive ticks. The arms patch
the law's module constant ``TIE_ULPS`` or add the sticky prototype, inside this spike only: nothing under src/ changes.
"""
from __future__ import annotations

import inspect
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests" / "control"))

import jax  # noqa: E402

import mime.control._allocation as law  # noqa: E402
from allocation_draws import provider, reach_of  # noqa: E402
from mime.units import m  # noqa: E402

jnp: Any = jax.numpy
lax: Any = jax.lax

#: The target's relative move per tick (|b_k - b_{k-1}| / |b_{k-1}|) under which a jump is chatter: 0.1 % of |target|.
TOLERANCE = 1e-3
#: F0's threshold (a fifth of the 5 A limit, a tenth of the 10 A box) and a second one for the CardioMag draw, whose two
#: faces are 0.05 to 0.06 A apart (a legitimate move on these paths is at most about 0.01 A a tick).
THRESHOLDS = (1.0, 0.03)
SCALES = np.asarray([1e-2] * 3 + [1.0] * 5)
LIMIT = 5.0
#: float32's 64-ulp window as wide in float64's units (eps32 / eps64 = 2^29).
WIDE64 = 64 * 2 ** 29


@dataclass(frozen=True)
class Spec:
    """A problem and its path: ``u(k) = centre + amp sin(2 pi k / T + phi)``, the target ``w0 e^u``; ``fast`` and ``slow``
    are ``(T, periods)``; the relative move per tick is at most ``2 pi amp / T``."""

    name: str
    matrix: np.ndarray
    scales: np.ndarray
    w0: np.ndarray
    centre: float
    amp: float
    fast: tuple[int, int]
    slow: tuple[int, int]


def specs() -> list[Spec]:
    octo = provider("octomag").actuation(m((0.0, 0.0, 0.0)))
    card = provider("cardiomag").actuation(m((0.0, 0.0, 0.0))).matrix()
    both = octo.matrix()
    w_octo = np.zeros(8)
    w_octo[1] = reach_of(both, LIMIT)
    w_card = np.asarray([0.0, 0.0, 0.03585362502957054, 0.0, 0.0, 0.0, 0.057365800047312863, 0.043024350035484646])
    field = np.asarray(octo.field)
    w_field = np.zeros(3)
    w_field[1] = 0.3 * reach_of(field, LIMIT)
    return [Spec("octomag", both, SCALES, w_octo, 1.0, 1.4, (9000, 3), (90000, 1)),
            Spec("cardiomag", card, SCALES, w_card, 0.0, 0.3, (2000, 3), (20000, 1)),
            Spec("octo_field", field, SCALES[:3], w_field, 0.0, 0.5, (4000, 3), (40000, 1))]


def path(spec: Spec, speed: str, phase: str) -> np.ndarray:
    period, periods = spec.fast if speed == "fast" else spec.slow
    ticks = period * periods
    shift = {"bottom": -np.pi / 2, "top": np.pi / 2}[phase]
    return spec.centre + spec.amp * np.sin(2 * np.pi * np.arange(ticks) / period + shift)


# -- the arms ------------------------------------------------------------------------------------------------------

STICKY: dict[str, Any] = {}


def sticky_selection(ties: int) -> Any:
    """``law._selection`` with one added step, made from the law's own source so that nothing else differs: if the previous
    tick's face (``STICKY['previous']``, a set index, -1 for none) is among the tied faces, it is returned."""
    source = textwrap.dedent(inspect.getsource(law._selection))
    anchor = "    return switch.argmin(ranked, owner=owner, resolution=lambda: switch.rounding(jnp.min(ranked)),"
    assert source.count(anchor) == 1, "the law's last line moved: the prototype must be re-anchored"
    added = ("    previous = STICKY['previous']\n"
             "    held = jnp.logical_and(previous >= 0, tied[jnp.maximum(previous, 0)])\n"
             "    ranked = jnp.where(jnp.logical_and(held, jnp.arange(ranked.shape[0]) == previous), -1.0, ranked)\n")
    source = source.replace(anchor, added + anchor)
    namespace = dict(vars(law))
    namespace.update(STICKY=STICKY, TIE_ULPS=ties)
    exec(compile(source, "<sticky prototype>", "exec"), namespace)
    return namespace["_selection"]


@dataclass(frozen=True)
class Arm:
    name: str
    ties: int
    sticky: bool = False
    dtypes: tuple[str, ...] = ("float32", "float64")


ARMS = {a.name: a for a in (Arm("tie0", 0), Arm("tie16", 16), Arm("tie64", 64), Arm("tie128", 128),
                            Arm("sticky64", 64, True), Arm("w64x", WIDE64, dtypes=("float64",)))}


def run_law(spec: Spec, targets: np.ndarray, arm: Arm, dtype: str) -> dict[str, Any]:
    """The commands and the face (set index) at every tick of the path, in ``dtype``; the traced window is recorded."""
    jax.config.update("jax_enable_x64", dtype == "float64")
    jax.clear_caches()
    commands = spec.matrix.shape[1]
    signs = jnp.asarray(law._sets(commands, spec.matrix.shape[0]).signs)
    matrix, scales = jnp.asarray(spec.matrix, dtype=dtype), jnp.asarray(spec.scales, dtype=dtype)
    traced: list[Any] = []
    original = law._selection
    if arm.sticky:
        replacement = sticky_selection(arm.ties)
        def hooked(*args: Any, **kwargs: Any) -> Any:
            traced.append(("sticky", replacement.__globals__["TIE_ULPS"]))
            return replacement(*args, **kwargs)
    else:
        def hooked(*args: Any, **kwargs: Any) -> Any:
            traced.append(("plain", law.TIE_ULPS))
            return original(*args, **kwargs)

    def tick(previous: Any, target: Any) -> Any:
        STICKY["previous"] = previous
        found = law.exact_allocation(matrix, target, lower=-LIMIT, upper=LIMIT, scales=scales, owner="spike")
        face = jnp.argmax(jnp.all(signs == found.active[None, :], axis=1)).astype(jnp.int32)
        return face, (found.commands, face)

    law.TIE_ULPS = arm.ties
    law._selection = hooked
    try:
        scan = jax.jit(lambda ts: lax.scan(tick, jnp.int32(-1), ts)[1])  # a fresh function: a fresh trace
        out_commands, out_face = scan(jnp.asarray(targets, dtype=dtype))
        out_commands = np.asarray(out_commands, dtype=np.float64)
        out_face = np.asarray(out_face)
    finally:
        law._selection = original
        law.TIE_ULPS = 64
        STICKY.clear()
    assert traced, "the arm's variant was never traced"
    return {"commands": out_commands, "face": out_face, "traced": sorted(set(traced))}


# -- the counter ---------------------------------------------------------------------------------------------------

def count(commands: np.ndarray, targets: np.ndarray, scales: np.ndarray, u: np.ndarray, face: np.ndarray | None = None,
          ) -> dict[str, Any]:
    """Chatter by F0's definition: a tick whose largest command jump exceeds the threshold while the scaled target moved
    less than ``TOLERANCE`` of itself. ``targets`` is what the law was given (rounded as it was)."""
    scaled = targets.astype(np.float64) / scales
    jump = np.max(np.abs(np.diff(commands, axis=0)), axis=1)
    move = np.linalg.norm(np.diff(scaled, axis=0), axis=1) / np.linalg.norm(scaled[:-1], axis=1)
    quiet = move < TOLERANCE
    result: dict[str, Any] = {"ticks": int(len(commands)), "max_relative_move": float(move.max()),
                              "all_moves_under_tolerance": bool(quiet.all()), "max_jump": float(jump.max())}
    for thr in THRESHOLDS:
        hits = np.flatnonzero((jump > thr) & quiet)
        result[f"switches_{thr:g}A"] = int(len(hits))
        result[f"switch_u_{thr:g}A"] = [round(float(u[i + 1]), 4) for i in hits[:40]]
    result["jumps_over_1A_with_the_target_moving_over_tolerance"] = int(np.sum((jump > 1.0) & ~quiet))
    if face is not None:
        result["face_changes"] = int(np.sum(np.diff(face) != 0))
    return result


# -- the controls --------------------------------------------------------------------------------------------------

def planted(runs: int = 20) -> dict[str, Any]:
    """A law that alternates between two faces every tick, through the same counter, on 20 randomly drawn paths and jump
    sizes: ``ticks - 1`` switches every run at jump sizes above the threshold; none where the jump is 0.9 of the 1 A
    threshold (it counts only at 0.03 A); none where the target moves 5 % a tick (over the tolerance)."""
    rng = np.random.default_rng(0)
    caught, quiet_small, quiet_fast = 0, 0, 0
    small_counted_at_low = 0
    for _ in range(runs):
        ticks = int(rng.integers(500, 3000))
        u = rng.uniform(-1, 1) + 0.3 * np.sin(2 * np.pi * np.arange(ticks) / rng.integers(2000, 20000))
        base = np.exp(u)[:, None] * np.array([1.0, 0.5, 0.25])
        size = float(rng.uniform(2.0, 10.0))
        alternate = np.zeros((ticks, 4))
        alternate[::2, 0] = size
        r = count(alternate, base, np.ones(3), u)
        caught += r["switches_1A"] == ticks - 1
        small = np.zeros((ticks, 4))
        small[::2, 0] = 0.9
        r = count(small, base, np.ones(3), u)
        quiet_small += r["switches_1A"] == 0
        small_counted_at_low += r["switches_0.03A"] == ticks - 1
        fast_u = u[0] + 0.05 * np.arange(ticks) % 1.0  # 5 % a tick
        r = count(alternate, np.exp(fast_u)[:, None] * np.array([1.0, 0.5, 0.25]), np.ones(3), fast_u)
        quiet_fast += r["switches_1A"] == 0
    return {"runs": runs, "alternating_law_caught_every_tick": int(caught), "under_threshold_jump_not_counted": int(quiet_small),
            "under_threshold_jump_counted_at_0.03A": int(small_counted_at_low), "fast_target_not_counted": int(quiet_fast)}
