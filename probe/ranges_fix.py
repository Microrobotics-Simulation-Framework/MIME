"""Oracles for discriminating_range.py (the fix round of PR 267), each deterministic (one draw a size), True when the
gate fails with a plant of ``size``: the cell read and its transpose scaled by ``1 + size`` (gate 1 with the cell read,
the review's M3); the own response's hand-written transpose scaled by ``1 + size`` (the reverse gate's forces block,
both dtypes); the given values read at the corner of ``size`` fine cells of motion before (gate 5's snap check)."""
import os, sys
sys.path[:0] = ["tests", "tests/patch"]
import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.flow.fvm import _patch

READ, BACK, TRANSPOSED = _patch.cell_read, _patch.cell_read_transpose, _patch._own_transposed


def cell(size: float, seed: int) -> bool:
    import dataclasses
    import test_patch_plane as plane
    from patch_cases import image_reader, normal_k, patch_reader, resistance
    from plane_oracles import brenner
    jax.config.update("jax_enable_x64", True)
    factor = 1.0 + size

    def scaled_read(*args):
        values, outside = READ(*args)
        return factor * values, outside

    def scaled_back(*args):
        return tuple(factor * part for part in BACK(*args))

    _patch.cell_read, _patch.cell_read_transpose = scaled_read, scaled_back
    try:
        case = dataclasses.replace(plane.CASE, read="cell")
        image = normal_k(resistance(case, image_reader(case, "cell"), ("Uz",)))
        reader, _ = patch_reader(case)
        share = 100.0 * (normal_k(resistance(case, reader, ("Uz",))) - image) / brenner(case.gap)
    finally:
        _patch.cell_read, _patch.cell_read_transpose = READ, BACK
    low, high = plane.SHARE_BAND["cell"]
    print("cell", size, share, flush=True)
    return not (low <= share <= high)


def transpose(size: float, seed: int) -> bool:
    import test_patch_reverse as reverse
    from mime.nodes.flow.fvm._patch import patch_reflection, patch_reflection_transpose, patch_stopping
    from mime.nodes.flow.fvm._solve import REPORT_SIZE
    _patch._own_transposed = lambda place, cotangent: (1.0 + size) * TRANSPOSED(place, cotangent)
    failed = False
    try:
        for name in ("float64", "float32"):
            jax.config.update("jax_enable_x64", name == "float64")
            dtype = jnp.dtype(name)
            for read in ("cell", "point"):
                patch, placement, rng = reverse.case(dtype, read)
                boundary, forces, cotangent = reverse.drawn(rng, patch, placement, dtype)
                zero_b = tuple(jnp.zeros_like(b) for b in boundary)
                stopping = patch_stopping(patch, dtype, tolerance=reverse.tolerance_of(dtype))
                tap = jnp.zeros(REPORT_SIZE, dtype=dtype)
                values = jax.jit(lambda b, f: patch_reflection(patch, placement, b, f, None, stopping, tap).values)(
                    zero_b, forces)
                _, force, _ = jax.jit(lambda c: patch_reflection_transpose(patch, placement, c, None, stopping, tap))(
                    cotangent)
                left, right = reverse.pairing(values, cotangent), reverse.pairing(forces, force)
                scale = reverse.size_of(values) * reverse.size_of(cotangent) + reverse.size_of(forces) * \
                    reverse.size_of(force)
                departure = abs(left - right) / scale
                print("transpose", size, name, read, departure, flush=True)
                failed = failed or departure > reverse.identity_line(dtype)
    finally:
        _patch._own_transposed = TRANSPOSED
        jax.config.update("jax_enable_x64", True)
    return failed


def snap(size: float, seed: int) -> bool:
    import test_patch_motion as motion
    jax.config.update("jax_enable_x64", True)
    held = os.environ.get("PATCH_PLANT")
    if size > 0.0:
        os.environ["PATCH_PLANT"] = "stale_given"
        motion.STALE_LAG = size
    try:
        patch, sphere, forces = motion.setting()
        m = jax.jit(motion.metric_function(patch, sphere, forces))
        step = 0.125
        offsets = np.arange(17) * step
        origins = [np.asarray(motion.snapped_origin(patch, motion.pose(u))) for u in offsets]
        first = next(k for k in range(1, len(offsets)) if not np.array_equal(origins[k], origins[k - 1]))
        flip = motion.snap_point(patch, offsets[first - 1], offsets[first])
        below = float(m(motion.pose(flip - motion.SNAP_STRADDLE), jnp.asarray(0.0)))
        above = float(m(motion.pose(flip + motion.SNAP_STRADDLE), jnp.asarray(0.0)))
    finally:
        if held is None:
            os.environ.pop("PATCH_PLANT", None)
        else:
            os.environ["PATCH_PLANT"] = held
        motion.STALE_LAG = 0.04
    jump = abs(above - below) / abs(below)
    print("snap", size, jump, flush=True)
    return jump > motion.SNAP_LINE
