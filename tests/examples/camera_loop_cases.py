"""Shared pieces of the camera loop's tests (``examples/camera_loop.py``): the example loaded once, camera spike R2's
draws, the loop's objective through MIME's differentiated run, and the planted errors the controls switch on.

**The objective** is R2's: ``J = mean over the run's steps of |p_n - r(t_n)|^2`` in mm^2, ``p_n`` the sphere's true
position after step ``n`` and ``r`` the reference. It is taken through ``mime.build._maddening.differentiated_run``, the
program ``Study.gradient`` runs (reverse mode in checkpointed chunks, forward mode in one), on the graph a study's plan
builds with the gains as leaves of the controller's node; the believed camera's focal length is a leaf of that node too.

**The draws** (``DETECTION_SEED``, which ``.github/probe/detection_rate.py`` sets for each run; 0 by default) are R2's
round 2: the gains each within 20 % of the example's, the reference's step 4.5 to 6 mm in a drawn direction, drawn
again from the same stream until the loop's first measured tick asks a coil for more than 1.25 times its limit (R2's
seeds were ``7000 + n``; a draw that binds no coil leaves the allocation's controls inert).

**The plants** (``CAMERA_LOOP_PLANT``, read before anything is traced), each a test's control:

- ``mode_bounds``: the allocation's bounds 1 % looser in the GRAPH arm only (the mode test's);
- ``limits``: the allocation's bounds 1 % looser (the BVLS check's, R2's swap plant);
- ``reverse_chunk``: the reverse arm's objective with the last chunk's poses stopped (a chunk's cotangent dropped, the
  failure reverse mode alone can have; a ``stop_gradient`` anywhere in the loop moves forward mode alike);
- ``edge_f32``: the camera's edge pixels' coverage stopped in float32's arithmetic only (float32 against float64: a
  float32-only defect in the camera's derivative, which every frame of every draw exercises);
- ``edge``, ``stop_branch``: the camera's edge pixels' coverage, and the bound coils' share, stopped (the
  central-difference check's);
- ``unguarded``: the tracker without its lost guard, R2's own (the lightly damped loop's);
- ``extra_port``, ``extra_controller``, ``extra_command``: one base step more latency in that home (the limit's);
- ``unnormalised``, ``belief_ignored``: the centroid not divided by its darkness, and the tracker reading the true camera
  in place of its belief (the tracker's gates)."""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import math
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PLANT = os.environ.get("CAMERA_LOOP_PLANT", "")
SEED = int(os.environ.get("DETECTION_SEED", "0") or 0)
#: The gains differentiated, and the believed camera's focal length, by the names the probe's rows use.
GAINS = ("kd", "ki", "kp")
FX = "believed:camera::fx"
STEPS = 200
jnp: Any = jax.numpy


def planted(name: str) -> bool:
    return PLANT == name


def example() -> Any:
    """``examples/camera_loop.py``, loaded once a process (its controllers registered once)."""
    held = sys.modules.get("example_camera_loop")
    if held is not None:
        return held
    spec = importlib.util.spec_from_file_location("example_camera_loop", ROOT / "examples" / "camera_loop.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["example_camera_loop"] = module
    spec.loader.exec_module(module)
    _plant_tracker(module)
    return module


@contextlib.contextmanager
def in_dtype(name: str) -> Iterator[None]:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", name == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


def _plant_tracker(module: Any) -> None:
    if planted("unguarded"):
        def unguarded(frame: Any, model: Any, *, height: Any, radius: float, backlight: float, held: Any) -> Any:
            su, sv, mass, _ = module.soft_centroid(frame, backlight)
            pixel = jnp.stack([su / mass, sv / mass])
            point = module.on_plane(model, pixel, height)
            zero = jnp.zeros((), dtype=frame.dtype)
            return module.Tracked(point, pixel, mass, jnp.zeros((), bool), zero, mass)

        module.tracked = unguarded
    elif planted("unnormalised"):
        centroid = module.soft_centroid

        def unnormalised(frame: Any, backlight: float) -> Any:
            su, sv, mass, border = centroid(frame, backlight)
            return su * mass, sv * mass, mass, border

        module.soft_centroid = unnormalised
    if planted("edge") or planted("edge_f32"):
        from mime.sensing import _coverage

        disc = _coverage.disc
        every = planted("edge")

        def edge_stopped(corners_x: Any, corners_y: Any, centre: Any, radius: Any) -> Any:
            share = disc(corners_x, corners_y, centre, radius)
            if not (every or share.dtype == jnp.float32):
                return share
            return jnp.where((share > 0.0) & (share < 1.0), jax.lax.stop_gradient(share), share)

        _coverage.disc = edge_stopped
    if planted("stop_branch"):
        _stop_branch(True)


def _stop_branch(on: bool) -> None:
    """The bound coils' share of the rows, ``A_B u_B``, stopped in the allocation's derivative (``on``) or not."""
    from mime.control import _allocation

    held = getattr(_allocation, "_held_left", None)
    if held is None:
        held = _allocation._left
        _allocation._held_left = held  # type: ignore[attr-defined]
    _allocation._left = (lambda t, s, b: t - jax.lax.stop_gradient(s @ b)) if on else held


@contextlib.contextmanager
def stopped_branch(on: bool) -> Iterator[None]:
    _stop_branch(on)
    try:
        yield
    finally:
        _stop_branch(planted("stop_branch"))


@contextlib.contextmanager
def looser_bounds(on: bool, factor: float = 1.01) -> Iterator[None]:
    """The allocation's bounds ``factor`` times the ports' while the law is made (``on``)."""
    from mime.control import _models

    held = _models.ControlModels._bounds

    def looser(self: Any, *args: Any) -> Any:
        low, high, box = held(self, *args)
        return ({k: v * factor for k, v in low.items()}, {k: v * factor for k, v in high.items()}, box)

    if on:
        _models.ControlModels._bounds = looser  # type: ignore[method-assign]
    try:
        yield
    finally:
        _models.ControlModels._bounds = held  # type: ignore[method-assign]


def drawn(seed: int, reject: bool = True) -> tuple[dict[str, float], tuple[float, float]]:
    """The draw of ``seed``, R2's: the gains within 20 % of the example's, the step 4.5 to 6 mm in a drawn direction,
    drawn again (``reject``) until the first tick's unconstrained currents exceed 1.25 times the limit."""
    module = example()
    rng = np.random.default_rng(7000 + seed)
    defaults = {"kd": 80.0, "ki": 3000.0, "kp": 2500.0}
    while True:
        gains = {name: float(defaults[name] * rng.uniform(0.8, 1.2)) for name in GAINS}
        angle, size = rng.uniform(0, 2 * math.pi), rng.uniform(4.5e-3, 6e-3)
        step = (size * math.cos(angle), size * math.sin(angle))
        if not reject or first_tick_saturates(module, gains, step):
            return gains, step


@functools.cache
def _coils_model() -> Any:
    from mime.control import _models

    return _models.ControlModels(example().experiment().plan()).believed("coils.magnetic")


def force_rows(module: Any, point: Any) -> np.ndarray:
    """The force per unit current on the sphere's moment at ``point`` (N/A, three by three, a column a coil), by the
    coils' forward model (the rows the allocation solves; their field is held to mag_manip by the multipole oracle),
    in float64."""
    with in_dtype("float64"):
        coefficients, _ = _coils_model().affine({"point": jnp.asarray(point, dtype=jnp.float64),
                                                 "moment": jnp.asarray(module.MOMENT, dtype=jnp.float64)},
                                                commands=("currents",), dtype=jnp.float64)
        return np.asarray(coefficients["force"]["currents"], dtype=np.float64)


def first_tick_saturates(module: Any, gains: Mapping[str, float], step: tuple[float, float]) -> bool:
    """R2's redraw rule, as R2 wrote it: the unconstrained currents of the rows at the start for the first tick's demand
    ``m (kp + ki T) e0``, ``e0`` the step, exceed 1.25 times the limit somewhere. (The example's first tick reads the
    camera before its first frame and is lost; the second asks nearly the same, the sphere not having moved.)"""
    error = np.array([step[0], step[1], 0.0])
    wanted = module.MASS * (gains["kp"] + gains["ki"] * 5e-3) * error
    return bool(np.max(np.abs(np.linalg.solve(force_rows(module, module.START), wanted))) >= 1.25 * module.LIMIT)


def leafy(experiment: Any, gains: Mapping[str, float]) -> Any:
    """The experiment built as a study's point builds it, the gains leaves of the controller's node."""
    from mime import Study
    from mime.study import grid
    from mime.units import ms

    study = Study(experiment).sweep(grid({f"controller.{name}": [value] for name, value in gains.items()}),
                                    duration=ms(STEPS))
    return study.plan().groups[0].built


def reference_track(module: Any, step: tuple[float, float], steps: int, dtype: Any) -> Any:
    times = (np.arange(steps) + 1) * 1e-3
    return jnp.asarray(np.asarray(module.reference(jnp.asarray(times), step)).T, dtype=dtype)


def objective(built: Any, step: tuple[float, float], steps: int = STEPS, *, checkpoint: bool,
              stop_last: int = 0) -> Callable[[Mapping[str, Any]], tuple[Any, Any]]:
    """``J(theta)`` and its per-step terms and the controller's writes, ``theta`` the gains and ``FX`` by name, through the
    differentiated run of ``built``'s graph; ``stop_last`` steps' poses stopped (the ``reverse_chunk`` plant)."""
    from mime.build._maddening import differentiated_run, full_state, graph_params
    from mime.nodes.control._controller import leaf_param

    module = example()
    graph = built.graph
    graph.reset_state()
    state = {node: dict(fields) for node, fields in full_state(graph).items()}
    params = graph_params(graph)
    dtype = jnp.zeros(()).dtype
    track = reference_track(module, step, steps, dtype)
    kept = (("sphere", "pose"), *(("controller", f) for f in ("write_currents", "state_lost", "state_point", "state_wanted",
                                                              "state_rows")))
    run = differentiated_run(graph, steps, (), kept, checkpoint=checkpoint)

    def at(theta: Mapping[str, Any]) -> tuple[Any, Any]:
        nodes = {node: dict(leaves) for node, leaves in params["nodes"].items()}
        for name, value in theta.items():
            nodes["controller"][leaf_param(name) if name.startswith("believed:") else f"setting_{name}"] = value
        _, _, out = run(state, {**params, "nodes": nodes})
        poses = out["sphere"]["pose"][:, :3]
        if stop_last:
            poses = jnp.concatenate([poses[:-stop_last], jax.lax.stop_gradient(poses[-stop_last:])])
        miss = poses - track
        per_step = jnp.sum(miss * miss, axis=1) / 1e-6 / steps
        return jnp.sum(per_step), (per_step, out)

    return at


def theta_of(built: Any, gains: Mapping[str, float]) -> dict[str, Any]:
    from mime.nodes.control._controller import leaf_param

    dtype = jnp.zeros(()).dtype
    held = built.graph.params["nodes"]["controller"]
    found = {name: jnp.asarray(gains[name], dtype=dtype) for name in GAINS}
    if leaf_param(FX) in held:  # the camera arm's: the true-state arm reads no camera
        found[FX] = jnp.asarray(held[leaf_param(FX)], dtype=dtype)
    return found


def difference_verdict(rows: list[tuple[float, float, float]], noise: float, quantum: float,
                       eps: float) -> dict[str, Any]:
    """The central-difference check of both test files, on ``rows`` ``(D(h_k), second difference, h_k)`` at ``h`` to
    ``h/16``: the order ``p`` measured from the last three levels and clamped to [1/2, 2], the finest's truncation
    ``tau = |D(h/8) - D(h/16)| / (2^p - 1)``, its rounding ``rho = r / (h/16)``, the line ``3 tau + 3 rho`` (a kink's,
    ``|D2(h/16)| / (h/8) + 3 rho``, where the second difference falls as ``h``) plus the difference's own ``16 eps``.

    ``r`` is one evaluation's rounding: three times ``noise`` (ECnoise's estimate of the objective's), and never under
    ``quantum``, the objective's own rounding bound (``eps`` times its terms' absolute sum, the run's coordinates
    rounded in its dtype). ECnoise reads nothing where the objective is quantised and its steps are equal: a float32
    objective of about 35 moved in steps of exactly 4 units in the last place has fourth differences of exactly 0, and
    a line of 0."""
    (d2, _, _), (d3, s3, _), (d4, s4, h4) = rows[2], rows[3], rows[4]
    a, b = abs(d2 - d3), abs(d3 - d4)
    order = min(2.0, max(0.5, math.log2(a / b))) if a > 0 and b > 0 else 2.0
    rounding = max(3.0 * noise, quantum)
    tau, rho = b / (2 ** order - 1), rounding / h4
    kinked = abs(s4) > max(6.0 * noise, 4.0 * quantum) and s3 != 0 and abs(s4 / s3) > 0.375
    line = (abs(s4) / (2 * h4) if kinked else 3.0 * tau) + 3.0 * rho + 16 * eps * abs(d4)
    return {"cd": d4, "order": order, "tau": tau, "rho": rho, "kinked": kinked, "line": line}


def ecnoise(table: Any) -> float:
    """More and Wild's ECnoise: the objective's rounding noise from the fourth differences of seven values."""
    return float(math.sqrt(np.mean(np.diff(np.asarray(table, dtype=np.float64), 4) ** 2) * math.factorial(4) ** 2
                           / math.factorial(8)))


def estimated() -> Any:
    """The camera arm at the example's defaults: the probe's memory estimate's factory (``mem_estimate.py --case``)."""
    return example().experiment()
