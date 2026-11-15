"""``examples/camera_loop.py``'s tracker, as the loop runs it: the soft centroid of the camera's expected frame,
back-projected on the declared plane through the controller's believed camera, its lost-target guard and its
out-of-plane flag (camera PR 3b's gates, which moved here when the tracker became example code).

The rig is the loop open: the example's experiment with the gains at nothing, so the law writes no current and the
sphere stays where its start state puts it; the controller's state holds what the tracker found at each tick
(``state_point``), read through MIME's differentiated run (one compiled program a dtype, the sphere's start moved in
the state it is given). The frame read at the tick of step ``n`` was made at step ``n - 1`` (the camera's ``age`` is
nought at every step at a frame period of one), so from the second tick on the tracker sees the sphere where it stands.

**What these bound.** The tracker's 3-D point against the believed back-projection of the ball's silhouette's centroid,
on the plane, within the soft centroid's own error; its derivative against central differences; the guard and the flag
on frames that need them. They do not bound how well the loop tracks (``test_camera_loop.py``), and depth is never
measured: the out-of-plane bias is stated, from the geometry, and checked.

**The tolerance of the centroid** (from its mechanism): the ball's silhouette through a pinhole is an ellipse, and the
camera covers each pixel by its exact area inside it (``mime.sensing._coverage.disc``), so the frame's darkness has the
ellipse's area and centroid, the ellipse's centre (in normalised coordinates ``u_x u_z / (u_z^2 - s^2)``, ``u`` the
direction to the ball's centre and ``s`` its radius over its distance: off the axis, outward of the centre's projection,
the conic centroid's bias). The soft centroid weighs each pixel at its centre, which misses the true centroid by each
part-covered pixel's first moment about its centre, at most 1/8 of a pixel along each axis (half the pixel covered on
one side); so along each axis it is within ``n_edge / (8 mass)`` pixels of the ellipse's centre, ``n_edge`` the pixels
part covered, plus ``64 eps`` of the frame's width for rounding. On the plane that is the back-projection's Jacobian
times it.

**The derivative** in the sphere's start (x and y, moved in the start state), the camera's own focal length and
principal point (its node's leaves) and the believed focal length (the controller's): forward mode against central
differences from ``h`` to ``h/16``, the order measured (the exact disc coverage makes the centroid C^(1,1/2) in a
position), as ``test_camera_loop.py`` judges it, from the build's state and from a state 7 steps on; float32 against
float64 with the camera's gate T3 line where central differences cannot judge.

**The controls** (``CAMERA_LOOP_PLANT``): ``unnormalised`` (the centroid not divided by its darkness), ``belief_ignored``
(the tracker reading the true camera where a belief differs) and ``unguarded`` (no lost guard)."""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import jax
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import camera_loop_cases as cases

jnp: Any = jax.numpy
EX = cases.example()
STEPS = 6
#: The believed camera of the miscalibrated arm: its focal length 2 % long, its principal point 1.5 px right.
MISCALIBRATED = dict(fx=EX.FOCAL * 1.02, cx=(EX.WIDTH - 1) / 2 + 1.5)
#: Where the sphere is drawn, from the start (m): inside the frame by more than its silhouette (the frame is 24 by 18 mm
#: at the plane, the ball 2 mm across).
REACH = (9e-3, 6e-3)


def believed_camera() -> Any:
    from mime.observations import Detector

    return EX.camera(Detector(width=EX.WIDTH, height=EX.HEIGHT, fy=EX.FOCAL, cy=(EX.HEIGHT - 1) / 2, **MISCALIBRATED),
                     frame_ms=1)


@cache
def open_loop(dtype: str, miscalibrated: bool) -> Any:
    """The open loop in ``dtype``: ``(offset, theta, start) -> outputs``, the sphere's start moved by ``offset`` (m),
    ``theta`` the camera's and the belief's leaves by name, ``start`` the state the run starts from (the build's
    where ``None``); and the build, its start state and its leaves."""
    from mime.build._maddening import differentiated_run, full_state, graph_params
    from mime.control import _models

    with cases.in_dtype(dtype):
        held = _models.ControlModels._declared
        if cases.planted("belief_ignored"):
            _models.ControlModels._declared = lambda self, key: None  # type: ignore[method-assign]
        try:
            built = EX.experiment(kp=0.0, ki=0.0, kd=0.0, frame_ms=1, rate_ms=1, latency_ms=0,
                                  believed=believed_camera() if miscalibrated else None).build()
        finally:
            _models.ControlModels._declared = held  # type: ignore[method-assign]
        graph = built.graph
        graph.reset_state()
        start = {node: dict(fields) for node, fields in full_state(graph).items()}
        params = graph_params(graph)
        kept = (("camera:", "expected"), *(("controller", f) for f in ("state_point", "state_lost", "state_off_plane",
                                                                        "state_depth_miss")))
        run = differentiated_run(graph, STEPS, (), kept, checkpoint=False)

    def outputs(offset: Any, theta: dict[str, Any], state: Any = None, velocity: Any = None) -> Any:
        held_state = state if state is not None else start
        moved = {node: dict(fields) for node, fields in held_state.items()}
        pose = moved["sphere"]["pose"]
        moved["sphere"]["pose"] = pose.at[:3].add(jnp.asarray(offset, dtype=pose.dtype))
        if velocity is not None:
            twist = moved["sphere"]["twist"]
            moved["sphere"]["twist"] = twist.at[:3].set(jnp.asarray(velocity, dtype=twist.dtype))
        nodes = {node: dict(leaves) for node, leaves in params["nodes"].items()}
        for (node, leaf), value in theta.items():
            nodes[node][leaf] = value
        final, _, out = run(moved, {**params, "nodes": nodes})
        return final, out

    return jax.jit(outputs), built, start, params


def ellipse_centre(model: Any, centre: np.ndarray) -> np.ndarray:
    """The pixel of the centre of a ball's silhouette (radius ``EX.RADIUS``, centred at ``centre``) through ``model``'s
    pinhole, distortion-free: its area centroid."""
    theta = {k: np.asarray(v, dtype=np.float64) for k, v in model.theta.items()}
    w, x, y, z = theta["orientation"]
    rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    local = (centre - theta["position"]) @ rotation
    distance = float(np.linalg.norm(local))
    u = local / distance
    s2 = (EX.RADIUS / distance) ** 2
    across = u[2] ** 2 - s2
    return np.array([theta["fx"] * u[0] * u[2] / across + theta["cx"], theta["fy"] * u[1] * u[2] / across + theta["cy"]])


@cache
def models() -> tuple[Any, Any]:
    """The true camera's model and the miscalibrated belief's, as a controller reads them."""
    from mime.control._models import camera_model

    return camera_model("camera:", EX.camera(frame_ms=1)), camera_model("camera:", believed_camera())


@cache
def back_projection(miscalibrated: bool) -> Any:
    """The believed back-projection of a pixel on the plane and its Jacobian in the pixel (m per px), in float64:
    through the true camera's model, or the miscalibrated belief's."""
    model = models()[1 if miscalibrated else 0]
    with cases.in_dtype("float64"):
        def at(pixel: Any) -> Any:
            return EX.on_plane(model, pixel, jnp.asarray(EX.START[2], dtype=jnp.float64))

        held = jax.jit(lambda pixel: (at(pixel), jax.jacfwd(at)(pixel)))

    def projected(pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with cases.in_dtype("float64"):
            point, jacobian = held(jnp.asarray(pixel, dtype=jnp.float64))
            return np.asarray(point), np.asarray(jacobian)

    return projected


def found(dtype: str, miscalibrated: bool, offset: np.ndarray) -> dict[str, np.ndarray]:
    outputs, _, _, _ = open_loop(dtype, miscalibrated)
    with cases.in_dtype(dtype):
        _, out = outputs(jnp.asarray(offset), {})
    return {f"{node}.{field}": np.asarray(value, dtype=np.float64) for node, fields in out.items()
            for field, value in fields.items()}


# -- the point, against the believed back-projection of the silhouette's centroid ----------------------------------

@pytest.mark.parametrize("dtype", ["float64", "float32"])
@settings(deadline=None, print_blob=True, max_examples=100)
@given(x=st.floats(-REACH[0], REACH[0]), y=st.floats(-REACH[1], REACH[1]))
def test_the_tracker_finds_the_ball_within_its_soft_centroids_own_error(dtype, x, y):
    offset = np.array([x, y, 0.0])
    centre = np.asarray(EX.START) + offset
    truth, _ = models()
    for miscalibrated in (False, True):
        seen = found(dtype, miscalibrated, offset)
        expected, line = centroid_line(dtype, seen["camera:.expected"][-1], truth, miscalibrated, centre)
        miss = float(np.linalg.norm(seen["controller.state_point"][-1][:2] - expected[:2]))
        assert seen["controller.state_lost"][-1] == seen["controller.state_lost"][1], "a frame in view was lost"
        assert miss <= line, (dtype, miscalibrated, offset, miss, line)


def centroid_line(dtype: str, frame: np.ndarray, truth: Any, miscalibrated: bool,
                  centre: np.ndarray) -> tuple[np.ndarray, float]:
    """The point the belief (miscalibrated or true) back-projects the true silhouette's centroid to, and the soft centroid's line about it on the
    plane (m): ``n_edge / (8 mass)`` px along each axis plus rounding, through the back-projection's Jacobian."""
    dark = (EX.BACKLIGHT - frame) / EX.BACKLIGHT
    rounding = 64 * float(np.finfo(dtype).eps)
    edge = int(np.sum((dark > rounding) & (dark < 1.0 - rounding)))
    error_px = edge / (8.0 * float(np.sum(dark))) + rounding * EX.WIDTH
    expected, jacobian = back_projection(miscalibrated)(ellipse_centre(truth, centre))
    return expected, float(np.linalg.norm(jacobian, 2)) * math.sqrt(2.0) * error_px


# -- the derivative ------------------------------------------------------------------------------------------------

#: The parameters and the scale of each one's central-difference step: the start's x and y (m), the camera's focal
#: length and principal point (px), the believed focal length (px).
PARAMETERS = {"x0": 1e-3, "y0": 1e-3, "camera_fx": EX.FOCAL, "camera_cy": 10.0, "believed_fx": EX.FOCAL}
RELATIVE_H, NOISE_SPACING = {"float64": 1e-4, "float32": 1e-2}, {"float64": 1e-9, "float32": 3e-5}


def objective(dtype: str, continued: bool) -> tuple[Any, dict[str, Any], Any]:
    """``J(theta)`` = the tracked points' sum over the run, weighted by a fixed direction, the sphere at the draw's offset
    and moving at its velocity (about 20 mm/s, a tenth of a pixel a step), from the build's state or from the state 7
    steps on (``continued``: frames made, the camera's ring and the tracker's held point the run's); ``theta`` by
    :data:`PARAMETERS`' names."""
    from mime.nodes.control._controller import leaf_param

    outputs, _, _, params = open_loop(dtype, True)
    rng = np.random.default_rng(5000 + cases.SEED)
    offset = np.array([rng.uniform(-REACH[0], REACH[0]) / 2, rng.uniform(-REACH[1], REACH[1]) / 2, 0.0])
    angle = rng.uniform(0.0, 2 * math.pi)
    velocity = 0.02 * np.array([math.cos(angle), math.sin(angle), 0.0])
    weights = rng.normal(size=3)
    weights[2] = 0.0
    with cases.in_dtype(dtype):
        dtype_ = jnp.zeros(()).dtype
        state = outputs(jnp.zeros(3, dtype=dtype_), {}, None, jnp.asarray(velocity, dtype=dtype_))[0] if continued \
            else None
        believed = leaf_param(cases.FX)
        theta0 = {"x0": jnp.asarray(0.0, dtype_), "y0": jnp.asarray(0.0, dtype_),
                  "camera_fx": jnp.asarray(params["nodes"]["camera:"]["fx"], dtype_),
                  "camera_cy": jnp.asarray(params["nodes"]["camera:"]["cy"], dtype_),
                  "believed_fx": jnp.asarray(params["nodes"]["controller"][believed], dtype_)}
        w = jnp.asarray(weights, dtype=dtype_)
        moving = None if continued else jnp.asarray(velocity, dtype=dtype_)

        def at(theta: dict[str, Any]) -> Any:
            moved = jnp.asarray(offset, dtype=dtype_) + jnp.stack([theta["x0"], theta["y0"], jnp.zeros((), dtype_)])
            leaves = {("camera:", "fx"): theta["camera_fx"], ("camera:", "cy"): theta["camera_cy"],
                      ("controller", believed): theta["believed_fx"]}
            _, out = outputs(moved, leaves, state, moving)
            return jnp.sum(out["controller"]["state_point"][1:] @ w) / 1e-3

    return jax.jit(at), theta0, jax.jit(lambda th: jax.jacfwd(at)(th))


def central(dtype: str, at: Any, theta0: dict[str, Any], name: str) -> dict[str, Any]:
    with cases.in_dtype(dtype):
        base = float(theta0[name])
        scale = PARAMETERS[name]
        h = RELATIVE_H[dtype] * scale

        def value(by: float) -> float:
            return float(at({**theta0, name: jnp.asarray(base + by, dtype=theta0[name].dtype)}))

        centre = value(0.0)
        rows = [((value(h / 2 ** k) - value(-h / 2 ** k)) / (2 * h / 2 ** k),
                 value(h / 2 ** k) - 2 * centre + value(-h / 2 ** k), h / 2 ** k) for k in range(5)]
        table = np.asarray([value(k * NOISE_SPACING[dtype] * scale) for k in range(-3, 4)])
    noise = float(math.sqrt(np.mean(np.diff(table, 4) ** 2) * math.factorial(4) ** 2 / math.factorial(8)))
    (d2, _, _), (d3, s3, _), (d4, s4, h4) = rows[2], rows[3], rows[4]
    a, b = abs(d2 - d3), abs(d3 - d4)
    order = min(2.0, max(0.5, math.log2(a / b))) if a > 0 and b > 0 else 2.0
    rho = 3.0 * noise / h4
    kinked = abs(s4) > 6 * noise and s3 != 0 and abs(s4 / s3) > 0.375
    line = (abs(s4) / (2 * h4) if kinked else 3.0 * b / (2 ** order - 1)) + 3.0 * rho
    return {"cd": d4, "line": line, "order": order, "discriminates": line <= 0.1 * abs(d4)}


@pytest.mark.parametrize("continued", [False, True], ids=["from_reset", "continued"])
def test_the_trackers_derivative_meets_central_differences_and_float32_meets_float64(continued):
    rows: dict[str, Any] = {}
    derivatives: dict[str, dict[str, float]] = {}
    for dtype in ("float64", "float32"):
        at, theta0, forward = objective(dtype, continued)
        with cases.in_dtype(dtype):
            derivatives[dtype] = {k: float(v) for k, v in forward(theta0).items()}
        for name in PARAMETERS:
            rows[f"{dtype} {name}"] = {**central(dtype, at, theta0, name), "forward": derivatives[dtype][name]}
    judged = {k: r for k, r in rows.items() if r["discriminates"]}
    assert any(k.startswith("float64") for k in judged), ("float64's central differences discriminate nowhere", rows)
    missed = {k: r for k, r in judged.items() if not abs(r["forward"] - r["cd"]) <= r["line"]}
    e32 = float(np.finfo("float32").eps)
    for name in PARAMETERS:
        if f"float32 {name}" in judged:
            continue
        f64, f32 = derivatives["float64"][name], derivatives["float32"][name]
        line = (128 * e32 + 1.5 * math.sqrt(EX.WIDTH * e32)) * max(abs(f64), 1e-300)
        if not abs(f32 - f64) <= line:
            missed[f"float32 {name} against float64"] = {"float32": f32, "float64": f64, "line": line}
    assert not missed, missed


# -- the guard -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("where", ["outside", "on_the_border"])
def test_a_ball_out_of_the_frame_or_on_its_border_is_lost_held_and_never_a_nan(where):
    """Outside the frame (15 mm off, past its 12 mm half-width) nothing is seen; on its border (11.5 mm off, the
    silhouette reaching past the edge) the shadow is clipped and its centroid pulled inwards. Each is lost: the point
    held at the last one (the start, before any frame), counted, and finite."""
    offset = np.array([15e-3 if where == "outside" else 11.5e-3, 0.0, 0.0])
    seen = found("float64", False, offset)
    points, lost = seen["controller.state_point"], seen["controller.state_lost"]
    assert np.all(np.isfinite(points)), points
    assert lost[-1] == STEPS, lost
    assert np.array_equal(points[-1], np.asarray(EX.START)), points[-1]


# -- out of the plane ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("height", [-2e-3, -0.25e-3, 0.25e-3, 2e-3])
def test_a_ball_off_the_plane_is_placed_by_the_planes_bias_and_flagged_past_the_tolerance(height):
    """The camera sees no depth: a ball ``height`` above the plane, ``offset`` from the optical axis, is seen along the
    ray through its centre, which meets the plane ``offset * height / (Z - height)`` further out (``Z`` the camera's
    height over the plane, 0.1 m): the bias the plane assumption gives, here 0.12 mm at 6 mm out and 2 mm up. The
    shadow's area says how far the depth has moved (to the conic's bias), and the flag is raised past the tolerance
    (0.5 mm) and not within it."""
    offset = np.array([6e-3, 0.0, height])
    seen = found("float64", False, offset)
    truth, _ = models()
    z = EX.CAMERA_AT[2] - EX.START[2]
    on_plane = np.asarray(EX.START) + np.array([offset[0] * z / (z - height), 0.0, 0.0])
    _, line = centroid_line("float64", seen["camera:.expected"][-1], truth, False, np.asarray(EX.START) + offset)
    point = seen["controller.state_point"][-1]
    assert float(np.linalg.norm(point[:2] - on_plane[:2])) <= line, (point - on_plane, line)
    miss = float(seen["controller.state_depth_miss"][-1])
    assert abs(miss - abs(height)) <= 0.02 * abs(height), (miss, height)
    flagged = bool(seen["controller.state_off_plane"][-1] > 0)
    assert flagged is (abs(height) > 5e-4), (flagged, height)
