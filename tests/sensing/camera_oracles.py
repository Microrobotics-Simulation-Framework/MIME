"""The camera core's oracles as ``discriminating_range.py`` takes them: ``function(size, seed) -> bool``, True when the
gate's own check flags an error of ``size`` planted at the draw of ``seed``; ``size == 0`` is the clean call, which must
not flag. Each runs the gate's check at its own line, in float64, so the range found is the gate's resolution:

- ``projection``: an error in ``k1`` of ``size`` against OpenCV's goldens (a golden case by seed), at G1's line;
- ``undistortion``: a ray turned by ``size`` radians against the float64 Newton oracle, at G2's line;
- ``edge_distance``: the box edge's line moved by ``size`` pixels on the 15 px sphere, against the exact areas at G15's
  per-pixel coverage line;
- ``noise_gain``: a gain error of relative ``size`` in G5's variance check, over the seed's keys;
- ``media``: the water's attenuation off by relative ``size`` in G7's transmission check;
- ``aperture``: the horizontal aperture off by relative ``size`` in G10's hand numbers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, str(Path(__file__).parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent


def projection(size: float, seed: int) -> bool:
    import test_camera_projection as gate

    from mime.sensing import BrownConrady, CameraPose, Pinhole, project

    cases = json.loads((HERE / "projection_goldens.json").read_text(encoding="utf-8"))["cases"]
    case = cases[2 + seed % (len(cases) - 2)]
    k1, k2, p1, p2, k3 = case["distortion"]
    camera = Pinhole(case["width"], case["height"], case["fx"], case["fy"], case["cx"], case["cy"],
                     BrownConrady(k1 + size, k2, p1, p2, k3))
    pixels, _ = project(camera, CameraPose(jnp.asarray(case["position"]), jnp.asarray(case["orientation"])),
                        jnp.asarray(case["points"]))
    expected = np.asarray(case["pixels"])
    error = np.max(np.abs(np.asarray(pixels) - expected) / (case["fx"] + np.abs(expected)))
    return bool(error > gate.ULPS * float(np.finfo(np.float64).eps))


def undistortion(size: float, seed: int) -> bool:
    import test_undistortion as gate

    rng = np.random.default_rng(seed)
    lens = (rng.uniform(-0.4, 0.3), rng.uniform(-0.2, 0.2), rng.uniform(-3e-3, 3e-3), rng.uniform(-3e-3, 3e-3),
            rng.uniform(-0.1, 0.1))
    worst, tolerance, _, _ = gate.rays_against_oracle(lens, rng.uniform(0.8, 1.6) * gate.WIDTH)
    return bool(worst + size > tolerance)


def edge_distance(size: float, seed: int) -> bool:
    import mime.sensing._coverage as coverage
    import test_box_edge as gate

    held = coverage.line

    def moved(g, gu, gv):
        distance, nu, nv = held(g, gu, gv)
        return distance + size, nu, nv

    coverage.line = moved
    try:
        areas, _ = gate.truth(15)
        theta = gate.theta_for(15) + np.array([0.0, 0.0, 0.0, 0.0])
        theta[0] += (seed % 97) * 1e-3 * gate.DISTANCE / theta[3]
        areas = gate.sphere_oracle.areas(theta, gate.RADIUS, gate.DISTANCE, (gate.WIDTH - 1) / 2,
                                          (gate.HEIGHT - 1) / 2, (gate.HEIGHT, gate.WIDTH))
        covered = np.asarray(gate.image(jnp.asarray(theta), "box_edge"))
    finally:
        coverage.line = held
    band = (areas > 0) & (areas < 1)
    return bool(np.linalg.norm((covered - areas)[band]) / np.linalg.norm(areas[band]) > gate.BOX[15][1])


def noise_gain(size: float, seed: int) -> bool:
    from mime.sensing import Response, respond

    response = Response(full_scale=2000.0, gain=0.8, offset=12.0, read_noise=6.0, bits=12)
    level = 0.2
    respond_at = jax.jit(lambda key: respond(jnp.full((16, 16), level), response, key)[1])
    frames = np.stack([np.asarray(respond_at(jax.random.key(seed * 400 + k))) for k in range(400)])
    gain = 0.8 * (1.0 + size)
    variance = gain ** 2 * (2000.0 * level + 36.0) + 1.0 / 12.0
    n = frames.size
    return bool(abs(frames.var(ddof=1) - variance) > 4.0 * variance * np.sqrt(2.0 / (n - 1)))


def media(size: float, seed: int) -> bool:
    import test_media as gate

    from mime.sensing import render

    width = 96
    camera, pose = gate.across(width)
    centre = (gate.CENTRE[0] + (seed % 13) * 1e-5, 0.0, gate.CENTRE[2])
    subjects, poses = gate.scene(gate.Appearance.attenuating(gate.SPHERE), centre)
    found = np.asarray(render(subjects, camera, pose, poses).transmission)
    origin, d, focal = gate.oracle_rays(width)
    outer, oe = gate.chord_cylinder(origin, d, gate.TUBE["inner_radius"] + gate.TUBE["wall"])
    lumen, le = gate.chord_cylinder(origin, d, gate.TUBE["inner_radius"])
    ball, be = gate.chord_sphere(origin, d, np.asarray(centre), gate.RADIUS)
    water = gate.WATER * (1.0 + size)
    expected = np.exp(-gate.WALL * (outer - lumen) - water * (lumen - ball) - gate.SPHERE * ball)
    clear = np.minimum.reduce([np.abs(oe), np.abs(le), np.abs(be)]) > 2.0 * 0.1 / focal
    return bool(np.max(np.abs(found[clear] / expected[clear] - 1.0)) > 1e-10)


def aperture(size: float, seed: int) -> bool:
    from mime.scene import write_camera
    from mime.scene._pxr import Usd, UsdGeom
    from mime.sensing import BrownConrady, CameraPose, Pinhole, to_camera_prim

    rng = np.random.default_rng(seed)
    width = int(rng.integers(100, 1000))
    camera = Pinhole(width, 480, 800.0, 820.0, 315.5, 243.25, BrownConrady())
    stage = Usd.Stage.CreateInMemory()
    held = to_camera_prim(camera, CameraPose(jnp.zeros(3), jnp.asarray((1.0, 0.0, 0.0, 0.0))), pitch=5e-6)
    prim = write_camera(stage, "/camera", held, meters_per_unit=1e-3)
    written = UsdGeom.Camera(prim).GetHorizontalApertureAttr().Get() * (1.0 + size)
    return bool(abs(written / (width * 5e-6 * 1e4) - 1.0) > 1e-6)
