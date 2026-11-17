"""The creased cylinder's images on the sampled route, as arrays in an NPZ, for ``.github/probe/bit_compare.py`` (the
command run from each tree's root): ``python scripts/camera_route2/creased_images.py PROGRAM OUT.npz``, PROGRAM
``creased32`` or ``creased64``. Each view's transmission and flags, and the forward render's structural count (XLA's
cost analysis of the compiled frame at the first view: flops plus 20 times the transcendentals, as
``_sampled.frame_cost`` weighs them). A change of derivatives only leaves every array equal; BUMP=1 moves the camera one
ulp along y (the control: the images differ)."""

from __future__ import annotations

import math
import os
import sys

import numpy as np

program, out = sys.argv[1], sys.argv[2]
dtype = "float64" if program.endswith("64") else "float32"
import jax  # noqa: E402

jax.config.update("jax_enable_x64", dtype == "float64")
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, "scripts/camera_route2")
import crease_line as C  # noqa: E402

from mime.sensing import Appearance, CameraPose, Pinhole, Subject, render  # noqa: E402

_one = np.float64 if dtype == "float64" else np.float32
Y = float(np.nextafter(_one(-C.DISTANCE), _one(0.0))) if os.environ.get("BUMP") == "1" else -C.DISTANCE
#: The views (turn, roll, offset in pixels): the three singular poses, the first forms' four and two others.
VIEWS = ((6.0, 75.0, (0.3, 0.8)), (3.0, 0.0, (0.0, 0.0)), (2.0, 0.0, (0.0, 0.0)), (0.0, 20.0, (0.0, 0.0)),
         (4.0, 20.0, (0.0, 0.0)), (15.0, 20.0, (0.0, 0.0)), (11.3, 47.0, (0.21, 0.63)), (27.0, 88.0, (0.9, 0.1)))

camera = Pinhole(C.WIDTH, C.HEIGHT, C.FOCAL, C.FOCAL, (C.WIDTH - 1) / 2, (C.HEIGHT - 1) / 2)
pose = CameraPose(jnp.asarray([0.0, Y, 0.0]), jnp.asarray([math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0]))
found: dict[str, np.ndarray] = {}
for number, (turn, roll, (du, dv)) in enumerate(VIEWS):
    axis = C.axis_of(turn, roll)
    centre = C.CENTRE + np.array([du, 0.0, -dv]) * C.PIXEL
    body = Subject("rod", "cylinder", {"radius": C.RADIUS, "length": C.LENGTH, "axis": tuple(float(a) for a in axis)},
                   appearance=Appearance.opaque(), moving=True, route="sampled")

    def frame(at, body=body):
        return render([body], camera, pose, {"rod": (at, jnp.asarray((1.0, 0.0, 0.0, 0.0)))}).transmission

    at = jnp.asarray(centre)
    found[f"view{number}"] = np.asarray(frame(at))
    flags = render([body], camera, pose, {"rod": (at, jnp.asarray((1.0, 0.0, 0.0, 0.0)))}).flags
    found[f"view{number}_flags"] = np.asarray([bool(flags[k]) for k in sorted(flags)])
    if number == 0:
        cost = jax.jit(frame).lower(at).compile().cost_analysis() or {}
        cost = cost[0] if isinstance(cost, list) else cost
        found["forward_count"] = np.asarray(float(cost.get("flops", 0.0)) + 20.0 * float(cost.get("transcendentals", 0.0)))
np.savez(out, **found)
print(program, "views", len(VIEWS), "forward count", float(found["forward_count"]))
