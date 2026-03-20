#!/usr/bin/env python3
"""Demo: helical microrobot navigating inside a parametric cylindrical tube.

Sets up the full magnetic actuation chain:
  ExternalMagneticFieldNode -> MagneticResponseNode -> RigidBodyNode

Renders with PyVistaViewport and saves frames as a GIF.

Usage:
    python examples/visualise_helical_robot.py

Requires: pip install mime-microrobotics[viz]
    (which pulls in pyvista and usd-core)
"""

from __future__ import annotations

import sys
import numpy as np

# Check optional deps before doing anything
try:
    from pxr import Usd
except ImportError:
    print("This demo requires usd-core: pip install usd-core")
    sys.exit(1)

try:
    import pyvista
except ImportError:
    print("This demo requires pyvista: pip install pyvista")
    sys.exit(1)

import jax.numpy as jnp

from mime.nodes.actuation.external_magnetic_field import ExternalMagneticFieldNode
from mime.nodes.robot.magnetic_response import MagneticResponseNode
from mime.nodes.robot.rigid_body import RigidBodyNode
from mime.core.geometry import CylinderGeometry
from mime.viz.stage_bridge import StageBridge
from mime.viz.pyvista_viewport import PyVistaViewport


def main():
    # -- Simulation parameters -----------------------------------------------

    dt = 0.0001           # 0.1 ms timestep
    n_steps = 2000        # 0.2 seconds of simulation
    render_every = 50     # render every 50 steps (5ms intervals = 200 FPS capture)
    freq_hz = 10.0        # 10 Hz field rotation
    field_mt = 10.0       # 10 mT field strength

    # Robot: prolate ellipsoid (a=150um, b=50um) — helical body approximation
    a = 150e-6
    b = 50e-6

    # Tube: D=2mm, L=10mm — parametric cylinder
    tube = CylinderGeometry(diameter_m=2e-3, length_m=10e-3, axis="z")

    # -- Create nodes --------------------------------------------------------

    field_node = ExternalMagneticFieldNode("field", dt)
    mag_node = MagneticResponseNode(
        "mag", dt,
        volume_m3=4.0 / 3.0 * 3.14159 * a * b * b,  # ellipsoid volume
        n_axi=0.1,
        n_rad=0.45,
    )
    body_node = RigidBodyNode(
        "body", dt,
        semi_major_axis_m=a,
        semi_minor_axis_m=b,
        fluid_viscosity_pa_s=8.5e-4,
    )

    # -- Set up USD stage + viewports ----------------------------------------

    bridge = StageBridge()

    # Add the tube as static geometry
    bridge.add_parametric_geometry(tube, prim_path="/World/Tube")

    # Register robot for dynamic updates
    bridge.register_robot("body", prim_path="/World/Robot", radius=float(a))

    # Register field arrow
    bridge.register_field("field", prim_path="/World/FieldArrow", arrow_length=1e-3)

    # Create viewport
    viewport = PyVistaViewport(width=640, height=480, background="white")

    # -- Run simulation and capture frames -----------------------------------

    print(f"Running {n_steps} steps at dt={dt}s (freq={freq_hz}Hz, B={field_mt}mT)")
    print(f"Robot: prolate ellipsoid a={a*1e6:.0f}um, b={b*1e6:.0f}um")
    print(f"Tube: D={tube.diameter_m*1e3:.1f}mm, L={tube.length_m*1e3:.1f}mm")

    fs = field_node.initial_state()
    ms = mag_node.initial_state()
    bs = body_node.initial_state()

    field_bi = {"frequency_hz": freq_hz, "field_strength_mt": field_mt}

    frames = []

    for step in range(n_steps):
        # Step the chain
        fs = field_node.update(fs, field_bi, dt)
        B = fs["field_vector"]
        grad_B = fs["field_gradient"]

        mag_bi = {
            "field_vector": B,
            "field_gradient": grad_B,
            "orientation": bs["orientation"],
        }
        ms = mag_node.update(ms, mag_bi, dt)

        body_bi = {
            "magnetic_force": ms["magnetic_force"],
            "magnetic_torque": ms["magnetic_torque"],
        }
        bs = body_node.update(bs, body_bi, dt)

        # Update USD stage
        state = {
            "body": {
                "position": np.asarray(bs["position"]),
                "orientation": np.asarray(bs["orientation"]),
            },
            "field": {
                "field_vector": np.asarray(fs["field_vector"]),
            },
        }
        bridge.update(state)

        # Capture frame
        if step % render_every == 0:
            img = viewport.render(bridge.stage)
            frames.append(img)
            if step % (render_every * 10) == 0:
                t = step * dt
                pos = np.asarray(bs["position"])
                print(f"  t={t:.4f}s  pos=[{pos[0]:.2e}, {pos[1]:.2e}, {pos[2]:.2e}]")

    print(f"Captured {len(frames)} frames")

    # -- Save as GIF ---------------------------------------------------------

    try:
        import imageio.v3 as iio
        output_path = "helical_robot_demo.gif"
        iio.imwrite(output_path, frames, duration=50, loop=0)
        print(f"Saved animation to {output_path}")
    except ImportError:
        print("imageio not installed — skipping GIF export")
        print("Install with: pip install imageio")

    # -- Export USD file ------------------------------------------------------

    bridge.export("helical_robot_demo.usd")
    print("Exported USD stage to helical_robot_demo.usd")

    viewport.close()
    print("Done.")


if __name__ == "__main__":
    main()
