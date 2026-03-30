"""Hook: scalar_extractor — UMR-specific derived scalars.

Extracts swimming speed, angular velocity, drag torque, field
frequency, and synchrony ratio from graph state.
"""

from __future__ import annotations

import math

import numpy as np


class ScalarExtractor:
    """Stateful scalar extractor with lazy initialization.

    Instantiated without params (in hooks.py), then lazy-inits
    on the first __call__ using ctx.params. This avoids the need
    for the hook loader to handle class instantiation with params.
    """

    def __init__(self):
        self._initialized = False
        self._prev_pos = 0.0
        self._swimming_axis = 2  # default Z

    def __call__(self, ctx) -> dict[str, float]:
        if not self._initialized:
            self._swimming_axis = ctx.params.get("SWIMMING_AXIS", 2)
            self._initialized = True

        ax = self._swimming_axis
        scalars = {}

        # Rigid body scalars (swimming speed, angular velocity)
        rb = ctx.state.get("rigid_body")
        if rb is not None:
            pos = float(np.asarray(rb["position"])[ax])
            scalars["swimming_speed_m_s"] = (pos - self._prev_pos) / ctx.dt
            self._prev_pos = pos
            scalars["omega_body_rad_s"] = float(
                np.asarray(rb["angular_velocity"])[ax],
            )

        # Drag torque from any node that provides it
        for name, ns in ctx.state.items():
            if name.startswith("_"):
                continue
            if "drag_torque" in ns:
                scalars["drag_torque"] = float(
                    np.asarray(ns["drag_torque"])[ax],
                )

        # Field frequency and synchrony ratio
        if ctx.ext_inputs:
            f_hz = float(
                ctx.ext_inputs.get("ext_field", {}).get("frequency_hz", 0),
            )
            scalars["field_frequency_hz"] = f_hz
            if f_hz > 0 and "omega_body_rad_s" in scalars:
                omega_field = 2.0 * math.pi * f_hz
                scalars["synchrony_ratio"] = (
                    abs(scalars["omega_body_rad_s"]) / omega_field
                )

        return scalars
