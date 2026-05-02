"""Live actuation control for the de Jongh confined-swimming experiment.

Each tick the runner calls ``get_external_inputs(params, step)``. We
read ``FIELD_FREQUENCY_HZ`` and ``FIELD_STRENGTH_MT`` from ``params``
— which the runner updates in place when MICROROBOTICA's
ParameterPanel sends a ``{"command": "params", ...}`` ZMQ message.
That gives us free hot-reload of the two live-editable knobs.
"""

from __future__ import annotations

import jax.numpy as jnp


def get_external_inputs(params: dict, step_count: int) -> dict:
    return {
        "field": {
            "frequency_hz": jnp.float32(params["FIELD_FREQUENCY_HZ"]),
            "field_strength_mt": jnp.float32(params["FIELD_STRENGTH_MT"]),
        }
    }
