"""Live actuation control for the AR4 + helical-UMR drive experiment.

Each tick the runner calls ``get_external_inputs(params, step)``. We
emit:

* ``motor.commanded_velocity`` — the angular velocity the rotor PI
  loop should track. Live-editable via ``FIELD_FREQUENCY_HZ`` in the
  params namespace; MICROROBOTICA's ParameterPanel hot-reloads it.
* ``arm.commanded_joint_torques`` — currently zero (arm holds its
  initial pose under joint friction; gravity compensation can be
  added here when the experiment grows a manipulation phase).

v1 sensing cheat (MIME-ANO-104): the controller does not yet read
microrobot position from a sensor node; the rotor is run open-loop at
the commanded frequency. A future SENSING plan introduces sensor-role
MimeNodes that publish noisy/incomplete observations.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def get_external_inputs(params: dict, step_count: int) -> dict:
    omega_rad_s = float(2.0 * np.pi * params["FIELD_FREQUENCY_HZ"])
    # Arm gets zero commanded torque — joint friction holds the home
    # pose. Edit here to add gravity compensation, IK trajectories,
    # or live ParameterPanel-driven joint targets.
    n_dof = len(params["ARM_HOME_RAD"])
    return {
        "motor": {
            "commanded_velocity": jnp.float32(omega_rad_s),
        },
        "arm": {
            "commanded_joint_torques": jnp.zeros(n_dof, dtype=jnp.float32),
        },
    }
