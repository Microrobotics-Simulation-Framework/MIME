"""The build's boundary for values handed to a node (``mime.core._precision.declared_values``, in ``MimeNode``'s
constructor and a controller's start): a float32 run narrows a float64 JAX array to float32, as JAX made it before
precision was declared; a float64 run refuses a float32 JAX array (``run.precision-lossy``: computed before a build
enabled 64-bit types, it would enter rounded), or widens it with a warning where the one switch says so, and takes
Python floats and NumPy float64 values as they are.

And D1's default where the suite's test-only bridge does not reach: a process started without it (or one outside a
pytest session, where shipped code never reads it) plans an experiment that states no precision in float64."""

from __future__ import annotations

import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from mime.build import BuildRefused
from mime.core import _precision
from mime.core._precision import FROM_JAX_FLAG, applied
from mime.nodes.body._gravity import GravityNode

MASS = 2.5e-6


def gravity(mass: object) -> GravityNode:
    return GravityNode("g", 1e-3, mass=mass, buoyant_volume=1e-9, fluid_density=1000.0,  # type: ignore[arg-type]
                       gravity=(0.0, 0.0, -9.81))


@pytest.mark.parametrize("given", [MASS, np.float64(MASS)], ids=["python-float", "numpy-float64"])
def test_a_float64_run_takes_python_and_numpy_floats_as_they_are(given):
    with applied("float64"):
        held = gravity(given).params["mass"]
    assert held is given


def test_a_float64_run_refuses_a_float32_jax_array():
    """The policy's default, ``refuse`` (``mime.core._precision.NARROW_IN_FLOAT64``: the maintainer's answer decides)."""
    rounded = jnp.asarray(MASS, dtype=jnp.float32)  # as jax.numpy makes it before a build enables 64-bit types
    with applied("float64"), pytest.raises(BuildRefused) as refused:
        gravity(rounded)
    assert refused.value.codes() == ["run.precision-lossy"]
    (held,) = refused.value.refusals
    assert "['mass'] (float32)" in held.message and "Python float or a NumPy float64 array" in held.action


def test_the_widening_policy_widens_a_float32_jax_array_with_a_warning_naming_it(monkeypatch):
    """The other policy, one line away: widened to float64, with a warning naming the value and the precision."""
    monkeypatch.setattr(_precision, "NARROW_IN_FLOAT64", "widen")
    rounded = jnp.asarray(MASS, dtype=jnp.float32)
    with applied("float64"), pytest.warns(_precision.LossyValueWarning, match=r"\['mass'\] \(float32\).*float64"):
        held = gravity(rounded).params["mass"]
    assert held.dtype == np.float64 and float(held) == float(np.float32(MASS))


def test_a_float32_run_narrows_a_float64_jax_array():
    with applied("float32"):
        held = gravity(jnp.asarray(MASS, dtype=jnp.float64)).params["mass"]
    assert held.dtype == np.float32 and float(held) == float(np.float32(MASS))


_PLANNED = ("import jax; from mime.experiment import Experiment; from mime.scene import Scene; from mime.units import ms; "
            "from mime.core import run_dtype; Experiment('p', scene=Scene.new(up=(0, 0, 1), units='m'), dt=ms(1)).plan(); "
            "print(jax.config.jax_enable_x64, run_dtype())")


@pytest.mark.parametrize("pytest_session", [False, True], ids=["bridge-set-outside-pytest", "no-bridge"])
def test_a_process_the_bridge_does_not_reach_plans_float64(pytest_session):
    """Outside pytest the bridge's variable is not read even where it is set; inside a session without it, D1 holds."""
    env = {**os.environ, "JAX_ENABLE_X64": "0"}
    if pytest_session:
        env.pop(FROM_JAX_FLAG, None)
    else:
        env.pop("PYTEST_VERSION", None)
        env[FROM_JAX_FLAG] = "1"
    done = subprocess.run([sys.executable, "-c", _PLANNED], capture_output=True, text=True, env=env, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.split()[-2:] == ["True", "float64"]


def test_control_the_bridge_inside_a_pytest_session_builds_as_jax_s_flag_says():
    """The control: the same process with the bridge, as the suite runs, plans in JAX's flag's float32."""
    env = {**os.environ, "JAX_ENABLE_X64": "0", FROM_JAX_FLAG: "1"}
    done = subprocess.run([sys.executable, "-c", _PLANNED], capture_output=True, text=True, env=env, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.split()[-2:] == ["False", "float32"]
