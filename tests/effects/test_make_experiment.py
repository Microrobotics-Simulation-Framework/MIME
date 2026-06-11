"""make_experiment(params) v1.x contract pilot (E5).

Pins the ADR §7 experiment-factory contract on the Stokes-drag demo: a
PARAMS_SCHEMA-validated, defaulted parameter surface; a make_experiment that
returns an unbuilt Experiment which builds + compiles; backend selection by
parameter; and (slow) the Stokeslet backend reproducing analytical Stokes drag
through the factory path.
"""

from __future__ import annotations

import math
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from mime.experiments.stokes_drag_demo import (
    PARAMS_SCHEMA,
    make_experiment,
    resolve_params,
)


# ── parameter resolution / validation ───────────────────────────────────────

def test_defaults_applied():
    p = resolve_params()
    assert p["backend"] == "stokeslet"
    assert p["radius_m"] == PARAMS_SCHEMA["radius_m"].default


def test_unknown_param_raises():
    with pytest.raises(ValueError, match="unknown parameter"):
        resolve_params({"bogus": 1})


def test_bad_choice_raises():
    with pytest.raises(ValueError, match="not in"):
        resolve_params({"backend": "lbm"})


def test_wrong_type_raises():
    with pytest.raises(ValueError, match="must be float"):
        resolve_params({"radius_m": "big"})


def test_int_coerced_to_float():
    p = resolve_params({"fluid_density_kg_m3": 1000})
    assert isinstance(p["fluid_density_kg_m3"], float)


# ── factory build / compile ──────────────────────────────────────────────────

def test_make_experiment_stokeslet_builds_and_compiles():
    exp = make_experiment({"backend": "stokeslet"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm, handles = exp.build()
    assert "body" in gm.node_names and "fluid" in gm.node_names


def test_make_experiment_default_is_stokeslet():
    exp = make_experiment()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm, _ = exp.build()
    assert "fluid" in gm.node_names


def test_make_experiment_returns_unbuilt_experiment():
    # The factory returns a configured-but-unbuilt Experiment (the caller
    # chooses when to build) — it has the body/medium/effect attached.
    exp = make_experiment()
    assert exp.body is not None and exp.medium is not None
    assert len(exp._effects) == 1


@pytest.mark.slow
def test_make_experiment_fvm_builds_and_compiles():
    exp = make_experiment({"backend": "fvm"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm, _ = exp.build()
    assert "fluid" in gm.node_names


@pytest.mark.slow
def test_stokeslet_factory_reproduces_stokes_drag():
    # Same validated physics as the concept-proof, now reached through the
    # make_experiment factory.
    a, mu, rho_f, dt, V = 1e-3, 1e-3, 1000.0, 1e-3, 1e-4
    exp = make_experiment({
        "backend": "stokeslet", "radius_m": a, "viscosity_pa_s": mu,
        "fluid_density_kg_m3": rho_f, "timestep_s": dt,
    })
    ext = {"body": {"external_velocity": jnp.asarray([V, 0.0, 0.0], jnp.float32),
                    "external_angular_velocity": jnp.zeros(3, jnp.float32)}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm, _ = exp.build()
        for _ in range(2):
            gm.step(ext)
        drag = np.asarray(gm.get_node_state("fluid")["drag_force"])

    F_stokes = 6.0 * math.pi * mu * a * V
    rel = abs(abs(drag[0]) - F_stokes) / F_stokes
    assert rel < 0.05, f"|drag| {abs(drag[0]):.4e} vs 6πμaV {F_stokes:.4e} (rel {rel:.1%})"
