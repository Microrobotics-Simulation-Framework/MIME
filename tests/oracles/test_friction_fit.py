"""Identifying mu (decided Fr5, ``mime.study.fit_friction``): ``Study.fit`` on sliding data, the gradient-free search
over [0.01, 1.2] on the same loss as its control, and the identifiability verdict, through ``built.run`` on a tilted
tube's floor (float64):

- **Synthetic recovery**: a block (a cylinder lying down the slope on the floor) sliding at ``tan(theta) = 2 mu``, its
  speed after 0.1 s the data, made at ``mu = 0.3``; fitted from 0.2 it ends within one standard error of 0.3, the
  control agrees, and the verdict is ``identified``.
- **The control, pure rolling data**: a sphere rolling without slipping at ``tan(theta) = 0.7 mu``: its speed does not
  depend on ``mu`` above its threshold, and the verdict is ``bounded below only``, with the bound the data hold
  (``(2/7) tan(theta)``, the traction rolling needs over its normal force).
- **Data that never slide** (the review's blocker): a block under its threshold, fitted from above it and from below,
  reads ``bounded below only`` with the slope's ``tan(theta)`` its bound, never ``identified``; sliding data fitted from
  a start where the model sticks start again from the control and read ``identified``.
- **The record**: a fitted value ships as a named record pinned by its content's and its data's hashes, never a default
  (``Coulomb`` has none for ``mu``)."""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from friction_cases import BALL, ROD, axis, on_the_floor

from mime.interactions.friction import Coulomb
from mime.study import fit_friction
from mime.units import s

AXIS = "contact:ball.friction.mu"
DURATION = 0.05
TRUE = 0.3
#: A margin of 10 um and the hull's points 80 um apart: within the margin for the sphere and the cylinder, and a few
#: thousand points (the default spacing at the default margin seeds tens of thousands).
COARSE = {"margin": 1e-5, "spacing": 8e-5}


@pytest.fixture(autouse=True, scope="module")
def float64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


def speed_down(theta: float):
    """The body's speed down the tube's lowest line from its final twist."""
    direction = -jnp.asarray(axis(theta))

    def read(final):
        return jnp.sum(final["ball"]["twist"][:3] * direction)
    return read


def data(experiment, theta: float) -> float:
    return float(speed_down(theta)(experiment.build().run(s(DURATION)).final))


@pytest.fixture(scope="module")
def sliding():
    theta = math.atan(2.0 * TRUE)
    observed = data(on_the_floor(theta=theta, mu=TRUE, body=ROD, along=True, **COARSE), theta)

    def residual(final):
        return jnp.atleast_1d(speed_down(theta)(final) - observed)

    found = fit_friction(on_the_floor(theta=theta, mu=0.2, body=ROD, along=True, **COARSE), residual, axis=AXIS, start=0.2,
                         noise=1e-3 * abs(observed), duration=s(DURATION), body="ball")
    return found, observed


def test_sliding_data_identify_mu_and_the_gradient_free_control_agrees(sliding):
    found, observed = sliding
    print(f"\n{found.verdict}: {found.estimate:.9g} (control {found.control:.9g}, one standard error "
          f"{found.uncertainty}); {found.why}")
    assert found.verdict == "identified"
    assert found.uncertainty is not None and abs(found.estimate - TRUE) <= found.uncertainty
    assert abs(found.estimate - found.control) <= 1e-4 * TRUE + found.uncertainty


def test_control_pure_rolling_data_are_bounded_below_only():
    theta = math.atan(3.5 * TRUE * 0.2)
    observed = data(on_the_floor(theta=theta, mu=TRUE, body=BALL, **COARSE), theta)

    def residual(final):
        return jnp.atleast_1d(speed_down(theta)(final) - observed)

    found = fit_friction(on_the_floor(theta=theta, mu=0.5, body=BALL, **COARSE), residual, axis=AXIS, start=0.5,
                         noise=1e-3 * abs(observed), duration=s(DURATION), body="ball")
    print(f"\n{found.verdict}: lower bound {found.lower_bound:.9g} against (2/7) tan = {2 / 7 * math.tan(theta):.9g}")
    assert found.verdict == "bounded below only"
    assert abs(found.lower_bound - 2.0 / 7.0 * math.tan(theta)) <= 1e-6


#: Blocks on a slope under their threshold, ``tan(theta) = ratio mu_true``, which never slide (the review's all-stick
#: draws): ``(mu_true, ratio, start / mu_true)``, one started above the threshold (where the loss is flat) and one
#: below it (the path climbs to the threshold, where the loss is flat above).
STUCK = [(0.194, 0.773, 2.18), (0.5, 0.9, 0.6)]
#: The search for data drawn past the register's range: starts up to 2.5 of a coefficient of 0.8.
WIDE = (0.01, 2.5)


@pytest.mark.parametrize("mu_true, ratio, share", STUCK, ids=["started-above", "started-below"])
def test_data_that_never_slide_bound_mu_below_only_never_identified(mu_true, ratio, share):
    """The review's blocker: a block that never slides holds ``mu`` from below only, whatever the start; the verdict
    says so (one standard error read above the estimate is flat: a larger coefficient moves no datum), and its bound
    is the slope's ``tan(theta)``, the traction over the normal force, at most the true coefficient."""
    theta = math.atan(ratio * mu_true)
    observed = data(on_the_floor(theta=theta, mu=mu_true, body=ROD, along=True, **COARSE), theta)

    def residual(final):
        return jnp.atleast_1d(speed_down(theta)(final) - observed)

    start = share * mu_true
    found = fit_friction(on_the_floor(theta=theta, mu=start, body=ROD, along=True, **COARSE), residual, axis=AXIS,
                         start=start, noise=1e-3 * 9.81 * math.sin(theta) * DURATION, duration=s(DURATION),
                         body="ball", search=WIDE)
    print(f"\n{found.verdict}: {found.why}")
    assert found.verdict == "bounded below only", found.why
    assert found.uncertainty is None and found.lower_bound is not None
    assert abs(found.lower_bound - math.tan(theta)) <= 1e-6 * math.tan(theta) and found.lower_bound <= mu_true


def test_sliding_data_started_where_the_model_sticks_start_again_from_the_control():
    """Sliding data fitted from a start past the threshold (the model sticks there, its derivative zero; the review's
    minor 2): the path starts again from the gradient-free control and identifies mu."""
    mu_true, ratio = 0.769, 1.6
    theta = math.atan(ratio * mu_true)
    observed = data(on_the_floor(theta=theta, mu=mu_true, body=ROD, along=True, **COARSE), theta)

    def residual(final):
        return jnp.atleast_1d(speed_down(theta)(final) - observed)

    found = fit_friction(on_the_floor(theta=theta, mu=1.84, body=ROD, along=True, **COARSE), residual, axis=AXIS,
                         start=1.84, noise=1e-3 * abs(observed), duration=s(DURATION), body="ball", search=WIDE)
    print(f"\n{found.verdict} (restarted {found.restarted}): {found.estimate:.9g}; {found.why}")
    assert found.restarted and found.verdict == "identified", found.why
    assert found.uncertainty is not None and abs(found.estimate - mu_true) <= found.uncertainty


def test_a_fitted_value_ships_as_a_hash_pinned_record_never_a_default(sliding):
    found, observed = sliding
    record = found.record("the_block", np.asarray([observed]).tobytes())
    again = found.record("the_block", np.asarray([observed]).tobytes())
    assert record == again and len(record["sha256"]) == 64 and record["label"].startswith("estimated from")
    assert record != found.record("the_block", np.asarray([observed * (1 + 1e-12)]).tobytes())
    assert dataclasses.fields(Coulomb)[0].name == "mu" and dataclasses.fields(Coulomb)[0].default is dataclasses.MISSING


def test_an_axis_that_is_no_friction_coefficients_is_refused_before_any_run():
    from mime.build._refusals import ArgumentRefused

    with pytest.raises(ArgumentRefused) as refused:
        fit_friction(on_the_floor(theta=0.1, mu=TRUE, body=BALL, **COARSE), lambda final: final, axis="ball.mass",
                     start=0.2, noise=1e-3, duration=s(DURATION), body="ball")
    assert refused.value.codes() == ["fit.start"] and "contact:ball.friction.mu" in str(refused.value)
