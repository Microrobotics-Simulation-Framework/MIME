"""The solve-site registry (design row #357), on the six steps the plants are applied to: every linear solve a step reaches
went through a shared solve function, nothing is nested in another solve's scope or outside an exempt site's size, and in a
float64 run none narrows.

The walk reads the traced program (``solve_walk``), not the source: a solve is seen wherever it is called from. Its
controls are planted, each a way a solve could skip the seam, and the clean tests below fail on every one of them when
``MIME_PLANT_SOLVE`` names it (``.github/probe/detection_rate.py`` measures that rate). A control that is not caught
would make "the walk is clean" no evidence, so each plant is also asserted caught, in process, here, and caught for the
reason it is planted (its finding holds the text named for it). Every registered node kind's step is walked in
``tests/saved/test_solve_walk.py``."""

from __future__ import annotations

import jax
import pytest
import solve_walk
from solve_walk import PLANTS, cases, downcasts, narrow_anywhere, planted_from_environment, solve_problems

# Every module that registers a site is imported, as the build imports them.
import mime.core.multibody  # noqa: F401
import mime.nodes.flow.fvm  # noqa: F401
import mime.nodes.hydro  # noqa: F401
import mime.nodes.hydro.mrs  # noqa: F401
import mime.nodes.hydro.near_field  # noqa: F401
from mime.core import solves
from mime.core.solves import SOLVE_SITES

CASES = cases()
#: The clean programs, by case and dtype: a clean trace is the same every time, so it is made once.
CLEAN: dict[tuple[str, bool], object] = {}


@pytest.fixture
def traced(monkeypatch: pytest.MonkeyPatch):
    """A case's step traced afresh where a plant is applied (JAX's caches dropped before and after, so that a patch is
    traced, never a cached program), under the plant ``MIME_PLANT_SOLVE`` names where it names one."""
    def trace(label: str, plant: str | None = None):
        named = plant or planted_from_environment()
        key = (label, jax.config.jax_enable_x64)
        if named is None and key in CLEAN:
            return CLEAN[key]
        jax.clear_caches()
        if named is not None:
            PLANTS[named][0](monkeypatch)
        program = CASES[label][1]().jaxpr
        if named is None:
            CLEAN[key] = program
        return program

    yield trace
    monkeypatch.undo()
    jax.clear_caches()


@pytest.mark.parametrize("label", sorted(CASES))
def test_every_solve_a_step_reaches_went_through_a_registered_site_and_stays_in_what_it_owns(dtype, traced, label):
    assert solve_problems(traced(label)) == []


@pytest.mark.parametrize("label", sorted(CASES))
def test_a_float64_runs_step_holds_no_narrower_float(float64, traced, label):
    """The whole step, inside a solve's scope or outside it: the review measured none in all six."""
    assert downcasts(traced(label)) == [] and narrow_anywhere(traced(label)) == []


CHECKS = {"problems": solve_problems, "downcasts": downcasts, "narrow": narrow_anywhere}


@pytest.mark.parametrize("plant", sorted(PLANTS))
def test_control_each_plant_is_caught_on_the_cases_it_touches(float64, traced, plant):
    """A planted bypass, a direct solve of any form, a nested or oversized one, a downcast and a round trip in the caller
    each fail the check named for them, on every case whose program they touch, and for the reason they are planted."""
    _, named, touched, expected = PLANTS[plant]
    found = {label: CHECKS[named](traced(label, plant)) for label in touched}
    assert [label for label, text in found.items() if not any(expected in line for line in text)] == [], found


@pytest.mark.parametrize("plant", sorted(name for name, (_, check, _, _) in PLANTS.items() if check == "problems"))
def test_control_a_planted_solve_is_caught_in_float32_too(dtype, traced, plant):
    """The walk reads a float32 run's program as well: the unregistered check does not depend on the dtype."""
    _, _, touched, expected = PLANTS[plant]
    assert [label for label in touched
            if not any(expected in line for line in solve_problems(traced(label, plant)))] == []


def test_control_a_scope_nobody_registered_is_found():
    def step(x):
        with jax.named_scope(f"{solves.SCOPE}.nobody_registered"):
            return jax.numpy.fft.fft(x)

    found = solve_problems(jax.make_jaxpr(step)(jax.numpy.ones(4)).jaxpr)
    assert any("no module registered" in line for line in found)


# -- the derivative programs --------------------------------------------------------------------------------------------

def _derivatives(dense: bool):
    import jax.numpy as jnp

    def solve(m, b):
        return jnp.sum(solves.dense_solve(m, b) if dense else jnp.linalg.solve(m, b))

    m = jnp.eye(3, dtype=jnp.float64) + 0.1 * jnp.ones((3, 3), dtype=jnp.float64)
    b = jnp.ones((3,), dtype=jnp.float64)
    return {"grad": jax.make_jaxpr(jax.grad(solve))(m, b), "jvp": jax.make_jaxpr(lambda m, b: jax.jvp(solve, (m, b), (m, b)))(m, b),
            "vjp": jax.make_jaxpr(lambda m, b: jax.vjp(solve, m, b)[1](1.0))(m, b)}


@pytest.mark.parametrize("program", ["grad", "jvp", "vjp"])
def test_the_derivative_programs_of_a_shared_solve_keep_its_scope(float64, program):
    """grad, jvp and vjp of ``dense_solve`` trace their own solves (the factorisation's transpose, the tangent solve), each
    inside the scope; the walk is clean on them, and on the same programs written raw it is not (the control)."""
    held = solve_walk.walk(_derivatives(True)[program].jaxpr)
    solving = [e for e in held if e.primitive in solves.SOLVE_PRIMITIVES]
    assert len(solving) >= 4 and all(e.innermost == "dense" for e in solving), [e.stack for e in solving if e.innermost != "dense"]
    assert solve_problems(_derivatives(True)[program].jaxpr) == []
    assert any("outside every registered solve scope" in line for line in solve_problems(_derivatives(False)[program].jaxpr))
