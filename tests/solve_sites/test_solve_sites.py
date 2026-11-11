"""The solve-site registry (design row #357): every linear solve a registered node kind's step traces went through a
shared solve function, and in a float64 run none of them narrows.

The walk reads the traced program (``solve_walk``), not the source: a solve is seen wherever it is called from. Its
controls are planted, each a way a solve could skip the seam, and the clean tests below fail on every one of them when
``MIME_PLANT_SOLVE`` names it (``.github/probe/detection_rate.py`` measures that rate). A control that is not caught
would make "the walk is clean" no evidence, so each plant is also asserted caught, in process, here."""

from __future__ import annotations

import jax
import pytest
import solve_walk
from solve_walk import PLANTS, cases, downcasts, planted_from_environment, sites_reached, unregistered

# Every module that registers a site is imported, as the build imports them.
import mime.nodes.flow.fvm  # noqa: F401
import mime.nodes.hydro.mrs  # noqa: F401
import mime.nodes.hydro.near_field  # noqa: F401
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
def test_every_solve_a_registered_node_step_reaches_went_through_a_registered_site(dtype, traced, label):
    assert unregistered(traced(label)) == []


@pytest.mark.parametrize("label", sorted(CASES))
def test_a_float64_runs_registered_solves_hold_no_narrower_float(float64, traced, label):
    assert downcasts(traced(label)) == []


def test_the_registry_and_the_walk_agree_on_which_node_kind_reaches_which_site(dtype, traced):
    """Both ways: a site names every node kind that reaches it, and no kind reaches a site that does not name it. A
    site no walk reaches is a registration that guards nothing."""
    reached: dict[str, set[str]] = {}
    for label, (kind, _) in CASES.items():
        reached.setdefault(kind, set()).update(sites_reached(traced(label)))
    claimed: dict[str, set[str]] = {}
    for site in SOLVE_SITES.values():
        for kind in site.nodes:
            claimed.setdefault(kind, set()).add(site.name)
    assert reached == claimed


def test_a_site_nobody_walks_says_why_and_is_exempt():
    for site in SOLVE_SITES.values():
        if not site.nodes:
            assert site.unwalked and site.exempt, site


@pytest.mark.parametrize("plant", sorted(PLANTS))
def test_control_each_plant_is_caught_on_the_cases_it_touches(float64, traced, plant):
    """A planted bypass, a planted direct solve and a planted downcast each fail the check named for them, on every
    case whose program they touch."""
    _, named, touched = PLANTS[plant]
    check = {"unregistered": unregistered, "downcasts": downcasts}[named]
    assert [label for label in touched if not check(traced(label, plant))] == []
