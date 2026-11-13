"""The solve walk (design row #357) over every registered node kind: each node's step, traced in the switch scan's
configurations (``switch_configurations.traced_steps``, once per process and shared with the scan), holds no linear solve
that went through no shared function.

**What it reads.** The programs, not the source (``tests/solve_sites/solve_walk.py``). Each solve primitive is judged by its
innermost ``mime-solve`` scope; an exempt site's operands are held to its declared size; in a float64 run no equation of
a step holds a narrower float, inside a scope or outside, except in the steps of a kind that reaches a site declared
``narrower`` (the arm-tracking controller, which computes in single precision by design).

**Which kinds.** Every registered node class: a class no configuration traces fails here, as it does in the switch scan.
A kind is walked in the configurations of ``switch_configurations``; a solve in a branch or an option none of them takes is
not seen, the scan's own limit.

**The registry and the walk agree both ways.** A site names every node kind whose step reaches it, and no kind reaches a
site that does not name it: a site no step reaches guards nothing. A new solve registers with ``register()`` in the module
that performs it, or goes through ``dense_solve`` and adds its node kind to the ``dense`` site's ``nodes``."""

from __future__ import annotations

import dataclasses
import sys
from collections import defaultdict
from pathlib import Path

import pytest
from switch_configurations import traced_steps

#: Every test reads ``by_class``, which is ``traced_steps``: minutes the first time in a process (tests/_fixture_first.py).
pytestmark = pytest.mark.fixture_first

sys.path.append(str(Path(__file__).resolve().parents[1] / "solve_sites"))

# Every module that registers a site is imported, as the build imports them.
import mime.core.multibody  # noqa: E402, F401
import mime.nodes.flow.fvm  # noqa: E402, F401
import mime.nodes.hydro  # noqa: E402, F401
import mime.nodes.hydro.mrs  # noqa: E402, F401
import mime.nodes.hydro.near_field  # noqa: E402, F401
from mime.docs import load_namespaces  # noqa: E402
from mime.core.solves import SOLVE_PRIMITIVES, SOLVE_SITES, sites_of  # noqa: E402
from mime.registry import EntryKind, lookup  # noqa: E402
from mime.testing import installed_entries  # noqa: E402
from solve_walk import downcasts, narrow_anywhere, sites_reached, solve_problems, walk  # noqa: E402


def registered_node_classes() -> set[type]:
    load_namespaces()
    found = [lookup(EntryKind(kind), name, version) for kind, name, version in installed_entries()]
    return {node for record in found if record is not None for nodes in dict(record.cls.builds).values() for node in nodes}


@pytest.fixture(scope="module")
def by_class() -> dict[str, list]:
    found: dict[str, list] = defaultdict(list)
    for step in traced_steps():
        found[step.node_class.__name__].append(step)
    return found


def test_every_registered_node_class_is_traced_by_some_configuration(by_class):
    """A registered kind no configuration traces is a kind whose solves the walk cannot see."""
    assert sorted(cls.__name__ for cls in registered_node_classes() if cls.__name__ not in by_class) == []


def test_no_step_of_any_node_kind_holds_a_solve_outside_the_rule(by_class):
    """Each primitive in its innermost scope's ownership, each exempt site inside its size, no unregistered scope."""
    found = {name: sorted({line for step in steps for line in solve_problems(step.jaxpr)}) for name, steps in by_class.items()}
    assert {name: lines for name, lines in found.items() if lines} == {}


def test_the_registry_and_the_walk_agree_on_which_node_kind_reaches_which_site(by_class):
    reached = {name: set().union(*(sites_reached(step.jaxpr) for step in steps)) for name, steps in by_class.items()}
    claimed: dict[str, set[str]] = defaultdict(set)
    for site in SOLVE_SITES.values():
        for kind in site.nodes:
            claimed[kind].add(site.name)
    assert {name: found for name, found in reached.items() if found} == dict(claimed)


def test_the_build_reports_the_kinds_the_registry_names(by_class):
    """``sites_of`` (the build report's solving nodes) holds every class the walk reached a site in."""
    reached = {name for name, steps in by_class.items() if any(sites_reached(step.jaxpr) for step in steps)}
    assert reached == {name for name, steps in by_class.items() if sites_of(steps[0].node_class)}


NARROWER = {name for name, site in SOLVE_SITES.items() if site.narrower}


def test_a_float64_runs_steps_hold_no_narrower_float_but_by_design(by_class):
    """The whole step of every node kind that solves: a narrowing completed in the caller before a solve, or on its answer
    after it, is seen as well as one inside a scope. The kinds that reach a ``narrower`` site compute narrower by design,
    and are held to the scope check outside it."""
    found: dict[str, list[str]] = {}
    for name, steps in by_class.items():
        if not any(sites_reached(step.jaxpr) for step in steps):
            continue
        by_design = any(sites_reached(step.jaxpr) & NARROWER for step in steps)
        lines = sorted({line for step in steps
                        for line in (downcasts(step.jaxpr) if by_design else narrow_anywhere(step.jaxpr))})
        if lines:
            found[name] = lines[:3]
    assert found == {}


EXEMPT = sorted(name for name, site in SOLVE_SITES.items() if site.exempt)


@pytest.mark.parametrize("site", EXEMPT)
def test_control_an_exempt_sites_size_bound_is_checked_on_the_operands_the_walk_traces(by_class, site):
    """The bound is read off the steps' systems, not taken on trust: it holds the largest system the walk traces in the
    site, and the same site with a bound one below that is found. The largest order is the premise (a site whose steps
    trace no matrix would make the bound unchecked, so it fails here)."""
    held = SOLVE_SITES[site]
    scoped = [e for kind in held.nodes for step in by_class[kind] for e in walk(step.jaxpr)
              if e.innermost == site and e.primitive in SOLVE_PRIMITIVES and e.order]
    assert scoped, f"no step traces a matrix inside {site}"
    largest = max(e.order for e in scoped)
    assert largest <= held.bound
    tight = {**SOLVE_SITES, site: dataclasses.replace(held, bound=largest - 1)}
    caught = [step for kind in held.nodes for step in by_class[kind]
              if any(f"inside the exempt site {site!r}" in line for line in solve_problems(step.jaxpr, tight))]
    assert caught, f"{site}: a bound one below the largest order {largest} is not found"
