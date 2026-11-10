"""The two-builds check on the calibrated multipole's leaves (float64): its sources' coefficients and positions, which
the registry walk cannot move (the calibration is an asset in the document, not its numbers), its sources' directions, its
placement's orientation and its background, the
walk's own move of it repeated here. Each pair of builds differs in that one value; the trace form is quiet (the step's
program is the same: the value is a leaf the step reads), and the run form sees the leaf move the run. The control: the
same pairs traced with the parameters closed over (every leaf a constant of the program) fire on every pair."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import numpy as np
import pytest
from cases import coil_array

from mime.build._maddening import chunk_step, external_inputs, full_state, graph_params, inputs
from mime.fields import _coil_array
from mime.nodes.actuation._multipole import Calibration
from mime.testing import two_builds
from mime.testing._two_builds import nested_program, programs_differ
from mime.units import T

RELATIVE = 1e-6


def moved(held: Calibration, what: str) -> Calibration:
    """``held`` with one of its numbers moved by a relative 1e-6: the first coil's first source's coefficient, or its
    position's x."""
    if what == "coefficient":
        coefficients = held.coefficients.copy()
        coefficients[0, 0] *= 1.0 + RELATIVE
        return dataclasses.replace(held, coefficients=coefficients)
    if what == "direction":
        directions = held.directions.copy()
        directions[0, 0, 0] += RELATIVE
        return dataclasses.replace(held, directions=directions)
    positions = held.positions.copy()
    positions[0, 0, 0] *= 1.0 + RELATIVE
    return dataclasses.replace(held, positions=positions)


#: The case's own background (T), moved by the relative step for the background's pair.
BACKGROUND = (2e-5, 0.0, -4e-5)


def other_build(what: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    if what == "background":
        return coil_array(background=T(tuple(c * (1.0 + RELATIVE) for c in BACKGROUND))).build()
    if what == "orientation":
        turned = (0.96, 0.0, 0.0, 0.28 + RELATIVE)
        length = float(np.linalg.norm(turned))
        return coil_array(orientation=tuple(c / length for c in turned)).build()
    calibration_of = _coil_array.calibration_of
    with monkeypatch.context() as patched:
        patched.setattr(_coil_array, "calibration_of", lambda asset: moved(calibration_of(asset), what))
        return coil_array().build()


def closed_program(graph: Any) -> tuple[str, list[np.ndarray]]:
    """The step traced with the parameters closed over: every leaf a constant of the program."""
    step = chunk_step(graph)
    graph.reset_state()
    state = full_state(graph)
    params = graph_params(graph)
    fed: dict[str, dict[str, Any]] = {}
    for node, field in external_inputs(graph):
        spec = inputs(graph.get_node(node))[field]
        fed.setdefault(node, {})[field] = np.zeros(tuple(spec.shape), dtype=np.dtype(spec.dtype or np.float64))
    return nested_program(jax.make_jaxpr(lambda s: step(s, fed, params))(state))


@pytest.fixture(scope="module")
def base() -> Any:
    return coil_array().build()


@pytest.mark.parametrize("what", ["coefficient", "position", "direction", "orientation", "background"])
def test_a_moved_calibration_number_or_background_is_a_leaf_the_step_reads(base: Any, what: str,
                                                                          monkeypatch: pytest.MonkeyPatch):
    found = two_builds(base, other_build(what, monkeypatch))
    assert found.moved and not found.trace_fired, found.verdict
    assert found.verdict == "quiet"
    assert found.run is not None and found.run.ratio < 1.0


@pytest.mark.parametrize("what", ["coefficient", "position", "direction", "orientation", "background"])
def test_control_the_same_pair_closed_over_fires(base: Any, what: str, monkeypatch: pytest.MonkeyPatch):
    assert programs_differ(closed_program(base.graph), closed_program(other_build(what, monkeypatch).graph))
