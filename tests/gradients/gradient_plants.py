"""The gradient gates' plants, each switched on in a test by its environment variable (what
``.github/probe/detection_rate.py`` sets to measure a control's rate) or by a control test directly:

- ``MIME_PLANT_GRADIENT``: ``baked``, a value baked at build (the MRS drag's held resistance read as built, never made
  from the traced radius; the tube's penalty viscosity taken from the build); ``stopped``, ``stop_gradient`` on a leaf
  the path uses (the radius the MRS drag makes its resistance from; the viscosity in the tube's kinematic viscosity);
- ``MIME_PLANT_GRADIENT_DROP``: ``flag``, ``refusal`` or ``extrapolation``, the result's assembly dropping that item;
- ``MIME_PLANT_SOLVE_CAP``: ``primal`` or ``transpose``, the tube's solve capped at one iteration on that solve alone;
- ``MIME_PLANT_GUARD``: ``nan``, the MRS drag's stop writing NaN in a differentiated run too (no flag form)."""

from __future__ import annotations

import dataclasses
import os
from contextlib import contextmanager

import jax
import pytest

import mime.nodes.flow.fvm._node as flow_node
import mime.nodes.flow.fvm._solve as solve_module
import mime.nodes.hydro.mrs._node as mrs_node
import mime.study._gradient as gradient_module
from mime.nodes.flow.fvm import ImplicitFlowNode
from mime.nodes.hydro.mrs import MRSDragNode

GRADIENT, DROP, CAP, GUARD = "MIME_PLANT_GRADIENT", "MIME_PLANT_GRADIENT_DROP", "MIME_PLANT_SOLVE_CAP", "MIME_PLANT_GUARD"


def from_environment(variable: str) -> str | None:
    return os.environ.get(variable) or None


def gradient_plant(patch: pytest.MonkeyPatch, kind: str | None) -> None:
    """``baked`` or ``stopped`` (``None``: nothing), on both consumers' nodes."""
    if kind is None:
        return
    made, resistivity, kinematic = MRSDragNode.run_constants, ImplicitFlowNode.resistivity, ImplicitFlowNode.kinematic
    if kind == "baked":
        patch.setattr(MRSDragNode, "run_constants", lambda self, params: (
            {mrs_node.HELD: params[mrs_node.HELD]} if mrs_node.HELD in params else {}))
        patch.setattr(ImplicitFlowNode, "resistivity", lambda self, p, fields: resistivity(
            self, {**p, "viscosity": float(self.constants()["viscosity"])}, fields))
    elif kind == "stopped":
        patch.setattr(MRSDragNode, "run_constants", lambda self, params: made(
            self, {**params, "shape_radius": jax.lax.stop_gradient(params["shape_radius"])}))
        patch.setattr(ImplicitFlowNode, "kinematic", staticmethod(lambda p: kinematic(
            {**p, "viscosity": jax.lax.stop_gradient(p["viscosity"])})))
    else:
        raise ValueError(f"{GRADIENT} is baked or stopped, not {kind!r}")


def drop_plant(patch: pytest.MonkeyPatch, kind: str | None) -> None:
    """The result's assembly dropping ``flag``, ``refusal`` or ``extrapolation``."""
    if kind is None:
        return
    if kind == "flag":
        patch.setattr(gradient_module, "_flags", lambda prepared, tally: ())
    elif kind == "refusal":
        patch.setattr(gradient_module, "stopped_refusals", lambda graph, final: [])
    elif kind == "extrapolation":
        patch.setattr(gradient_module, "_extrapolations", lambda point: ())
    else:
        raise ValueError(f"{DROP} is flag, refusal or extrapolation, not {kind!r}")


def cap_plant(patch: pytest.MonkeyPatch, which: str | None) -> None:
    """The tube's solve capped at one iteration on its primal solve (and so its tangent solves, which JAX makes with
    the same solve) or on its transpose solve alone: the solve given its transpose, each with its own stopping."""
    if which is None:
        return
    if which not in ("primal", "transpose"):
        raise ValueError(f"{CAP} is primal or transpose, not {which!r}")
    def capped(grid, apply, diagonal, rhs, guess, stopping, tap, preconditioner, magnitude=None):
        one = dataclasses.replace(stopping, cap=1, check_every=1)
        primal, transpose = (one, stopping) if which == "primal" else (stopping, one)
        return _projected_two(grid, apply, diagonal, rhs, guess, primal, transpose, tap, preconditioner, magnitude)

    patch.setattr(flow_node, "projected_solve", capped)


def _projected_two(grid, apply, diagonal, rhs, guess, primal, transpose, tap, preconditioner, magnitude):
    """``projected_solve`` with its transpose solve given apart, so that each takes its own stopping."""
    lax = jax.lax
    project = grid.project
    inner = grid.inner
    precondition_raw = preconditioner.made(diagonal)
    projected_guess = None if guess is None else project(guess)

    def precondition(r):
        return project(precondition_raw(project(r)))

    def matvec(pair):
        x, block = pair
        y = project(apply(project(x)))
        return y, block + 0.0 * y[0][0, 0, 0]

    def solver(stopping):
        def solve(operator, pair):
            b, block = pair

            def field_operator(x):
                return operator((x, block))[0]

            b = project(b)
            start = solve_module.scaling_start(projected_guess, b, field_operator, inner)
            x, report = solve_module.guarded_cg(field_operator, b, precondition, start, stopping, inner, magnitude)
            return project(x), report
        return solve

    coupled_rhs = solve_module._add(rhs, (tap[0], tap[0], tap[0]), 0.0)  # noqa: SLF001
    coupled_tap = tap + 0.0 * rhs[0][0, 0, 0]
    return lax.custom_linear_solve(matvec, (coupled_rhs, coupled_tap), solver(primal),
                                   transpose_solve=solver(transpose))


def guard_plant(patch: pytest.MonkeyPatch, kind: str | None) -> None:
    """``nan``: the MRS drag's stop writes NaN in a differentiated run as in a forward one."""
    if kind is None:
        return
    if kind != "nan":
        raise ValueError(f"{GUARD} is nan, not {kind!r}")
    patch.setattr(mrs_node, "differentiated", lambda p: False)


@contextmanager
def stopping_every_step():
    """The MRS drag's guard made to stop at every step (its least eigenvalue judged against a line above any): a
    planted trigger, so that the flag form is what is under test."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mrs_node, "stops_where", lambda accepted, least: accepted & (least < 1e300))
        yield


__all__ = ["CAP", "DROP", "GRADIENT", "GUARD", "cap_plant", "drop_plant", "from_environment", "gradient_plant",
           "guard_plant", "stopping_every_step"]
