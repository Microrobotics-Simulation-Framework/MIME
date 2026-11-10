"""The gradient gates' plants, each switched on in a test by its environment variable (what
``.github/probe/detection_rate.py`` sets to measure a control's rate) or by a control test directly:

- ``MIME_PLANT_GRADIENT``: ``baked``, a value baked at build (the MRS drag's held resistance read as built, never made
  from the traced radius; the tube's penalty viscosity taken from the build); ``stopped``, ``stop_gradient`` on a leaf
  the path uses (the radius the MRS drag makes its resistance from; the viscosity in the tube's kinematic viscosity);
- ``MIME_PLANT_GRADIENT_DROP``: ``flag``, ``refusal`` or ``extrapolation``, the result's assembly dropping that item;
- ``MIME_PLANT_SOLVE_CAP``: ``primal`` or ``transpose``, the tube's solve capped at one iteration, through the node's own
  ``cap`` setting or the seam by which the solve takes its transpose solves' stopping apart (:func:`cap_plant`);
- ``MIME_PLANT_SOLVE_REPORT``: ``tangent``, ``transpose``, ``warning`` or ``recheck``, a part of the solve-report
  machinery broken (:func:`report_plant`);
- ``MIME_PLANT_GUARD``: ``nan``, the MRS drag's stop writing NaN in a differentiated run too (no flag form)."""

from __future__ import annotations

import dataclasses
import os
from contextlib import contextmanager, nullcontext

import jax
import pytest

import mime.nodes.flow.fvm._solve as solve_module
import mime.nodes.hydro.mrs._node as mrs_node
import mime.study._gradient as gradient_module
from mime.core import MimeNode
from mime.nodes.flow.fvm import ImplicitFlowNode
from mime.nodes.hydro.mrs import MRSDragNode

GRADIENT, DROP, CAP, GUARD, REPORT = ("MIME_PLANT_GRADIENT", "MIME_PLANT_GRADIENT_DROP", "MIME_PLANT_SOLVE_CAP",
                                      "MIME_PLANT_GUARD", "MIME_PLANT_SOLVE_REPORT")


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
    """The tube's solve capped at one iteration, through what the production solve reads: ``primal``, the node's own
    ``cap`` setting (its constants: the primal solve and so the tangent solves JAX makes with it, and the transpose
    solves, which share its stopping); ``transpose``, ``mime.nodes.flow.fvm._solve.transposed``, the seam through which
    the solve takes its transpose solves' stopping apart, here a cap of one on them alone. Either lasts only until the
    recheck lifts the cap: the transpose plant caps no stopping that was lifted."""
    if which is None:
        return
    if which not in ("primal", "transpose"):
        raise ValueError(f"{CAP} is primal or transpose, not {which!r}")
    if which == "primal":
        patch.setattr(ImplicitFlowNode, "constants", lambda self, params=None: {
            **MimeNode.constants(self, params), "cap": 1})
    else:
        patch.setattr(solve_module, "transposed", lambda stopping: dataclasses.replace(stopping, cap=1)
                      if stopping.cap <= solve_module.DEFAULT_CAP else stopping)


def report_plant(patch: pytest.MonkeyPatch, kind: str | None) -> None:
    """The result's solve-report machinery broken, one part at a time (``MIME_PLANT_SOLVE_REPORT``): ``tangent``,
    forward mode's tangent solves' report zeroed; ``transpose``, reverse mode's transpose report zeroed; ``warning``,
    the warnings dropped; ``recheck``, the recheck re-solving nothing (its program traced without the nodes marked, so
    it reports a change of zero)."""
    if kind is None:
        return
    if kind in ("tangent", "transpose"):
        original = gradient_module._report  # noqa: SLF001
        mode = "forward" if kind == "tangent" else "reverse"

        def zeroed(solving, final, out, backward, taken):
            if taken == mode:
                backward = {node: jax.numpy.zeros_like(held) for node, held in backward.items()}
            return original(solving, final, out, backward, taken)

        patch.setattr(gradient_module, "_report", zeroed)
    elif kind == "warning":
        patch.setattr(gradient_module, "_warnings", lambda reports: ())
    elif kind == "recheck":
        patch.setattr(gradient_module, "rechecked_nodes", lambda names: nullcontext())
    else:
        raise ValueError(f"{REPORT} is tangent, transpose, warning or recheck, not {kind!r}")


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


__all__ = ["CAP", "DROP", "GRADIENT", "GUARD", "REPORT", "cap_plant", "drop_plant", "from_environment",
           "gradient_plant", "guard_plant", "report_plant", "stopping_every_step"]
