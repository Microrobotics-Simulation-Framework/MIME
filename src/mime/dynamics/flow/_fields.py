"""The flow field kinds' declarations under the solved-field contract (SF-1 to SF-11), which SF-12's suite holds each
kind to: the FVM's by its scheme (its ledger route and its steady solve are its scheme's), the prescribed flow's on
its entry."""

from __future__ import annotations

from mime.fields._solved import NOT_DECLARED, SolvedField
from mime.nodes.flow.fvm import scheme_named
from mime.scene import PeriodicBox
from mime.vocabulary import FLOW_BODY_FORCE, FLOW_VELOCITY, FLOW_WAVEFORM


def fvm_field(scheme: str) -> SolvedField:
    """The FVM's declarations with the scheme ``scheme``."""
    chosen = scheme_named(scheme)
    return SolvedField(
        hosting="solved", role="flow", provides=(FLOW_VELOCITY.name,), time_level=NOT_DECLARED,
        own_contribution="none", accepts=(FLOW_BODY_FORCE.name, FLOW_WAVEFORM.name), exchange=NOT_DECLARED,
        outside=False, domain_kinds=(PeriodicBox.group,), ports="none",
        properties=("viscosity", "density", "inverse_permeability"),
        settings=("domain", "spacing", "cells_across", "scheme", "stokes", "tolerance", "preconditioner", "wall",
                  "solve_report_rows"),
        ledger_route=chosen.ledger_route, ledger_reason="", steady_solve=chosen.steady,
        affine_response=False, stores="the fluid's inertia" if chosen.stores_inertia else "nothing", batches=True,
        dtypes=("float32", "float64"), rates=NOT_DECLARED)


__all__ = ["fvm_field"]
