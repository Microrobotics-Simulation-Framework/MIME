"""schwarz_vessel_helix — physics graph builder (runner entry point).

Thin wrapper over the effects-first composer
``mime.experiments.schwarz_vessel_helix.build_graph``; the whole graph is assembled
via ``Experiment.attach`` there (no hand-wiring here, by design — the effects-first
``make_experiment`` migration).
"""

from __future__ import annotations

from mime.experiments.schwarz_vessel_helix import build_graph as _build_graph


def build_graph(params: dict):
    """Build the coupled two-scale gate graph from the runner's params namespace."""
    return _build_graph(params)
