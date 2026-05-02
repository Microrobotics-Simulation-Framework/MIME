"""Reusable experiment-graph builders.

Each submodule exposes a ``build_graph`` callable that returns a
configured :class:`maddening.core.graph_manager.GraphManager`. These
modules are imported both by ad-hoc scripts and by the runner-style
``physics/setup.py`` adapters under ``MIME/experiments/``.
"""
