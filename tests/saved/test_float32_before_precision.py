"""Saves made before precision was declared, in a float32 run (JAX's 64-bit flag off, the lock's ``x64: false``), load
and run as they did, without the suite's test-only bridge: as declared float32, with 64-bit types enabled and the
parameters built in float32, so that the graph each rebuilds is the graph its lock records (D2: a float32 save may move
in its last bits under 64-bit types; it is never refused).

Each folder under ``float32_before_precision/`` was written by ``built.save(folder, steps=2)`` at the merge base before
precision P1 (``501dd7e7``), with JAX's flag off and one BLAS thread: script A, and the kinds a float32 build under
64-bit types first broke at. Before the fixes, script A's tube radius was rebuilt in float64 (refused
``load.derived-hash`` in the environment it was saved in; the comparison of numbers here sees it in any environment); a
camera's disc coverage and the MRS drag's lubricated hand-over raised in a ``cond`` whose branches came out in two
dtypes; the lab fixture's gantry, whose state took JAX's flag for its dtype, was refused ``maddening.dtype``; a
controller whose start did was refused ``controller.untraceable``, and another's start and a toy's static array were
rebuilt in float64.

The load is the hard order, as ``mime run`` makes it: the lock's precision applied first (64-bit types on) and the
document read after. The derived graph is compared number for number, every array by its dtype and shape (its bytes
are the load's own check, within the lock's tolerance across environments)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from mime._lock_json import read_lock
from mime.core import MimeNode, run_dtype
from mime.core._precision import FROM_JAX_FLAG
from mime.env._check import apply_lock_precision
from mime.experiment import Experiment
from mime.saved._folder import differences
from mime.units import s

SAVES = Path(__file__).resolve().parent / "float32_before_precision"


def _shapes_only(tree: Any) -> Any:
    """``tree`` with each array record's bytes left out: its dtype and shape kept."""
    if isinstance(tree, dict):
        return {key: _shapes_only(value) for key, value in tree.items() if key != "sha256"}
    if isinstance(tree, list):
        return [_shapes_only(item) for item in tree]
    return tree


@pytest.mark.parametrize("name", sorted(folder.name for folder in SAVES.iterdir() if (folder / "mime.lock").is_file()))
def test_a_float32_save_from_before_precision_loads_and_runs_without_the_bridge(monkeypatch, name):
    monkeypatch.delenv(FROM_JAX_FLAG)
    folder = SAVES / name
    lock = read_lock(folder / "mime.lock")
    assert lock["environment"]["x64"] is False, "a float32 save, as JAX's flag off wrote it"
    jax.config.update("jax_enable_x64", False)
    apply_lock_precision(folder)
    loaded = Experiment.load(folder)
    built = loaded.build()
    assert jax.config.jax_enable_x64 and run_dtype() == np.float32
    assert built.record.environment["x64"] is False
    for section in ("physics", "observations"):
        saved, rebuilt = _shapes_only(lock["derived"][section]), _shapes_only(getattr(built.record.derived, section))
        assert differences(saved, rebuilt, section) == []
    built.run(s(2 * built.plans.dt))
    # MIME's nodes step in float32 (a plain MADDENING node, the toys', computes in its own dtype: precision's P2)
    ours = [name for name in built.graph.node_names if isinstance(built.graph.get_node(name), MimeNode)]
    floating = {str(np.asarray(value).dtype) for node, fields in built.current_state().items() if node in ours
                for value in fields.values() if np.asarray(value).dtype.kind == "f"}
    assert floating <= {"float32"}
