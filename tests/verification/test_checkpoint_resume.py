"""§5 M2-M4 — checkpoint schema adoption + preempt/resume facilities.

Two preemption-resilience patterns, both pinned here:

* **Sweeps** use ``mime.data.sweep_resume.ResumableSweep`` — the reusable
  facility for combo-level resume: skip finished items, checkpoint each one
  with a MADDENING v0.2 integrity manifest.
* **Single-graph experiments** use MADDENING's graph-state checkpoint API
  (``save_state_with_manifest`` / ``load_state_with_manifest``).
"""

import numpy as np
import pytest

from maddening.core.simulation.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointIntegrityError,
    load_state_with_manifest,
    save_state_with_manifest,
)
from mime.data.sweep_resume import ResumableSweep, progress_path
from mime.experiments.dejongh import default_mlp_weights_path

# The dejongh graph builds an MLPResistanceNode from the trained Cholesky-MLP
# weights — a gitignored artifact (too large for the repo) that is absent in CI
# and fresh checkouts. Skip the build-dependent test there; it still runs wherever
# the data tree is present (mirrors test_mlp_clamp_and_accuracy.py's convention).
_requires_mlp_weights = pytest.mark.skipif(
    not default_mlp_weights_path().exists(),
    reason=f"MLP weights artifact absent (gitignored): {default_mlp_weights_path()}",
)


def _items(n):
    return [{"label": f"c{i}", "value": i} for i in range(n)]


_KEY = lambda it: it["label"]  # noqa: E731 — terse test key fn


# --- ResumableSweep: the sweep preempt/resume facility -------------------

def test_resumable_sweep_resumes_after_interruption(tmp_path):
    """A sweep relaunched after an interruption skips finished items and
    yields only the unfinished ones — the preemption-survival contract."""
    items = _items(5)
    ckpt = tmp_path / "sweep.h5"

    # First run: finish 3 items, then the process "dies".
    s1 = ResumableSweep(items, key=_KEY, checkpoint_path=ckpt)
    for it in s1.pending[:3]:
        s1.record(it, {"label": it["label"], "ok": True})

    # Relaunch on the same box — resumes from the local checkpoint.
    s2 = ResumableSweep(items, key=_KEY, checkpoint_path=ckpt)
    assert s2.n_done == 3
    assert {r["label"] for r in s2.completed} == {"c0", "c1", "c2"}
    assert [it["label"] for it in s2.pending] == ["c3", "c4"]

    for it in s2.pending:
        s2.record(it, {"label": it["label"], "ok": True})
    assert s2.n_done == 5 and s2.pending == []


def test_resumable_sweep_resume_from_durable_snapshot(tmp_path):
    """Resume on a *fresh* machine from a snapshot mirrored to durable
    storage — the cloud-preemption case."""
    items = _items(4)
    ckpt1 = tmp_path / "box1" / "sweep.h5"
    ckpt1.parent.mkdir()
    snap = tmp_path / "durable"

    s1 = ResumableSweep(items, key=_KEY, checkpoint_path=ckpt1,
                        snapshot_dir=snap)
    for it in s1.pending[:2]:
        s1.record(it, {"label": it["label"]})

    # Fresh box: only the durable snapshot is available.
    ckpt2 = tmp_path / "box2" / "sweep.h5"
    s2 = ResumableSweep(
        items, key=_KEY, checkpoint_path=ckpt2,
        resume_from=f"file://{snap / 'sweep.h5.progress.json'}")
    assert [it["label"] for it in s2.pending] == ["c2", "c3"]


def test_resumable_sweep_missing_resume_source_starts_fresh(tmp_path):
    """A resume URL pointing at a not-yet-existing checkpoint is a fresh
    start, not an error — so a job spec can set it unconditionally."""
    s = ResumableSweep(
        _items(3), key=_KEY, checkpoint_path=tmp_path / "sweep.h5",
        resume_from=f"file://{tmp_path / 'nonexistent.json'}")
    assert s.n_done == 0
    assert len(s.pending) == 3


def test_resumable_sweep_rejects_tampered_checkpoint(tmp_path):
    """The MADDENING integrity manifest catches a corrupted checkpoint."""
    items = _items(2)
    ckpt = tmp_path / "sweep.h5"
    s1 = ResumableSweep(items, key=_KEY, checkpoint_path=ckpt)
    s1.record(items[0], {"label": "c0"})

    p = progress_path(ckpt)
    p.write_text(p.read_text() + "\n# tampered")
    with pytest.raises(CheckpointIntegrityError):
        ResumableSweep(items, key=_KEY,
                       checkpoint_path=tmp_path / "box2" / "sweep.h5",
                       resume_from=f"file://{p}")


# --- Single-graph checkpoint (per-graph experiments) --------------------

@_requires_mlp_weights
def test_single_graph_checkpoint_round_trip(tmp_path):
    """A MIME experiment graph round-trips through MADDENING's graph-state
    checkpoint API — the preempt/resume path for non-sweep experiments."""
    from mime.experiments.dejongh import build_graph

    src = build_graph()
    src.compile()
    for _ in range(5):
        src.step()

    npz, manifest = save_state_with_manifest(
        src, tmp_path / "snap.npz", extra={"steps": 5})
    assert npz.exists() and manifest.exists()
    assert len(np.load(npz, allow_pickle=True).files) > 0  # state captured

    dst = build_graph()
    dst.compile()
    info = load_state_with_manifest(dst, npz)  # verifies manifest, then loads
    assert info["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert info["extra"]["steps"] == 5
