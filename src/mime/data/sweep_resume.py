"""Preempt/resume support for parameter-sweep experiments (v0.2 fit-up §5).

A sweep runs many independent work items. :class:`ResumableSweep` makes such
a sweep preemption-resilient: each finished item is written to a JSON
progress checkpoint — carrying a MADDENING v0.2 integrity manifest
(``write_manifest`` / ``CHECKPOINT_SCHEMA_VERSION``) — so a relaunch skips
the finished items and runs only what is left.

:class:`ResumableSweep` is the facility to use when **writing a sweep
experiment script**; see its docstring for the pattern, ``docs/preempt_resume.md``
for both patterns side by side, and ``scripts/run_confinement_sweep.py`` for
a full worked example. The module-level :func:`save_progress` /
:func:`load_progress` / :func:`fetch_resume` helpers are the underlying
checkpoint primitives, exposed for callers that want finer control.

For **single-graph experiments** (one long-running graph rather than a
sweep) use MADDENING's graph-state checkpoint API directly —
``save_state_with_manifest`` / ``download_and_load_state`` /
``make_preempt_snapshot_hook``. See ``docs/preempt_resume.md``.

Files written beside the sweep's output path ``<out>``:

* ``<out>.progress.json``               — ``{"completed": [<record>, ...]}``
* ``<out>.progress.json.manifest.json`` — MADDENING integrity manifest
"""
from __future__ import annotations

import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from maddening.core.simulation.checkpoint import verify_manifest, write_manifest


class ResumableSweep:
    """Combo-level preempt/resume for a parameter-sweep experiment.

    Wrap a list of work items; finished items are checkpointed as they
    complete, so a relaunch skips them. Minimal pattern for a new sweep
    script (``scripts/run_confinement_sweep.py`` is the full example)::

        import os
        from mime.data.sweep_resume import ResumableSweep

        sweep = ResumableSweep(
            items=my_param_combos,
            key=lambda combo: combo["label"],          # unique id per item
            checkpoint_path="data/my_experiment.h5",   # progress sits beside it
            resume_from=os.environ.get("SWEEP_RESUME_FROM"),
            snapshot_dir=os.environ.get("SWEEP_SNAPSHOT_DIR"),
        )

        for result in sweep.completed:     # items finished in a prior run —
            write_output(result)           #   replay them into the output

        for combo in sweep.pending:        # only the unfinished items
            result = run_one(combo)
            write_output(result)
            sweep.record(combo, result)    # checkpointed immediately

    ``resume_from`` accepts a path, ``file://`` URL, or ``http(s)://`` URL; a
    missing source is treated as a fresh start (so a job spec may set it
    unconditionally). With no ``resume_from`` a relaunch on the same machine
    still resumes from the local checkpoint beside ``checkpoint_path``. Point
    ``snapshot_dir`` at durable storage (a mounted volume/bucket) so the
    checkpoint survives an instance teardown.
    """

    def __init__(self, items, key, checkpoint_path, *,
                 resume_from=None, snapshot_dir=None):
        self._items = list(items)
        self._key = key
        self._checkpoint_path = checkpoint_path
        self._snapshot_dir = snapshot_dir or None
        if resume_from:
            try:
                self._records = fetch_resume(resume_from, checkpoint_path)
            except FileNotFoundError:
                self._records = []  # nothing to resume — fresh start
        else:
            self._records = load_progress(checkpoint_path)

    @property
    def completed(self) -> list:
        """Results of items finished in a previous run (oldest first)."""
        return [rec["result"] for rec in self._records]

    @property
    def pending(self) -> list:
        """The items not yet finished — iterate these and run them."""
        done = {rec["key"] for rec in self._records}
        return [it for it in self._items if self._key(it) not in done]

    def record(self, item, result) -> None:
        """Mark ``item`` finished with ``result`` and persist the checkpoint
        (mirroring it to ``snapshot_dir`` when one was given)."""
        self._records.append({"key": self._key(item), "result": result})
        save_progress(self._checkpoint_path, self._records,
                      snapshot_dir=self._snapshot_dir)

    @property
    def n_total(self) -> int:
        """Total number of items in the sweep."""
        return len(self._items)

    @property
    def n_done(self) -> int:
        """Number of items finished so far (replayed + recorded)."""
        return len(self._records)


# --- Checkpoint primitives (used by ResumableSweep; exposed for reuse) ----

def progress_path(checkpoint_path) -> Path:
    """Path of the progress checkpoint beside a sweep's output file."""
    return Path(str(checkpoint_path) + ".progress.json")


def load_progress(checkpoint_path) -> list:
    """Records from a local progress checkpoint ([] if none exists)."""
    p = progress_path(checkpoint_path)
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("completed", [])


def save_progress(checkpoint_path, records, *, snapshot_dir=None) -> None:
    """Atomically write the progress checkpoint + its integrity manifest,
    and (if ``snapshot_dir`` is given) mirror both into that durable dir."""
    p = progress_path(checkpoint_path)
    _atomic_write(p, json.dumps({"completed": records},
                                indent=1, default=_json_default))
    write_manifest(p, extra={"kind": "sweep_progress",
                             "n_records": len(records)})
    if snapshot_dir:
        snap = Path(snapshot_dir)
        snap.mkdir(parents=True, exist_ok=True)
        for src in (p, Path(str(p) + ".manifest.json")):
            _atomic_copy(src, snap / src.name)


def fetch_resume(url, checkpoint_path) -> list:
    """Fetch a prior progress checkpoint from ``url`` into place beside
    ``checkpoint_path``, verify its manifest, and return the records.

    ``url`` may be a bare path, a ``file://`` URL, or ``http(s)://``.
    """
    p = progress_path(checkpoint_path)
    _fetch(url, p)
    try:
        _fetch(url + ".manifest.json", Path(str(p) + ".manifest.json"))
        verify_manifest(p)
    except FileNotFoundError:
        pass  # a snapshot without a manifest — tolerate (no integrity check)
    return load_progress(checkpoint_path)


# --- internals -----------------------------------------------------------

def _json_default(o):
    if hasattr(o, "item"):      # numpy / jax 0-d scalar
        return o.item()
    if hasattr(o, "tolist"):    # ndarray
        return o.tolist()
    return str(o)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in ("", "file"):
        src = Path(parsed.path if parsed.scheme else url)
        if not src.exists():
            raise FileNotFoundError(f"resume source not found: {src}")
        dest.write_bytes(src.read_bytes())
    elif parsed.scheme in ("http", "https"):
        with urllib.request.urlopen(url) as r:  # noqa: S310 — trusted resume URL
            dest.write_bytes(r.read())
    else:
        raise ValueError(f"unsupported resume URL scheme: {parsed.scheme!r}")
