"""A study member's stage samples: the bound fields of its state (``BuiltExperiment.binding``: each body's pose, each
chain frame's, each field vector) at the base step it began and then every ``stage_every``, kept beside its outputs so
that a ghosted comparison of the study's members (``compare.usda``) is written from what each member recorded. A batched
member has no recording layer of its own, because the members of a batch run together on one compiled step and their
states are kept stacked; this is each member's share of that.

In a study folder a member's samples are entries of its group's NPZ, under ``m<point>/stage/``: ``steps`` (the base step
each sample is the state at the end of, from step 0), ``meta`` (JSON text: the binding's entries, the stage's up axis
and metres per unit, the base step, the experiment's name and input hash) and ``fields/<node>/<field>`` (the field's
values at each sample, in the state's own float type). Plain arrays and JSON: readable without MIME."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

STAGE_PREFIX = "stage/"
STAGE_STEPS = f"{STAGE_PREFIX}steps"
STAGE_META = f"{STAGE_PREFIX}meta"
STAGE_FORMAT = "mime-stage-samples/1"


def field_key(node: str, field: str) -> str:
    return f"{STAGE_PREFIX}fields/{node}/{field}"


@dataclass(frozen=True)
class StageRecord:
    """A member's bound fields at ``steps`` (sample 0 is the state it began from), ``fields`` by (node, field), and the
    ``meta`` that places them on a stage: ``binding`` (the entries of ``StageBinding.record()``), ``up``,
    ``meters_per_unit``, ``dt`` (the member's base step, s), ``experiment`` and ``input_hash``."""

    steps: NDArray[np.int64]
    fields: Mapping[tuple[str, str], NDArray[Any]]
    meta: Mapping[str, Any]

    @property
    def dt(self) -> float:
        return float(self.meta["dt"])

    @property
    def every(self) -> int:
        """The base steps between two samples."""
        return int(self.steps[1] - self.steps[0]) if len(self.steps) > 1 else 0

    def entries(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self.meta["binding"])

    def rows(self, entry: Mapping[str, Any]) -> NDArray[np.float64]:
        """The samples of one bound entry, double precision: ``(samples, 7)`` for a pose, ``(samples, 3)`` for a vector."""
        values = np.asarray(self.fields[(str(entry["node"]), str(entry["field"]))], dtype=np.float64)
        return values if entry["index"] is None else values[:, int(entry["index"])]

    def arrays(self) -> dict[str, NDArray[Any]]:
        """What a study's group NPZ holds for this member, under ``stage/``."""
        found: dict[str, NDArray[Any]] = {STAGE_STEPS: np.asarray(self.steps, dtype=np.int64),
                                          STAGE_META: np.asarray(json.dumps({"format": STAGE_FORMAT, **self.meta},
                                                                            sort_keys=True))}
        found.update({field_key(node, name): np.asarray(value) for (node, name), value in self.fields.items()})
        return found

    @classmethod
    def read(cls, files: Sequence[str], get: Any) -> StageRecord | None:
        """The record in an archive (its member-relative keys ``files``, ``get`` one array by key), or ``None`` where the
        member kept none."""
        if STAGE_STEPS not in files:
            return None
        meta = cast(dict[str, Any], json.loads(str(get(STAGE_META))))
        meta.pop("format", None)
        fields: dict[tuple[str, str], NDArray[Any]] = {}
        for key in files:
            if key.startswith(f"{STAGE_PREFIX}fields/"):
                node, _, name = key.removeprefix(f"{STAGE_PREFIX}fields/").rpartition("/")
                fields[(node, name)] = get(key)
        return cls(np.asarray(get(STAGE_STEPS), dtype=np.int64), fields, meta)


def sampled(initial: Mapping[str, Mapping[str, Any]], kept: Mapping[str, Mapping[str, Any]],
            fields: Sequence[tuple[str, str]], steps: int, every: int) -> tuple[NDArray[np.int64],
                                                                              dict[tuple[str, str], NDArray[Any]]]:
    """The bound ``fields`` at step 0 (``initial``) and at every ``every`` steps to ``steps`` (``kept``: each field at
    every step, from step 1, on the member's own axis)."""
    counts = np.arange(1, steps + 1, dtype=np.int64)
    chosen = counts % every == 0
    taken = np.concatenate([[0], counts[chosen]]).astype(np.int64)
    found: dict[tuple[str, str], NDArray[Any]] = {}
    for node, name in fields:
        first = np.asarray(initial[node][name])
        rest = np.asarray(kept[node][name])[chosen]
        found[(node, name)] = np.concatenate([first[None], rest.astype(first.dtype)], axis=0)
    return taken, found


__all__ = ["STAGE_FORMAT", "STAGE_META", "STAGE_PREFIX", "STAGE_STEPS", "StageRecord", "field_key", "sampled"]
