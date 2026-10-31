"""What a run's result is read by: each observation's declared outputs and each flag an entry raises, placed on the
graph's nodes when the experiment is built. Plain data, so that a result read back from a run folder has the same
layout as the run's own, and the manifest that folder holds is written from it."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mime.registry import OutputKind, Reduction
from mime.units import Dimension, unit_symbol
from mime.validity import Stamp


@dataclass(frozen=True)
class OutputLayout:
    """An observation's declared output as built: its observation's id, its name, the graph node and state field that
    hold it, when it has a value, its dimension and shape, and for an output at the end its window's start (s, read
    from its setting when the experiment was built; ``None`` from the run's start) and, when the graph reduces it from
    a per-step output, that output and the rule."""

    observation: str
    name: str
    node: str
    field: str
    kind: OutputKind
    dimension: Dimension
    shape: tuple[int, ...]
    description: str
    reduces: str | None = None
    rule: Reduction | None = None
    after: float | None = None

    @property
    def path(self) -> str:
        """``<observation id>.<output>``, how ``record=`` and every export name it."""
        return f"{self.observation}.{self.name}"

    @property
    def unit(self) -> str:
        return unit_symbol(self.dimension)


@dataclass(frozen=True)
class ObservationLayout:
    """An observation as built: its id, its entry's label and its outputs."""

    id: str
    entry: str
    outputs: tuple[OutputLayout, ...]

    def output(self, name: str) -> OutputLayout | None:
        return next((output for output in self.outputs if output.name == name), None)


@dataclass(frozen=True)
class FlagLayout:
    """A flag a built entry raises: the entry's id and label, the node and state field that hold it, and what it means
    when it is raised."""

    owner: str
    entry: str
    node: str
    field: str
    meaning: str

    @property
    def path(self) -> str:
        return f"{self.node}.{self.field}"


@dataclass(frozen=True)
class Layout:
    """A built experiment's outputs and flags: its name, its base step (s), its observations, the flags its entries
    raise, and the stamp of its profile, which every printout and export carries (``None`` for a run folder written
    before profiles)."""

    experiment: str
    dt: float
    observations: tuple[ObservationLayout, ...]
    flags: tuple[FlagLayout, ...]
    stamp: Stamp | None = None

    def observation(self, id: str) -> ObservationLayout | None:
        return next((held for held in self.observations if held.id == id), None)

    def output(self, path: str) -> OutputLayout | None:
        """The output ``<observation id>.<output>``."""
        observation, _, name = path.rpartition(".")
        held = self.observation(observation)
        return held.output(name) if held is not None else None

    def per_step(self) -> list[OutputLayout]:
        return [output for held in self.observations for output in held.outputs if output.kind is OutputKind.PER_STEP]


#: A window's ends lie on a step's end when they are within this much of it, relative: a time written as a multiple of
#: the base step is that step's end, whatever its last bit.
ON_THE_GRID = 1e-9


def position(time: float, dt: float) -> float:
    """``time`` in base steps: the step count whose end it is, snapped to it when it lies on the grid."""
    ratio = time / dt
    nearest = round(ratio)
    return float(nearest) if math.isclose(ratio, nearest, rel_tol=ON_THE_GRID, abs_tol=ON_THE_GRID) else ratio


def ended_after(steps: NDArray[np.int64], start: float, dt: float, clock: np.dtype[Any]) -> NDArray[np.bool_]:
    """Which of ``steps`` (step counts) end after ``start``, decided as a node decides it for its ``after``: the step's
    end is its count times the base step, and both it and ``start`` are rounded to the run's float type (``clock``)
    before they are compared. In float32 a ``start`` on the grid can therefore fall either side of the step that ends
    there, as the node's own test does, and a series, a reading and a node agree on which."""
    kind = clock.type
    return steps.astype(kind) * kind(dt) > kind(start)


__all__ = ["ON_THE_GRID", "FlagLayout", "Layout", "ObservationLayout", "OutputLayout", "ended_after", "position"]
