"""The small live rig the live session's tests run: ``control_cases.spun_magnet``'s motor alone above a body, its angle
written every 3 ms by :class:`Steered`, a callback with one live input, ``offset``, added to every angle it writes once
a session sets it. Cheap to build and to compile, so that a test can run it live, in batch and again, and compare."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from control_cases import spun_magnet

from mime.experiment import Experiment
from mime.gradients import GradientStatus
from mime.registry import ControlLaw, Controller, GradientRecord, LiveInput, Signal, register_controller
from mime.units import ms


@register_controller
@dataclass(frozen=True, kw_only=True)
class Steered(Controller):
    """Writes as the motor's angle ``rate`` times its tick count, plus its live input ``offset`` once a session sets
    it: what a live input moves is the angle, so the motor's state shows it the step its write acts."""

    name = "oracle.steered"
    version = "0.1.0"
    differentiability = GradientStatus.CUT
    gradient = GradientRecord(where="its step", reason="a test's callback", route="none: it exists to be called")
    builds = {"": ()}
    reads = {"pose": Signal("m; 1", (7,))}
    writes = {"angle": Signal("1", ())}
    live = {"offset": LiveInput("1", (), "added to every angle it writes")}
    rate: float = 0.1

    def law(self, ctx: Any) -> ControlLaw:
        rate = self.rate

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            ticks = int(state["ticks"])
            angle = rate * ticks + (float(reads["offset"]) if "offset" in reads else 0.0)
            return {"ticks": ticks + 1}, {"angle": np.asarray(angle)}

        return ControlLaw(state={"ticks": 0}, held={"angle": np.asarray(0.0)}, step=step)


def steered(name: str = "steered", **options: Any) -> Experiment:
    """The live rig: :class:`Steered` every 3 ms writing the motor's angle, the body's swim speed observed."""
    return spun_magnet(name, controller=Steered(), mode="callback", rate=ms(3), observe=True, **options)


def in_graph(name: str = "steered_in_graph") -> Experiment:
    """The rig's motor written by a controller inside the compiled step, which no live input reaches."""
    from control_cases import Counter

    return spun_magnet(name, controller=Counter(), mode="graph", rate=ms(3))


def saved(tmp_path: Path, name: str = "steered") -> Path:
    folder = tmp_path / name
    steered(name).save(folder)
    return folder


def files_of(folder: Path) -> dict[str, bytes]:
    """Every file below ``folder`` by its path there, as bytes."""
    return {str(path.relative_to(folder)): path.read_bytes() for path in sorted(folder.rglob("*")) if path.is_file()}


__all__ = ["Steered", "dataclasses", "files_of", "in_graph", "saved", "steered"]
