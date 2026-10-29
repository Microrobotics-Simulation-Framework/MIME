"""The controller as phase 1 resolves it: its mode, its period and latency in base steps, each read with its class
and each write with the command it replaces the drive of. What the run does with it is ``mime.control``'s."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from mime.registry import Controller, Registered
from mime.units import Magnitude


class Mode(Enum):
    """How a controller runs. A controller is declared once, and the build makes each mode's binding from that one
    declaration, so swapping modes changes no physics."""

    GRAPH = 1
    CALLBACK = 2
    CLIENT = 3
    FMU = 4

    @property
    def word(self) -> str:
        return self.name.lower()

    @property
    def meaning(self) -> str:
        return {Mode.GRAPH: "inside the compiled step, as a node at its own rate",
                Mode.CALLBACK: "a callback on the host, between compiled chunks",
                Mode.CLIENT: "an external client that steps the simulation",
                Mode.FMU: "the experiment exported as an FMU, which another tool steps"}[self]


#: The local name of a controller's delay line, beside its node inside the compiled step (``controller.delay``).
DELAY = "delay"


def controller_nodes(mode: Mode | None, latency: int) -> tuple[str, ...]:
    """The local names of the nodes a controller builds: none as a callback; inside the compiled step its node, and
    a delay line when its latency is a step or more."""
    if mode is not Mode.GRAPH:
        return ()
    return ("", DELAY) if latency else ("",)


#: Where each mode not built yet is built.
UNBUILT = {Mode.CLIENT: "MIME 0.3.0a2", Mode.FMU: "MIME 0.3.0a2"}


def mode_of(value: object) -> Mode | None:
    """The mode ``value`` names: its word (``"callback"``) or its number (``2``)."""
    if isinstance(value, Mode):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return next((mode for mode in Mode if mode.value == value), None)
    if isinstance(value, str):
        return next((mode for mode in Mode if mode.word == value), None)
    return None


class ReadClass(Enum):
    """What a controller's read is: the physics state itself, or an observation's output. A controller that reads the
    state reads what no instrument gives a real one."""

    STATE = "state"
    OBSERVATION = "observation"

    @property
    def meaning(self) -> str:
        if self is ReadClass.STATE:
            return "ground truth, not deployable"
        return "an observation's output, as deployable as the observation's own model of its instrument"


@dataclass(frozen=True)
class ClassedRead:
    """One read label as bound: the path the experiment wrote, the node and state field the run reads it from, its
    class, what it is (an interface of a thing, or an observation's output), its units and shape."""

    label: str
    path: str
    node: str
    field: str
    read_class: ReadClass
    source: str
    units: str
    shape: tuple[int, ...]

    def text(self) -> str:
        return f"{self.label} <- {self.path} ({self.source}): {self.read_class.value}, {self.read_class.meaning}"


@dataclass(frozen=True)
class ResolvedWrite:
    """One write label as bound: the command path the experiment wrote (``rpm.arm.joint_positions``), the node input
    it feeds as an external input of the graph, in place of the command's drive, with its units and shape."""

    label: str
    path: str
    node: str
    field: str
    units: str
    shape: tuple[int, ...]

    def text(self) -> str:
        return f"{self.label} -> {self.path} (an external input of {self.node}.{self.field}, shape {self.shape})"


#: The rule a declared latency follows, which every mode keeps.
LATENCY_RULE = ("a tick at time t reads the state as it stands at t, before the step that starts there; its writes "
                "act on every step from the one that starts at t + latency until the next tick's writes do, and the "
                "writes it holds before its first act on the steps before them")


@dataclass(frozen=True)
class ControllerPlan:
    """The controller as phase 1 resolved it: ``owner`` its id, its entry and registration, its settings in SI, its
    mode, its period (every ``period`` base steps) and ``latency`` in base steps, and its reads and writes in label
    order."""

    owner: str
    entry: Controller
    registered: Registered
    settings: Mapping[str, Magnitude]
    mode: Mode
    period: int
    latency: int
    reads: tuple[ClassedRead, ...]
    writes: tuple[ResolvedWrite, ...]

    def inputs(self) -> tuple[str, ...]:
        """The external inputs its writes derive, as ``node.field``."""
        return tuple(f"{w.node}.{w.field}" for w in self.writes)

    def lines(self, dt: float) -> list[str]:
        latency = f"latency {self.latency} step{'s' if self.latency != 1 else ''} ({self.latency * dt!r} s)"
        lines = [f"controller: {self.registered.cls.label()}, mode {self.mode.value} ({self.mode.meaning}), every "
                 f"{self.period} step{'s' if self.period != 1 else ''} ({self.period * dt!r} s), {latency}: "
                 f"{LATENCY_RULE}",
                 "  reads, each with its class:"]
        lines += [f"    {read.text()}" for read in self.reads] or ["    none"]
        lines.append("  writes, each replacing its command's drive:")
        lines += [f"    {write.text()}" for write in self.writes]
        lines.append(f"  gradients: {self.gradients()}")
        return lines

    def gradients(self) -> str:
        """How a derivative of the run passes through the controller in its mode."""
        status = self.registered.cls.differentiability.value
        if self.mode is Mode.GRAPH:
            return (f"its step runs inside the compiled step, so a derivative of the run passes through its writes "
                    f"as through any node's: its law is {status}, from the switches it declares, and a path through "
                    f"it is no better than that")
        return (f"none pass through it in mode {self.mode.value}: it runs on the host between compiled chunks, so a "
                f"derivative of the run stops at its writes (its law is {status} inside the compiled step, mode "
                f"{Mode.GRAPH.value})")


def refused_reads(profile: str | None, reads: Sequence[ClassedRead]) -> list[tuple[ClassedRead, str]]:
    """The reads a profile refuses, each with its reason. The seam where a profile judges a controller's reads by
    their class: no profile exists yet, so none is refused here, and the build report names every state read ground
    truth, not deployable. A profile that holds a controller to what an instrument could give (``strict``) is to
    refuse each read of :attr:`ReadClass.STATE` here."""
    return []


__all__ = ["DELAY", "LATENCY_RULE", "UNBUILT", "ClassedRead", "ControllerPlan", "Mode", "ReadClass", "ResolvedWrite",
           "controller_nodes", "mode_of", "refused_reads"]
