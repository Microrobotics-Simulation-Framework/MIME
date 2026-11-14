"""Ports: what an apparatus's link to the world is, declared as data on the apparatus (``Apparatus(ports={...})``).

A **command port** (:class:`CommandPort`), keyed by the command it carries (``"currents"``, ``"arm.joint_positions"``),
is the link from whatever writes the command (its drive, the controller) to what takes it: the writer sampled at the
port's ``rate`` and held between, delayed by its ``latency``, held within its ``bounds`` (clipped, or tripped to ``rest``
for the rest of the run), and limited in how fast it moves (``slew``). Its last rung, a ``response`` that shapes what is
delivered (a coil's L/R lag, a servo), is refused until it is built, in MIME 0.3.0.dev2. A **measurement port** (:class:`Measurement`), keyed by a name of its own, is the link from what
it measures (``of``: a thing's state, ``"umr.pose"``, or an observation's output, ``"camera:.expected"``) to whatever
reads it: sampled at its ``rate``, delayed by its ``latency``, with seeded ``noise`` and a ``quantum``. A controller
reads one as ``"<apparatus>.<port>"``, and its age in base steps as ``"<apparatus>.<port>.age"``.

A command or a read with no port is ideal: it reaches its receiver at once, unchanged, and the build adds nothing for
it. A port whose every field is its default is ideal too. Every number a port holds is declared, from a datasheet or a
measurement named in its ``reason``, never fitted. ``drift``, ``fault``, ``bias`` and ``dropout`` are named and
refused: they are seams, built in MIME 0.3.0a1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from mime.units import Quantity, Unit

#: What a port holds between its updates: the one kind built (the last sample, held).
ZERO_ORDER = "zero_order"
HOLDS = (ZERO_ORDER,)
#: What a command port does with a value beyond its bounds: deliver the bound, or latch to ``rest`` for the rest of
#: the run.
CLIP, TRIP = "clip", "trip"
ON_BOUND = (CLIP, TRIP)
#: Where the fields named as seams and refused are built, and where a command port's response is.
SEAMS_BUILT_IN = "MIME 0.3.0a1"
RESPONSE_BUILT_IN = "MIME 0.3.0.dev2, with the saved form that holds ports"


@dataclass(frozen=True, kw_only=True)
class Gaussian:
    """Additive Gaussian noise of standard deviation ``sigma``, in the units of what the port measures, drawn afresh at
    each sample from the run's seed and the sample's step: rendering the run again reproduces it."""

    sigma: Quantity


@dataclass(frozen=True, kw_only=True)
class CommandPort:
    """The link a command takes from its writer to what receives it. ``unit`` and ``shape`` restate the command's, and
    are checked equal to them. ``bounds`` is ``(low, high)``, each a quantity of the command's dimension of one value or
    one per element, or one quantity ``q`` for ``(-q, q)``; ``on_bound`` is ``"clip"`` (deliver the bound) or ``"trip"``
    (deliver ``rest`` from that step on, flagged). ``rest`` is what is delivered before the first write arrives and
    after a trip: the receiver's initial value where its entry states one (a kinematic arm's initial positions), else
    zero. ``slew`` is the most the delivered value moves per second (``A(4000.0) / s(1.0)``). ``rate`` is the link's
    update period (or frequency), ``latency`` its delay, each a whole number of base steps. ``response``, what shapes the
    value delivered last (a coil's L/R lag, a servo), is refused until it is built (``port.seam``). ``reason`` says where
    the numbers come from, and is required wherever one is given."""

    unit: Unit | None = None
    shape: tuple[int, ...] | None = None
    bounds: Quantity | tuple[Quantity, Quantity] | None = None
    on_bound: str = CLIP
    rest: Quantity | None = None
    slew: Quantity | None = None
    rate: Quantity | None = None
    latency: Quantity | None = None
    hold: str = ZERO_ORDER
    response: object = None
    reason: str | None = None
    drift: object = None
    fault: object = None

    #: The fields named as seams, refused.
    seams: ClassVar[tuple[str, ...]] = ("drift", "fault")
    #: The fields that hold a number, which a reason must come with.
    numbers: ClassVar[tuple[str, ...]] = ("bounds", "rest", "slew", "rate", "latency")


@dataclass(frozen=True, kw_only=True)
class Measurement:
    """The link a measured signal takes from what it measures, ``of`` (a thing's state, ``"umr.pose"``, a chain
    segment's, ``"rpm.arm.joint_positions"``, or an observation's output, ``"camera:.expected"``), to whatever reads it.
    ``unit`` and ``shape`` restate the source's, and are checked equal to them. ``rate`` is the sampling period (or
    frequency), a whole number of base steps, and for a source with a clock of its own (a camera) a whole multiple of
    its period, which decimates it; ``latency`` the delay from a sample to its being read. ``noise`` (:class:`Gaussian`)
    and ``quantum`` (a step the value is rounded to, in the source's units) are added after the delay, the noise first.
    ``reason`` says where the numbers come from, and is required wherever one is given."""

    of: str
    unit: Unit | None = None
    shape: tuple[int, ...] | None = None
    rate: Quantity | None = None
    latency: Quantity | None = None
    noise: Gaussian | None = None
    quantum: Quantity | None = None
    hold: str = ZERO_ORDER
    reason: str | None = None
    bias: object = None
    drift: object = None
    dropout: object = None

    seams: ClassVar[tuple[str, ...]] = ("bias", "drift", "dropout")
    numbers: ClassVar[tuple[str, ...]] = ("rate", "latency", "noise", "quantum")


type Port = CommandPort | Measurement


__all__ = ["CLIP", "HOLDS", "ON_BOUND", "RESPONSE_BUILT_IN", "SEAMS_BUILT_IN", "TRIP", "ZERO_ORDER", "CommandPort",
           "Gaussian", "Measurement", "Port"]
