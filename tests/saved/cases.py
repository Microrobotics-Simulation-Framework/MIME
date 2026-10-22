"""The round-trip cases the registry walk requires: together they use every registered entry, MIME's toys and the
lab fixture's, and every kind of thing. An entry registered without a case here fails the walk.

A case whose build is refused (an entry that exists to be refused) round-trips its document instead: written, read
back in a fresh process, and refused there for the same reasons.
"""

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass

from mime.experiment import Experiment
from mime.testing.toys import (
    ToyAnchoredBody,
    ToySpring,
    ToyTwist,
    coupled_toy,
    full_toy,
    toy_experiment,
)
from mime.things import Body, State, Structure
from mime.units import N, m


@dataclass(frozen=True)
class Case:
    name: str
    make: Callable[[], Experiment]
    refused: tuple[str, ...] = ()


def walled_toy() -> Experiment:
    """The coupled toy beside a passive structure, which builds no node and saves as a thing."""
    experiment = coupled_toy()
    experiment.add_structure(Structure("wall"))
    return experiment


def anchored_toy() -> Experiment:
    experiment = toy_experiment("anchored_toy")
    experiment.add_body(Body("anchor", dynamics=ToyAnchoredBody(), initial=State(position=m((0.1, 0.0, -0.2)))))
    experiment.attach(ToySpring(k=N(2.3) / m(1.0)), body="anchor")
    return experiment


def twisted_toy() -> Experiment:
    experiment = coupled_toy()
    experiment.attach(ToyTwist(), body="probe")
    return experiment


def nudged_toy() -> Experiment:
    """The lab fixture's entry beside MIME's toys: its lock names its own distribution."""
    nudge = importlib.import_module("mime_lab_fixture").Nudge
    experiment = coupled_toy()
    experiment.attach(nudge(force=N((0.0, 0.01, -0.04))), body="probe")
    return experiment


CASES = (
    Case("coupled_toy", coupled_toy),
    Case("full_toy", full_toy),
    Case("walled_toy", walled_toy),
    Case("anchored_toy", anchored_toy),
    Case("twisted_toy", twisted_toy, refused=("vocabulary.reserved",)),
    Case("nudged_toy", nudged_toy),
)

if os.environ.get("MIME_CI_PLANT_FAILURE") == "1":
    # The control that the walk fails CI when an entry has no case: the planted run drops the only case using
    # toy.anchored_body.
    CASES = tuple(case for case in CASES if case.name != "anchored_toy")
