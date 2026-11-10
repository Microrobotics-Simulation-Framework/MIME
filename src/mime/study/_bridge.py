"""The bridge from a study to an experiment: ``result.member(i).as_experiment(name)``. A member is the study's base
experiment with its point's values declared (and, for a study run on a ramp, its ramp declared from the point's value,
as the study declared it); this makes that an experiment of its own, through ``Experiment.based_on(<the base>,
name=name)`` and the declarations a person would make, so that the lineage is never lost: the new experiment records
the base it was made from, each value declared carries the reason ``from study <id>, member <i>``, and each declaration
it replaced is in ``plan().replaced``. A member that lay outside a declared range of the base was built for its own value,
without that range, and its experiment is made so: the range is removed (``plan().removed`` says so)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mime.build import BuildRefused, Refusal, RefusalClass
from mime.results._study_folder import BASE_DIR
from mime.saved._folder import LOCK_FILE
from mime.study._axes import STUDY_REASON, AxisKind, declared, resolve
from mime.study._protocol import Ramp, declared_with

if TYPE_CHECKING:
    from mime.experiment import Experiment
    from mime.results._result import RunResult


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> BuildRefused:
    return BuildRefused([Refusal(code, category, message, action)])


def member_as_experiment(run: RunResult, name: str, base: Path | None) -> Experiment:
    """The member ``run`` as an experiment named ``name``: see :meth:`RunResult.as_experiment`."""
    from mime.experiment import Experiment

    member = run.member_of
    if member is None:
        raise _refuse("study.not-a-member", RefusalClass.USER,
                      "this run is not a member of a study, so there is no study to make an "
                                            "experiment from",
                      "take it from a study's result: Study(exp).sweep(...).run_to(folder).member(i).as_experiment(name)")
    folder = base if base is not None else member.base
    if folder is None or not (folder / LOCK_FILE).is_file():
        raise _refuse("study.no-base", RefusalClass.USER,
                      "this member's study has no saved base experiment to make a new one from"
                      + (f" ({folder} holds none)" if folder is not None else ""),
                      f"run the study into a folder, `study.run_to(<folder>)`, which saves its base as {BASE_DIR}/; or "
                      f"name the base: member.as_experiment(name, base=<the saved experiment>)")
    experiment = Experiment.based_on(folder, name=name)
    settings, declarations, _ = experiment.declared_as()
    axes = {resolve(path, declarations): value for path, value in member.values}
    if any(axis.kind is AxisKind.PRECISION for axis in axes):
        raise _refuse("study.bridge-precision", RefusalClass.USER,
                      "the member ran in a precision the study varied, and a precision is a setting of the process, "
                      "not a declaration of an experiment",
                      "start the process in that precision (JAX_ENABLE_X64=1 for float64) and make the experiment there")
    why = f"from study {member.study}, member {member.member}"
    reasons = {axis.path: f"{why}: {member.reasons[axis.path]}" for axis in axes
               if member.reasons.get(axis.path, STUDY_REASON) != STUDY_REASON}
    held = declared(settings, declarations, axes, why, reasons)
    ramped = held.declarations
    if isinstance(member.protocol, Ramp):
        ramped, _ = declared_with(member.protocol, held.declarations)
    experiment.redeclared(held.settings, ramped)
    for path in held.outside:  # a member outside a declared range was built for its own value, without it
        experiment.remove(path, kind="range")
    return experiment


__all__ = ["member_as_experiment"]
