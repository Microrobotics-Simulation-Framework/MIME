"""What ``mime run`` says when it ends, on a terminal: what needs attention first, then a short statement of what ran, then
the readings, and the rest behind ``--full``. Plain lines only: ``mime.cli._display`` styles them.

Nothing is dropped from the files, and a pipe still gets the whole of it: this is only how a person at a terminal is
shown a run, by the same objects ``--full`` and the files hold in full.

**What it can and cannot say.** A validity line is free text with no machine-readable "at its limit" status, so the
lines it lists are the ones the build *computed* (``validity of <owner> (computed at build)``), which a build gives when
a model is at or past what it covers; the declared ones (assumptions every run carries, "not yet checked") are in the
build report, behind ``--full``, not in the lead. A post-run watch is silent only when it is within its range: one that
is raised, or that could not be checked (no range declared, an input missing), is listed, since "could not check" is not
"fine". A flag a run kept no
per-step record of is read from what the run reduced of it as it ran (how many steps raised it, the first and the last:
``mime.results.FlagWhole``), and is unknown between steps only where the run reduced none: it is said to be, not read as
"not raised". MIME has no structured record
of a fallback, so none is reported as one."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mime.cli._progress import duration_text, simulated_text
from mime.cli._shell import quoted
from mime.results._containment import WITHIN as WITHIN_THE_REGION
from mime.results._watch import ABOVE, WITHIN

if TYPE_CHECKING:
    from mime.experiment import BuiltExperiment, RunWritten
    from mime.saved import LoadReport

#: How many reading lines the default summary shows before pointing at ``--full``.
READINGS_SHOWN = 12


@dataclass(frozen=True)
class RunSummary:
    """The terminal's account of a finished run, section by section, each a list of plain lines."""

    attention: tuple[str, ...]
    ran: tuple[str, ...]
    readings: tuple[str, ...]
    wrote: tuple[str, ...]
    next_steps: tuple[str, ...]
    more: str


def attention_lines(built: BuiltExperiment, written: RunWritten, load: LoadReport | None, folder: Path) -> list[str]:
    """Everything the run or the build says that a person should look at before trusting the numbers."""
    result, report = written.result, built.report
    lines: list[str] = []
    if written.stopped is not None:
        lines.append(f"STOPPED: {written.stopped.reason}; the run folder holds the first {written.steps} of the "
                     f"{written.stopped.steps_asked} steps asked for")
    lines += report.range_warning_lines()
    raised = [read for read in result.flags if read.was_raised]
    lines += [f"flag raised: {read.text(meaning=True)}" for read in raised]
    unknown = [read for read in result.flags if read.was_raised is None]
    if unknown:
        lines.append(f"{len(unknown)} flag{'s' if len(unknown) != 1 else ''} not known between steps (the run kept "
                     "no per-step record; --history or --record keeps every step): "
                     + ", ".join(read.path for read in unknown))
    for watch in result.watches:
        if watch.status == ABOVE:
            lines.append(f"post-run watch raised: {watch.text()}")
        elif watch.status != WITHIN:  # no_range, not_computed, and any status added later: not passed, so not "fine"
            lines.append(f"post-run watch could not be checked: {watch.text()}")
    for contained in result.containment:
        if contained.raised:
            lines.append(f"post-run containment raised: {contained.text()}")
        elif contained.status != WITHIN_THE_REGION:
            lines.append(f"post-run containment could not be checked: {contained.text()}")
    checked = [(owner, line) for owner, held in sorted(report.checked.items()) for line in held]
    lines += [f"validity of {owner}, computed at build: {line}" for owner, line in checked]
    lines += [f"build refused: {line}" for refusal in report.refusals for line in str(refusal).splitlines()]
    if report.warnings:
        lines.append(f"MADDENING gave {len(report.warnings)} warning{'s' if len(report.warnings) != 1 else ''} "
                     "while compiling (--full shows them)")
    profile = report.profile
    if profile is not None and profile.pending:
        lines.append(f"profile {profile.profile.value}: {len(profile.pending)} item{'s' if len(profile.pending) != 1 else ''}"
                     f" pending, not sealed; `mime pending {quoted(str(folder))}` lists them with the lines "
                     "to paste")
    if load is not None:
        if not load.same_environment:
            lines.append("environment: built under another than the lock's; results are not guaranteed to match "
                         f"(`mime verify {quoted(str(folder))}` checks)")
        if load.derived_differences:
            lines.append("derived graph differs from the saved one: " + "; ".join(load.derived_differences))
        lines += list(load.notes)
        if load.derived_prims_edited:
            lines.append("stage: derived prims edited after it was written (reported, not refused)")
    return lines


def what_ran(built: BuiltExperiment, written: RunWritten, started: float, ended: float) -> list[str]:
    """A short statement of what ran: the experiment, its steps and time, and where the time went."""
    result = written.result
    seconds = written.record["seconds"]
    spent = [f"{label} {duration_text(float(seconds[key]))}"
             for key, label in (("load_and_build", "loading and building"), ("run", "running, compiling included"),
                                ("save", "writing"))
             if seconds.get(key) is not None]
    first = (f"ran {result.layout.experiment}: {written.steps:,} steps of {built.plans.dt!r} s, "
             f"{simulated_text(result.end_step * built.plans.dt)} of simulated time")
    return [first, f"took {duration_text(ended - started)} in all: " + ", ".join(spent)]


def reading_lines(written: RunWritten) -> list[str]:
    """The observations' readings, as the result prints them, without the lines about flags, watches and the profile
    (those are in the attention section when they matter)."""
    result = written.result
    return [line for each in result for line in result[each].text().splitlines()]


def wrote_lines(written: RunWritten, out: Path) -> list[str]:
    files = [written.outputs.name, written.manifest.name, written.record_file.name, "report.json", "topology.json",
             f"{written.experiment.name}/ (the run's saved form)"]
    if written.recording is not None:
        files.append(written.recording.name)
    return [f"wrote {out}{'/' if not str(out).endswith(('/', chr(92))) else ''}: " + ", ".join(files)]


def next_steps(folder: Path, out: Path) -> list[str]:
    """The exact commands for this folder and this run, on a terminal only."""
    return [f"read the result:  python -c \"from mime.results import RunResult; import sys; "
            f"print(RunResult.read(sys.argv[1]).text())\" {quoted(str(out))}",
            f"without MIME:     {out / 'outputs.json'} (every output with its unit), {out / 'report.json'} (the whole "
            "report)",
            f"check it again:   mime verify {quoted(str(out / 'experiment'))}",
            f"watch it live:    mime serve {quoted(str(folder))} --start"]


def more_line(out: Path) -> str:
    return (f"--full prints the load report, the build report's summary and every reading; the whole report is in "
            f"{out / 'report.json'}")
