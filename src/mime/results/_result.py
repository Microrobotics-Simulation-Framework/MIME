"""One ``RunResult``, from a run and from a run's folder, with the same accessors: indexed by the handle ``observe``
returned or by the observation's id, each output read by its name, with units, windows by time and the flags raised
over them; and its exports, a table, CSV with the units in its header, NPZ, and the manifest a run folder holds beside
its arrays (``outputs.json``), from which a folder is readable without MIME."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Protocol, cast, override

import numpy as np
from numpy.typing import NDArray

from mime.build._refusals import ArgumentRefused, ArgumentTypeRefused, MimeError, Refusal, RefusalClass
from mime.errors import public
from mime.registry import OutputKind, Reduction
from mime.results._layout import FlagLayout, Layout, ObservationLayout, OutputLayout
from mime.results._watch import WatchRead, watches_text
from mime.results._series import (
    FlagRead,
    FlagSource,
    FlagTrack,
    RECORDED_ONLY,
    NotInResult,
    NotRecorded,
    Reading,
    Series,
    Table,
    flags_text,
    quantity_of,
    series_table,
    window_text,
)
from mime.things._handles import did_you_mean
from mime.units import TIME, Quantity, parse_unit
from mime.validity import Stamp

type Arrays = Mapping[str, Mapping[str, NDArray[Any]]]

#: The manifest a run folder holds beside ``outputs.npz``, and the format it is written in.
RUN_MANIFEST = "outputs.json"
MANIFEST_FORMAT = "mime-run-outputs"
#: Version 2 holds the profile's stamp (``stamp``); a manifest of version 1, written before profiles, is read with none.
MANIFEST_VERSION = 2
READ_VERSIONS = (1, 2)
#: The manifest's key inside an NPZ written by :meth:`RunResult.to_npz`.
NPZ_MANIFEST = "manifest"
#: The build report's summary a run folder holds beside ``run.json`` (``BuiltExperiment.report_summary()``), and the
#: format it is written in: the experiment's validity lines, the flags a run may raise and what each means, each
#: setting's role and default, and the profile.
RUN_REPORT = "report.json"
REPORT_FORMAT = "mime.report/1"



class NoOutput(MimeError, AttributeError):
    """An output an observation's outputs do not hold, read as an attribute: an ``AttributeError``, as it was."""


class NotARunFolder(MimeError, FileNotFoundError):
    """A path :meth:`RunResult.read` cannot read a run from: a ``FileNotFoundError``, as it was."""

class _Identified(Protocol):
    @property
    def id(self) -> str: ...


@dataclass(frozen=True)
class OutputRef:
    """An observation's output by name, as ``record=`` takes it: what an observation's handle gives for one of its
    outputs (``swim.speed``)."""

    observation: str
    name: str

    @property
    def path(self) -> str:
        return f"{self.observation}.{self.name}"


@dataclass(frozen=True)
class Samples:
    """A recorded series' samples: the step count each was taken at, and the values there."""

    steps: NDArray[np.int64]
    values: NDArray[Any]


def series_key(path: str) -> str:
    return f"series/{path}"


def flag_key(path: str) -> str:
    return f"flags/{path}"


class _Npz:
    """An NPZ file read lazily, each array on first use and kept: a large run folder is opened without reading what
    is not asked for."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with np.load(path, allow_pickle=False) as held:
            self.files = list(held.files)
        self._held: dict[str, NDArray[Any]] = {}

    def __getitem__(self, key: str) -> NDArray[Any]:
        if key not in self._held:
            with np.load(self.path, allow_pickle=False) as held:
                self._held[key] = np.asarray(held[key])
        return self._held[key]

    def many(self, keys: Sequence[str]) -> dict[str, NDArray[Any]]:
        missing = [key for key in keys if key not in self._held]
        if missing:
            with np.load(self.path, allow_pickle=False) as held:
                self._held.update({key: np.asarray(held[key]) for key in missing})
        return {key: self._held[key] for key in keys}


class _Fields(Mapping[str, NDArray[Any]]):
    def __init__(self, npz: _Npz, prefix: str, names: Sequence[str]) -> None:
        self._npz, self._prefix, self._names = npz, prefix, list(names)

    @override
    def __getitem__(self, key: str) -> NDArray[Any]:
        if key not in self._names:
            raise KeyError(key)
        return self._npz[f"{self._prefix}/{key}"]

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    @override
    def __len__(self) -> int:
        return len(self._names)


class _Nodes(Mapping[str, Mapping[str, NDArray[Any]]]):
    """The arrays of an NPZ under ``<prefix>/<node>/<field>``, by node and field, each read when first used."""

    def __init__(self, npz: _Npz, prefix: str) -> None:
        self._npz, self._prefix = npz, prefix
        found: dict[str, list[str]] = {}
        for key in npz.files:
            if key.startswith(f"{prefix}/"):
                node, _, field = key.removeprefix(f"{prefix}/").rpartition("/")
                found.setdefault(node, []).append(field)
        self._fields = {node: _Fields(npz, f"{prefix}/{node}", names) for node, names in found.items()}

    @override
    def __getitem__(self, key: str) -> Mapping[str, NDArray[Any]]:
        return self._fields[key]

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    @override
    def __len__(self) -> int:
        return len(self._fields)


class ObservationOutputs:
    """One observation's outputs in a run's result, each read by its name, as an attribute (``result[swim].speed``)
    or by key (``result[swim]["speed"]``): a per-step output is a :class:`Series` when the run recorded it and is
    refused by name when it did not; an output at the end is a :class:`Reading`, always there."""

    def __init__(self, result: RunResult, layout: ObservationLayout) -> None:
        self._result = result
        self.layout = layout

    @property
    def id(self) -> str:
        return self.layout.id

    @property
    def outputs(self) -> tuple[str, ...]:
        return tuple(output.name for output in self.layout.outputs)

    def __contains__(self, name: object) -> bool:
        """Whether the observation has an output of that name, recorded or not: ``"speed" in result[swim]``."""
        return isinstance(name, str) and self.layout.output(name) is not None

    def __iter__(self) -> Iterator[str]:
        """The observation's outputs, by name."""
        return iter(self.outputs)

    @public
    def __getitem__(self, name: str) -> Series | Reading:
        if not isinstance(cast(object, name), str):
            raise ArgumentTypeRefused([Refusal(
                "call.wrong-type", RefusalClass.USER,
                f"an observation's output is read by its name, not by {type(name).__name__} {name!r}",
                f"read it by name: result[{self.id!r}][{self.outputs[0] if self.outputs else 'speed'!r}], or as "
                f"an attribute; its outputs are {', '.join(self.outputs) or 'none'}")])
        output = self.layout.output(name)
        if output is None:
            raise NotInResult([Refusal(
                "result.no-output", RefusalClass.USER,
                f"the observation {self.id} ({self.layout.entry}) has no output {name!r}",
                f"read one of its outputs, result[{self.id!r}].<output>: {', '.join(self.outputs)}. "
                f"{did_you_mean(name, self.outputs)}")])
        return self._result.output(output)

    def __getattr__(self, name: str) -> Series | Reading:
        if name.startswith("_") or name == "layout":
            raise AttributeError(name)
        try:
            return self[name]
        except NotInResult as error:
            if error.code != "result.no-output":
                raise
            raise NoOutput(error.refusals) from None

    @public
    def text(self) -> str:
        lines = [f"{self.id} ({self.layout.entry}):"]
        for output in self.layout.outputs:
            if output.kind is OutputKind.AT_END:
                lines.append(f"  {self._result.output(output)}")
            elif output.path in self._result.recorded:
                lines.append(f"  {self._result.output(output)}")
            else:
                lines.append(f"  {output.path}: per step, in {output.unit}, not recorded (record=[{output.path!r}])")
        return "\n".join(lines)

    @override
    def __str__(self) -> str:
        return self.text()


class RunResult:
    """A run's result. ``steps`` is the steps it ran, ``final`` every node's fields after its last step and
    ``history``, when the run was asked for it, every field at every step, as before. Above them, by observation
    (``result[swim]`` or ``result["swim_speed:umr"]``), each declared output by name, with its units: an output at the
    end as a :class:`Reading` over its window, a recorded per-step output as a :class:`Series` with its time axis.
    Every reading, series and export carries the flags its window's nodes raised.

    The same object comes from :meth:`BuiltExperiment.run` and from :meth:`read`, of a folder ``run_to`` or ``mime run``
    wrote (on a local disk or a Modal Volume's), or of an NPZ :meth:`to_npz` wrote."""

    def __init__(self, steps: int, final: Arrays, history: Arrays | None = None, *, layout: Layout,
                 start_step: int = 0, recorded: Mapping[str, Samples] | None = None,
                 tracks: Mapping[str, FlagTrack] | None = None, every: int | None = None,
                 loader: Mapping[str, str] | None = None, npz: _Npz | None = None,
                 clock: str | None = None, watches: Sequence[WatchRead] = (),
                 report: Mapping[str, Any] | None = None) -> None:
        self.steps = steps
        self.final = final
        self.history = history
        self.layout = layout
        self.start_step = start_step
        self.every = every
        self._recorded: dict[str, Samples] = dict(recorded or {})
        self._loader = dict(loader or {})
        self._npz = npz
        self.tracks: Mapping[str, FlagTrack] = dict(tracks or {})
        #: The post-run watches (:class:`~mime.results.WatchRead`), read from the run's end state and, where the run
        #: kept it, its history: a flag on the result, never a refusal.
        self.watches: tuple[WatchRead, ...] = tuple(watches)
        #: The build report's summary (:data:`REPORT_FORMAT`): the built experiment's for a run in memory, a run
        #: folder's ``report.json`` for one read back, ``None`` for a folder written before run folders held it.
        self.report: Mapping[str, Any] | None = report
        self._clock = np.dtype(clock) if clock is not None else None

    @property
    def clock(self) -> np.dtype[Any]:
        """The float type the run's nodes computed in, which decides a window's start (as a node decides its
        ``after``): the type of the run's first floating-point field."""
        if self._clock is None:
            kinds = (np.asarray(value).dtype for node in sorted(self.final) for value in self.final[node].values())
            self._clock = next((kind for kind in kinds if kind.kind == "f"), np.dtype(np.float64))
        return self._clock

    @property
    def dt(self) -> float:
        return self.layout.dt

    @property
    def end_step(self) -> int:
        return self.start_step + self.steps

    @property
    def start(self) -> Quantity:
        """The time the run started at, from the graph's start."""
        return Quantity(self.start_step * self.dt, TIME)

    @property
    def end(self) -> Quantity:
        return Quantity(self.end_step * self.dt, TIME)

    @property
    def time(self) -> Quantity:
        """The end of each of the run's steps, from the graph's start: the axis of ``history``'s rows."""
        return Quantity.of(np.arange(self.start_step + 1, self.end_step + 1, dtype=np.float64) * self.dt, TIME)

    @property
    def recorded(self) -> tuple[str, ...]:
        """The per-step outputs the run recorded, by path."""
        return tuple(sorted({*self._recorded, *self._loader}))

    @property
    def observations(self) -> tuple[str, ...]:
        return tuple(held.id for held in self.layout.observations)

    def _flag_source(self) -> FlagSource:
        return FlagSource(self.layout.flags, self.tracks, self.start_step, self.end_step, self.dt, self.clock)

    @property
    def flags(self) -> tuple[FlagRead, ...]:
        """The flags raised over the run, (start, end]."""
        return self._flag_source().over(self.start_step * self.dt, self.end_step * self.dt)

    def __contains__(self, key: object) -> bool:
        """Whether the run holds the observation, by its handle or its id: ``swim in result``."""
        id = key if isinstance(key, str) else getattr(key, "id", None)
        return isinstance(id, str) and self.layout.observation(id) is not None

    def __iter__(self) -> Iterator[str]:
        """The run's observations, by id."""
        return iter(self.observations)

    @public
    def __getitem__(self, key: str | _Identified) -> ObservationOutputs:
        if not isinstance(key, str) and not isinstance(getattr(key, "id", None), str):
            raise ArgumentTypeRefused([Refusal(
                "call.wrong-type", RefusalClass.USER,
                f"a run's result is read by an observation, not by {type(key).__name__} {key!r}",
                f"index it by the handle observe returned, result[swim], or by an observation's id: "
                f"{', '.join(f'result[{o!r}]' for o in self.observations) or 'this run observes nothing'}")])
        id = key if isinstance(key, str) else key.id
        held = self.layout.observation(id)
        if held is None:
            raise NotInResult([Refusal(
                "result.no-observation", RefusalClass.USER,
                f"no observation {id!r} in this run",
                "index the result by the handle observe returned, result[swim], or by an observation's id: "
                f"{', '.join(f'result[{o!r}]' for o in self.observations) or 'this run observes nothing'}. "
                f"{did_you_mean(id, self.observations)}")])
        return ObservationOutputs(self, held)

    def _samples(self, path: str) -> Samples | None:
        if path not in self._recorded and path in self._loader and self._npz is not None:
            key = self._loader[path]
            held = self._npz.many([f"{key}/steps", f"{key}/values"])
            self._recorded[path] = Samples(held[f"{key}/steps"].astype(np.int64), held[f"{key}/values"])
        return self._recorded.get(path)

    @public
    def output(self, output: OutputLayout) -> Series | Reading:
        """``output`` in this run: a series when it is per step and recorded, a reading when it is at the end."""
        if output.kind is OutputKind.PER_STEP:
            samples = self._samples(output.path)
            if samples is None:
                raise NotRecorded([Refusal(
                    "result.not-recorded", RefusalClass.USER,
                    f"{output.path} is a per-step output, and this run did not record it",
                    f"record it: run(..., record=[{output.path!r}]) (or the handle observe returned, .{output.name}), "
                    "with every= for how often",
                    why=RECORDED_ONLY)])
            return Series(output, samples.steps, samples.values, dt=self.dt, every=self.every or 1,
                          start=self.start_step * self.dt, end=self.end_step * self.dt, flags=self._flag_source(),
                          clock=self.clock)
        start = output.after if output.after is not None else 0.0
        end = self.end_step * self.dt
        value = quantity_of(np.asarray(self.final[output.node][output.field]), output.dimension)
        return Reading(output.path, value, start, end, self._flag_source().over(start, end), output.rule)

    @public
    def series(self) -> list[Series]:
        return [cast(Series, self.output(output)) for output in self.layout.per_step()
                if output.path in self.recorded]

    @public
    def table(self) -> Table:
        """Every recorded series side by side at its samples' times, with a time column and the flags column: a
        mapping that ``pandas.DataFrame`` and ``polars.DataFrame`` take as it is, each column's unit in its name."""
        held = self.series()
        if not held:
            names = ", ".join(repr(output.path) for output in self.layout.per_step()) or "none declared"
            raise NotRecorded([Refusal(
                "result.not-recorded", RefusalClass.USER,
                "this run recorded no series, so it has no table",
                f"record the per-step outputs to keep: run(..., record=[...]), naming any of {names}; its outputs at "
                "the end are in text() and the manifest",
                why=RECORDED_ONLY)])
        return series_table(held)

    @public
    def to_csv(self, path: str | PathLike[str]) -> Path:
        """The table as CSV, its header the column names with their units, the flags its last column."""
        table = self.table()
        target = Path(path)
        names = list(table)
        with target.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(names)
            for row in range(len(table[names[0]])):
                writer.writerow([str(table[name][row]) if table[name].dtype.kind == "U"
                                 else repr(float(table[name][row])) for name in names])
        return target

    @public
    def arrays(self) -> dict[str, NDArray[Any]]:
        """What ``outputs.npz`` holds: every field after the last step (``final/<node>/<field>``), every field at
        every step when the run kept its history (``history/...``), each recorded series' step counts and values
        (``series/<observation>.<output>/steps``, ``/values``), and each flag at every step when the run recorded
        anything (``flags/<node>.<field>``)."""
        found = {f"final/{node}/{name}": np.asarray(value) for node, fields in self.final.items()
                 for name, value in fields.items()}
        for node, fields in (self.history or {}).items():
            found.update({f"history/{node}/{name}": np.asarray(value) for name, value in fields.items()})
        for path in self.recorded:
            samples = cast(Samples, self._samples(path))
            found[f"{series_key(path)}/steps"] = samples.steps
            found[f"{series_key(path)}/values"] = np.asarray(samples.values)
        for path, track in self.tracks.items():
            if track.raised is not None:
                found[flag_key(path)] = track.raised.astype(np.uint8)
        return found

    @public
    def manifest(self) -> dict[str, Any]:
        """What ``outputs.json`` holds: the run's steps and times, every output of every observation with its unit,
        dimension, window and where its values are, each value at the end written out, and every flag with what it
        means and what the run kept of it. With it, a run folder is readable without MIME."""
        observations: list[dict[str, Any]] = []
        for held in self.layout.observations:
            outputs: list[dict[str, Any]] = []
            for output in held.outputs:
                entry: dict[str, Any] = {
                    "name": output.name, "path": output.path, "node": output.node, "field": output.field,
                    "kind": output.kind.value, "unit": output.unit, "dimension": str(output.dimension),
                    "shape": list(output.shape), "description": output.description, "reduces": output.reduces,
                    "rule": output.rule.value if output.rule is not None else None, "after": output.after}
                if output.kind is OutputKind.AT_END:
                    reading = cast(Reading, self.output(output))
                    value = reading.value
                    entry.update({"array": f"final/{output.node}/{output.field}",
                                  "value": value.tolist() if isinstance(value, np.ndarray) else value,
                                  "window": [reading.start, reading.end],
                                  "flags": [read.manifest() for read in reading.flags]})
                else:
                    recorded = output.path in self.recorded
                    entry.update({"recorded": recorded,
                                  "steps": f"{series_key(output.path)}/steps" if recorded else None,
                                  "values": f"{series_key(output.path)}/values" if recorded else None})
                outputs.append(entry)
            observations.append({"id": held.id, "entry": held.entry, "outputs": outputs})
        flags: list[dict[str, Any]] = []
        for flag in self.layout.flags:
            track = self.tracks.get(flag.path)
            kept = track is not None and track.raised is not None
            flags.append({"path": flag.path, "entry": flag.owner, "label": flag.entry, "node": flag.node,
                          "field": flag.field, "meaning": flag.meaning, "name": flag.name,
                          "limits": [limit if math.isfinite(limit) else None for limit in flag.limits]
                          if flag.limits is not None else None,
                          "recorded": "every step" if kept else "the last step",
                          "array": flag_key(flag.path) if kept else None,
                          "was_raised": track.was_raised if track is not None else None,
                          "raised_at_last_step": track.at_last if track is not None else None})
        return {"format": MANIFEST_FORMAT, "version": MANIFEST_VERSION, "experiment": self.layout.experiment,
                "stamp": self.layout.stamp.record() if self.layout.stamp is not None else None,
                "dt": self.dt, "clock": self.clock.name, "start_step": self.start_step, "steps": self.steps,
                "start": self.start_step * self.dt, "end": self.end_step * self.dt,
                "every": self.every, "history": self.history is not None, "recorded": list(self.recorded),
                "observations": observations, "flags": flags,
                "run_flags": [read.manifest() for read in self.flags],
                "watches": [read.manifest() for read in self.watches],
                "reading": "steps count base steps from the graph's start; a sample at step k is the state at the end "
                           "of step k, at time k dt; a window (start, end] holds the steps whose end lies after start "
                           "and not after end; values are in SI, in each output's unit"}

    @public
    def to_npz(self, path: str | PathLike[str]) -> Path:
        """Every array :meth:`arrays` lists and the manifest (under ``manifest``, a JSON string), in one NPZ that
        :meth:`read` reads back and NumPy reads alone."""
        target = Path(path)
        np.savez(target, allow_pickle=False, **self.arrays(),
                 **{NPZ_MANIFEST: np.asarray(json.dumps(self.manifest(), sort_keys=True))})
        return target

    @public
    def to_mat(self, path: str | PathLike[str]) -> Path:
        """The declared outputs and flags as a MATLAB v5 file, through ``scipy.io.savemat``: a struct per observation
        (``swim_speed_umr``) holding a struct per output (``speed``, ``mean_speed``) with its ``unit`` and
        ``dimension``, its ``value`` for an output at the end, or its ``time`` axis (s) and ``values`` for a series the
        run recorded, with the window and the flags raised over it; ``flags``, every flag with what the run kept of it
        and the steps it was raised at (``was_raised`` is ``unknown``, never ``false``, where the run kept only its last
        step); ``run``; ``manifest``, the manifest of :meth:`manifest` as a JSON string; and ``name_map``, from each
        MATLAB name to the run's (an id such as ``swim_speed:umr.speed`` is not a MATLAB name). The raw state
        (``final``, ``history``) is in :meth:`to_npz` and not here. A name that cannot be mapped, or a variable of 2
        GiB or more, is refused naming it and NPZ. The file is read back by ``scipy.io.loadmat`` in MIME's tests;
        MATLAB itself has not been run on one (``loadmat`` reads an empty text, a dimensionless unit, as an empty
        array). The same on a result read from a folder."""
        from mime.results._mat import write_mat

        return write_mat(self, path)

    @public
    def text(self) -> str:
        lines = [f"run of {self.layout.experiment}: {self.steps} steps of {self.dt!r} s, "
                 f"{window_text(self.start_step * self.dt, self.end_step * self.dt)}",
                 *(ObservationOutputs(self, held).text() for held in self.layout.observations),
                 f"over the run, {flags_text(self.flags, meaning=True)}",
                 watches_text(self.watches),
                 self.layout.stamp.text() if self.layout.stamp is not None else "profile: not recorded (a run "
                                                                                 "written before profiles)"]
        return "\n".join(lines)

    @override
    def __str__(self) -> str:
        return self.text()

    @override
    def __repr__(self) -> str:
        return (f"<RunResult {self.layout.experiment}: {self.steps} steps, {len(self.observations)} observations, "
                f"{len(self.recorded)} series recorded>")

    @classmethod
    @public
    def read(cls, path: str | PathLike[str]) -> RunResult:
        """The result of a run, read back: from a folder ``run_to`` or ``mime run`` wrote (its ``outputs.json`` and
        ``outputs.npz``), on a local disk or on a Modal Volume (its mount, or what ``modal volume get`` fetched), or
        from an NPZ :meth:`to_npz` wrote. Arrays are read when first used, so a large folder opens at the cost of its
        manifest and its final state."""
        source = Path(path)
        if source.is_dir():
            manifest_path, npz_path = source / RUN_MANIFEST, source / "outputs.npz"
            if not manifest_path.is_file() or not npz_path.is_file():
                raise NotARunFolder([Refusal(
                    "result.not-a-run-folder", RefusalClass.USER,
                    f"{source} is not a run folder: it holds no {RUN_MANIFEST} and outputs.npz",
                    "read a folder that built.run_to(...) or `mime run <folder> --out <dir>` wrote: "
                    "RunResult.read(<dir>)")])
            manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
            npz = _Npz(npz_path)
            result = cls.from_manifest(manifest, npz)
            result.report = read_report(source / RUN_REPORT)
            return result
        else:
            npz = _Npz(source)
            if NPZ_MANIFEST not in npz.files:
                raise ArgumentRefused([Refusal(
                    "result.not-a-run-folder", RefusalClass.USER,
                    f"{source} holds no manifest: it is not an NPZ RunResult.to_npz wrote",
                    "read an NPZ result.to_npz(path) wrote, or a run folder: RunResult.read(<dir>)")])
            manifest = cast(dict[str, Any], json.loads(str(npz[NPZ_MANIFEST])))
        return cls.from_manifest(manifest, npz)

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any], npz: _Npz) -> RunResult:
        if manifest.get("format") != MANIFEST_FORMAT or manifest.get("version") not in READ_VERSIONS:
            raise ArgumentRefused([Refusal(
                "result.format", RefusalClass.USER,
                f"a manifest of {manifest.get('format')!r} version {manifest.get('version')!r}; this MIME reads "
                f"{MANIFEST_FORMAT!r} versions {', '.join(map(str, READ_VERSIONS))}",
                "read it with the MIME that wrote it (`pip install mime-engine==<version>`), or run the experiment "
                "again with this one: `mime run <folder> ...`")])
        layout = layout_of(manifest)
        final = _Nodes(npz, "final")
        _ = [npz.many([f"final/{node}/{field}" for field in fields]) for node, fields in final.items()]
        history = _Nodes(npz, "history") if manifest.get("history") else None
        tracks: dict[str, FlagTrack] = {}
        for flag in cast(list[dict[str, Any]], manifest["flags"]):
            key = cast(str | None, flag.get("array"))
            raised = npz[key].astype(bool) if key is not None else None
            tracks[str(flag["path"])] = FlagTrack(raised, bool(flag.get("raised_at_last_step")))
        loader = {str(path): series_key(str(path)) for path in cast(list[str], manifest.get("recorded", []))}
        return cls(int(manifest["steps"]), final, history, layout=layout, start_step=int(manifest["start_step"]),
                   tracks=tracks, every=cast(int | None, manifest.get("every")), loader=loader, npz=npz,
                   clock=cast(str | None, manifest.get("clock")),
                   watches=[WatchRead.from_manifest(w) for w in cast(list[dict[str, Any]], manifest.get("watches", []))])


def read_report(path: Path) -> dict[str, Any] | None:
    """A run folder's build report summary, or ``None`` when the folder holds none (one written before they did)."""
    if not path.is_file():
        return None
    held = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(held, dict) or cast(dict[str, Any], held).get("format") != REPORT_FORMAT:
        found = cast(dict[str, Any], held).get("format") if isinstance(held, dict) else None
        raise ArgumentRefused([Refusal(
            "result.format", RefusalClass.USER, f"{path} is a report of {found!r}; this MIME reads {REPORT_FORMAT!r}",
            "read it with the MIME that wrote it (`pip install mime-engine==<version>`), or run the experiment again "
            "with this one: `mime run <folder> ...`")])
    return cast(dict[str, Any], held)


def layout_of(manifest: Mapping[str, Any]) -> Layout:
    """The layout a manifest was written from."""
    observations: list[ObservationLayout] = []
    for held in cast(list[dict[str, Any]], manifest["observations"]):
        outputs = tuple(OutputLayout(
            str(held["id"]), str(o["name"]), str(o["node"]), str(o["field"]), OutputKind(o["kind"]),
            parse_unit(str(o["unit"])), tuple(int(n) for n in o["shape"]), str(o["description"]),
            cast(str | None, o.get("reduces")), Reduction(o["rule"]) if o.get("rule") else None,
            cast(float | None, o.get("after"))) for o in cast(list[dict[str, Any]], held["outputs"]))
        observations.append(ObservationLayout(str(held["id"]), str(held["entry"]), outputs))
    flags = tuple(FlagLayout(str(f["entry"]), str(f["label"]), str(f["node"]), str(f["field"]), str(f["meaning"]),
                             cast(str | None, f.get("name")),
                             tuple(math.inf if v is None else float(v) for v in cast(list[float | None], f["limits"]))
                             if f.get("limits") else None)
                  for f in cast(list[dict[str, Any]], manifest["flags"]))
    stamp = manifest.get("stamp")
    return Layout(str(manifest["experiment"]), float(manifest["dt"]), tuple(observations), flags,
                  Stamp.from_record(cast(Mapping[str, Any], stamp)) if isinstance(stamp, Mapping) else None)


__all__ = ["MANIFEST_FORMAT", "MANIFEST_VERSION", "NPZ_MANIFEST", "REPORT_FORMAT", "RUN_MANIFEST", "RUN_REPORT",
           "ObservationOutputs", "OutputRef", "RunResult", "Samples", "flag_key", "layout_of", "read_report",
           "series_key"]
