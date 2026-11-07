"""A study's spec file, ``study.mime.yaml``: a study as data, so that it saves, travels and reruns on another machine.

::

    format: mime-study/1
    base:
      folder: experiment            # the saved experiment, from this file's folder (``--base`` names another)
      input_hash: 3f2a...           # its lock's input hash: a base that differs is refused
    combine: grid                   # grid (every combination) or along (the i-th value of each axis together)
    axes:
      field.commands.frequency:
        values: [10 Hz, 20 Hz]      # a number and a unit of mime.units, or a name
        reason: why this axis is varied
    duration: 1 s
    record: [swim_speed:umr.speed]
    every: 1 ms
    batch: 16
    drop_refused: false
    seed: {policy: experiment}

It holds no code: it is read with YAML's safe loader, a key given twice is refused, and every key is one of these. Its
schema is closed and versioned, and a newer one is refused."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

import yaml

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.results._study_folder import encode_value
from mime.study._override import parse_value
from mime.units import Quantity

if TYPE_CHECKING:
    from mime.study._study import Points, Study

SPEC_FORMAT = "mime-study/1"
SPEC_KEYS = frozenset({"format", "base", "combine", "axes", "duration", "record", "every", "batch", "drop_refused", "seed"})
BASE_KEYS = frozenset({"folder", "input_hash"})
AXIS_KEYS = frozenset({"values", "reason"})
SEED_POLICIES = ("experiment",)


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> ArgumentRefused:
    return ArgumentRefused([Refusal(code, category, message, action)])


class _Loader(yaml.SafeLoader):
    """YAML's safe loader, refusing a key given twice in a mapping (it would silently keep the last)."""

    @override
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[object] = set()
        for key_node, _ in cast(list[tuple[yaml.Node, yaml.Node]], node.value):
            key = cast(object, self.construct_object(key_node, deep=True))  # pyright: ignore[reportUnknownMemberType]
            if key in seen:
                raise _refuse("study.spec", RefusalClass.USER, f"the key {key!r} is given twice in one mapping (line {key_node.start_mark.line + 1})",
                                    "give each key once, as in `axes: {dt: {values: [1 ms]}}`; the last would silently replace the first")
            seen.add(key)
        return super().construct_mapping(node, deep)


def value_text(value: object, path: str) -> object:
    """An axis value as a spec writes it: a quantity as ``<number(s)> <unit>``, a name as it is, a number or boolean as
    it is."""
    if isinstance(value, Quantity):
        held = encode_value(value, path)
        magnitude = held["value"]
        text = ", ".join(repr(float(v)) for v in cast(list[float], magnitude)) if isinstance(magnitude, list) \
            else repr(float(cast(float, magnitude)))
        unit = held["unit"]
        return text if unit == "1" else f"{text} {unit}"
    if isinstance(value, str):
        if parse_value(value) is not None:
            raise _refuse("study.spec", RefusalClass.USER, f"the value {value!r} of the axis {path} reads as a quantity in a spec file",
                                "write a name that is not a number with a unit, such as `float64`, or the quantity as `5 mm` without quotes")
        return value
    if isinstance(value, (bool, int, float)):
        return value
    encode_value(value, path)  # refuses, naming the axis
    return value


def value_of(item: object, path: str) -> object:
    """An axis value read from a spec: text that is a quantity is one, other text a name, other scalars as they are."""
    if isinstance(item, str):
        held = parse_value(item)
        return item if held is None else held
    if isinstance(item, (bool, int, float)):
        return item
    raise _refuse("study.spec", RefusalClass.USER, f"a value of the axis {path} is {item!r}, which is none of a quantity ('20 Hz'), a name, a number "
                        f"or a boolean", "write each value as `10 Hz`, a number or a name")


def spec_text(points: Points, *, duration: Quantity, record: Sequence[str], every: Quantity | None, batch: int | None,
              drop_refused: bool, reasons: Mapping[str, str], base: str, input_hash: str | None) -> str:
    """The spec file of a study, in its normal form."""
    paths = [str(axis) for axis in points.axes]
    lists = points.lists if points.lists else tuple(tuple(row[i] for row in points.rows) for i in range(len(paths)))
    combine = "grid" if points.kind == "grid" else "along"
    axes: dict[str, Any] = {}
    for path, values in zip(paths, lists, strict=True):
        held: dict[str, Any] = {"values": [value_text(v, path) for v in values]}
        if reasons.get(path):
            held["reason"] = reasons[path]
        axes[path] = held
    document: dict[str, Any] = {
        "format": SPEC_FORMAT, "base": {"folder": base, "input_hash": input_hash}, "combine": combine, "axes": axes,
        "duration": value_text(duration, "duration"), "record": list(record)}
    if every is not None:
        document["every"] = value_text(every, "every")
    if batch is not None:
        document["batch"] = batch
    document.update(drop_refused=drop_refused, seed={"policy": "experiment"})
    return yaml.safe_dump(document, sort_keys=False, default_flow_style=None, width=120, allow_unicode=True)


def read_spec(path: Path) -> dict[str, Any]:
    """The spec at ``path``, read with the safe loader and held to the schema: closed, versioned, no key twice."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise _refuse("study.spec", RefusalClass.USER, f"{path} cannot be read ({error})", "name a spec file: `mime study plan study.mime.yaml`") from error
    try:
        document = cast(object, yaml.load(text, Loader=_Loader))  # noqa: S506 - the safe loader, with a check
    except yaml.YAMLError as error:
        raise _refuse("study.spec", RefusalClass.USER, f"{path} is not YAML ({error})", "write the spec as YAML, `axes: {dt: {values: [1 ms, 2 ms]}}`: docs/results.md has the keys") from error
    if not isinstance(document, dict):
        raise _refuse("study.spec", RefusalClass.USER, f"{path} is not a mapping", "a spec file is a YAML mapping, `format: mime-study/1`, `base: {...}`, `axes: {...}`: docs/results.md has the keys")
    held = cast(dict[str, Any], document)
    if held.get("format") != SPEC_FORMAT:
        raise _refuse("study.spec", RefusalClass.USER, f"a study spec of format {held.get('format')!r}; this MIME reads {SPEC_FORMAT!r}",
                            "read it with the MIME that wrote it, or write the study again with this one: `Study.run_to(folder)` writes `study.mime.yaml`")
    extra = sorted(set(held) - SPEC_KEYS)
    if extra:
        raise _refuse("study.spec", RefusalClass.USER, f"the spec holds {', '.join(map(repr, extra))}, which this format does not "
                            f"(its keys: {', '.join(sorted(SPEC_KEYS))})",
                            "a spec holds no code and no key this format does not name: delete the key, or keep it in your script as `Study(exp).sweep(...)`")
    for key in ("base", "axes", "duration"):
        if key not in held:
            raise _refuse("study.spec", RefusalClass.USER, f"the spec has no {key!r}", f"add `{key}:`; docs/results.md has the keys")
    base = held["base"]
    if not isinstance(base, dict) or set(cast(dict[str, Any], base)) - BASE_KEYS or "folder" not in base:
        raise _refuse("study.spec", RefusalClass.USER, "`base` is the experiment's `folder` and its `input_hash`", "write `base: {folder: experiment, input_hash: sha256:...}`")
    axes = held["axes"]
    if not isinstance(axes, dict) or not axes:
        raise _refuse("study.spec", RefusalClass.USER, "`axes` is a mapping of each axis's path to its `values` (and its `reason`)",
                            "write `axes: {field.commands.frequency: {values: [10 Hz, 20 Hz]}}`")
    for path_, axis in cast(dict[str, Any], axes).items():
        if not isinstance(axis, dict) or "values" not in axis or set(cast(dict[str, Any], axis)) - AXIS_KEYS \
                or not isinstance(cast(dict[str, Any], axis)["values"], list):
            raise _refuse("study.spec", RefusalClass.USER, f"the axis {path_!r} is a mapping of a list `values` and, if there is one, a `reason`",
                                f"write `{path_}: {{values: [...], reason: ...}}`")
    if held.get("combine", "grid") not in ("grid", "along"):
        raise _refuse("study.spec", RefusalClass.USER, f"`combine` is 'grid' or 'along', not {held['combine']!r}", "write `combine: grid`")
    if not isinstance(held.get("drop_refused", False), bool):
        raise _refuse("study.spec", RefusalClass.USER, f"`drop_refused` is {held['drop_refused']!r}, which is not true or false",
                            "write `drop_refused: true` or `drop_refused: false`, unquoted")
    record = held.get("record", [])
    if not isinstance(record, list) or not all(isinstance(each, str) for each in cast(list[object], record)):
        raise _refuse("study.spec", RefusalClass.USER, f"`record` is {record!r}, which is not a list of outputs",
                            "write `record: [swim_speed:umr.speed]`, a list even for one output")
    policy = cast(dict[str, Any], held.get("seed") or {"policy": "experiment"})
    if set(policy) - {"policy"} or policy.get("policy") not in SEED_POLICIES:
        raise _refuse("study.spec", RefusalClass.USER, f"the seed policy is {policy!r}; this release has {', '.join(SEED_POLICIES)} (every member "
                            f"runs the experiment's own seed)",
                            "write `seed: {policy: experiment}`; a policy that draws seeds comes with the first entry that draws noise")
    return held


def study_from_spec(path: Path, base: Path | None = None) -> tuple[Study, dict[str, Any]]:
    """The study a spec file describes, over the experiment it names (or ``base``): loaded and checked against its
    lock, and held to the input hash the spec states. Returns the study and the spec as read."""
    from mime.experiment import Experiment
    from mime.study._study import Study, along, grid

    spec = read_spec(path)
    folder = base if base is not None else (path.parent / str(spec["base"]["folder"])).resolve()
    experiment = Experiment.load(folder)
    form = experiment.declared_as()[2]
    assert form is not None
    found = str(form.lock["hashes"]["input"])
    wanted = spec["base"].get("input_hash")
    if wanted is not None and wanted != found:
        raise _refuse("study.spec", RefusalClass.USER, f"the spec is of the experiment with input hash {wanted}, and {folder} holds {found}",
                            "run the spec on the experiment it was written for, or write it again for this one "
                            "(`mime study plan` prints its base)")
    axes = cast(dict[str, dict[str, Any]], spec["axes"])
    lists = {name: [value_of(v, name) for v in cast(list[object], held["values"])] for name, held in axes.items()}
    points = (along if spec.get("combine") == "along" else grid)(cast(Mapping[object, Sequence[object]], lists))
    duration, every = value_of(spec["duration"], "duration"), spec.get("every")
    if not isinstance(duration, Quantity):
        raise _refuse("study.spec", RefusalClass.USER, f"`duration` is {spec['duration']!r}, not a time", "write `duration: 1 s`")
    study = Study(experiment).sweep(
        points, duration=duration, record=[str(r) for r in spec.get("record", [])],
        every=cast(Quantity, value_of(every, "every")) if every is not None else None,
        batch=spec.get("batch"), drop_refused=spec.get("drop_refused", False),
        reasons={name: str(held["reason"]) for name, held in axes.items() if held.get("reason")})
    return study, spec


__all__ = ["SPEC_FORMAT", "read_spec", "spec_text", "study_from_spec", "value_of", "value_text"]
