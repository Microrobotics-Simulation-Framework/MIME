"""The saved form's JSON Schemas against the loader, the authority: two readers of one source, the closed TypedDicts.

The schemas are generated from the TypedDicts (``mime.docs._schemas``) and checked here by an independent validator,
``jsonschema``. The differential runs both readers over one corpus: the committed documents and locks (every golden
one validates, as the loader reads it, and the raw files of the earlier forms are the plants of the gap
``earlier-form``), planted inputs (an unknown key, a missing key, a value of another type, a boolean for a number, and
inputs for each recorded gap), and documents and locks drawn from the TypedDicts with Hypothesis, with a key added or
removed. The set of gaps on which they disagree must **equal** the recorded gaps that are disagreements
(``GAPS``): a new disagreement fails, and so does a recorded one that no longer shows. An input outside the corpus is
not seen.

What the loader is here: for a document, ``read_document``, the YAML loader that refuses a repeated key and the shape
check, reading an earlier form under the current one; for a lock, MIME's one lock reader (``mime._lock_json``, which
refuses a repeated key) and the shape check, as ``read_folder`` reads one. The meaning of a value (a unit, an entry, a
file name) is a later stage of the load, outside both readers (the gap ``meaning``).
"""

import copy
import json
import math
import re
from collections.abc import Callable
from pathlib import Path

import jsonschema
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from test_schema import instances_of

from mime._lock_json import lock_data
from mime.docs._schemas import GAPS, Gap, json_schema, saved_form_schemas
from mime.saved import DOCUMENT_SCHEMA, LOCK_SCHEMA, ExperimentDocument, Lock, conforms, read_document
from mime.saved._schema import reachable_typed_dicts, typed_dict_fields
from mime.saved._versions import upgraded_document, upgraded_lock

PROPERTY = settings(max_examples=100, deadline=None)
SAVED = Path(__file__).parent
CURRENT = SAVED / "current" / "chain_screw"
#: The committed folders of earlier forms: the versions 1 to 3 under ``previous/``, the version 4 folder the stage
#: writer's test holds (its stage is the first writer's), and the version 5 to 9 documents and locks
#: (``form5/chain_screw`` to ``form9/chain_screw``, the current folder as versions 5 to 9 wrote it).
PREVIOUS = [*sorted((SAVED / "previous").iterdir()), SAVED / "stage_writers" / "chain_screw",
            SAVED / "form5" / "chain_screw", SAVED / "form6" / "chain_screw", SAVED / "form7" / "chain_screw",
            SAVED / "form8" / "chain_screw", SAVED / "form9" / "chain_screw"]
SCHEMAS = saved_form_schemas()


def validator(file: str, schemas=SCHEMAS) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(schemas[f"{file}.schema.json"])


def loader_refuses(file: str, text: str, tmp: Path) -> bool:
    if file == "document":
        path = tmp / "experiment.mime.yaml"
        path.write_text(text, encoding="utf-8")
        return bool(read_document(path)[1])
    try:
        data = lock_data(text)
    except ValueError:
        return True
    return bool(conforms(upgraded_lock(data)[0], Lock)[1])


def schema_refuses(file: str, text: str, schemas=SCHEMAS) -> bool:
    data = yaml.safe_load(text) if file == "document" else json.loads(text)
    return not validator(file, schemas).is_valid(data)


def verdicts(file: str, text: str, tmp: Path, schemas=SCHEMAS) -> tuple[str, str]:
    return (("refuses" if loader_refuses(file, text, tmp) else "accepts"),
            ("refuses" if schema_refuses(file, text, schemas) else "accepts"))


def golden(file: str) -> str:
    return (CURRENT / ("experiment.mime.yaml" if file == "document" else "mime.lock")).read_text(encoding="utf-8")


def replaced(text: str, old: str, new: str) -> str:
    assert text.count(old) >= 1, old
    return text.replace(old, new, 1)


def earlier_forms() -> list[tuple[str, str]]:
    """The committed folders of earlier forms, their raw files as written, never upgraded."""
    names = {"document": "experiment.mime.yaml", "lock": "mime.lock"}
    return [(file, (folder / names[file]).read_text(encoding="utf-8")) for folder in PREVIOUS for file in names]


#: Each recorded gap's planted inputs, made from the committed folders' files: gap -> [(file, text)].
GAP_PLANTS: dict[str, Callable[[], list[tuple[str, str]]]] = {
    "duplicate-key": lambda: [("document", replaced(golden("document"), "experiment: chain_screw\n",
                                                    "experiment: chain_screw\nexperiment: chain_screw\n"))],
    "duplicate-key-in-lock": lambda: [("lock", replaced(golden("lock"), '"schema": "mime.lock/14"',
                                                        '"schema": "mime.lock/14",\n "schema": "mime.lock/14"'))],
    "integral-float": lambda: [("document", replaced(golden("document"), "seed: 0\n", "seed: 0.0\n")),
                               ("lock", replaced(golden("lock"), '"steps": 8', '"steps": 8.0'))],
    "non-finite": lambda: [("document", replaced(golden("document"), "value: 0.001\n", "value: .nan\n")),
                           ("lock", replaced(golden("lock"), '"rtol": 2e-14', '"rtol": NaN'))],
    "non-string-key": lambda: [("document", replaced(golden("document"), "sources: {}\n",
                                                     "sources:\n  1: a reason\n"))],
    "earlier-form": earlier_forms,
    "meaning": lambda: [("document", replaced(golden("document"), "unit: s\n", "unit: furlong\n")),
                        ("document", replaced(golden("document"), "  seed: 0\n",
                                              "  seed: 0\n  duration:\n    value: 0.0015\n    unit: s\n")),
                        ("lock", replaced(golden("lock"), '"distribution": "mime-engine"',
                                          '"distribution": "not a name!"'))],
}

#: Planted inputs on which the two readers must agree, both refusing: (name, file) -> text.
AGREEMENT_PLANTS: dict[tuple[str, str], Callable[[], str]] = {
    ("unknown key", "document"): lambda: golden("document") + "colour: red\n",
    ("missing key", "document"): lambda: replaced(golden("document"), "profile: exploratory\n", ""),
    ("another type", "document"): lambda: replaced(golden("document"), "seed: 0\n", "seed: zero\n"),
    ("a boolean for a number", "document"): lambda: replaced(golden("document"), "value: 0.001\n", "value: true\n"),
    ("a boolean for an integer", "document"): lambda: replaced(golden("document"), "seed: 0\n", "seed: true\n"),
    ("a fraction for an integer", "document"): lambda: replaced(golden("document"), "seed: 0\n", "seed: 0.5\n"),
    ("a fraction for an integer", "lock"): lambda: replaced(golden("lock"), '"steps": 8', '"steps": 8.5'),
    ("unknown key", "lock"): lambda: replaced(golden("lock"), '"schema": "mime.lock/14"',
                                              '"colour": "red",\n "schema": "mime.lock/14"'),
    ("a boolean for a number", "lock"): lambda: replaced(golden("lock"), '"rtol": 2e-14', '"rtol": true'),
}


def gap_observations(tmp: Path, gaps=GAPS, schemas=SCHEMAS) -> dict[str, list[tuple[str, tuple[str, str]]]]:
    """Each recorded gap's planted inputs with the two readers' verdicts on them, each with its file."""
    return {gap.name: [(file, verdicts(file, text, tmp, schemas)) for file, text in GAP_PLANTS[gap.name]()]
            for gap in gaps}


def differential_problems(tmp: Path, gaps: tuple[Gap, ...] = GAPS, schemas=SCHEMAS) -> list[str]:
    """Every way the readers' disagreements on the deterministic corpus differ from the recorded gaps ``gaps``: a
    planted input of a gap that does not give the gap's verdicts, a disagreement on a golden file or an agreement plant,
    and a plant whose gap is not recorded."""
    problems: list[str] = []
    recorded = {gap.name: gap for gap in gaps}
    for gap_id in sorted(set(GAP_PLANTS) - set(recorded)):
        for file, text in GAP_PLANTS[gap_id]():
            loader, schema = verdicts(file, text, tmp, schemas)
            if loader != schema:
                problems.append(f"the {file} plant of {gap_id!r}: the loader {loader} it and the schema {schema} it, "
                                f"a disagreement no recorded gap explains")
    for gap_id, gap in recorded.items():
        for file, text in GAP_PLANTS[gap_id]():
            seen = verdicts(file, text, tmp, schemas)
            if seen != (gap.loader, gap.schema):
                problems.append(f"the {file} plant of {gap_id!r}: the loader {seen[0]} it and the schema {seen[1]} "
                                f"it; the gap records {gap.loader} and {gap.schema}")
    for file in ("document", "lock"):
        if verdicts(file, golden(file), tmp, schemas) != ("accepts", "accepts"):
            problems.append(f"the committed {file} is not accepted by both readers")
    for (name, file), make in AGREEMENT_PLANTS.items():
        if verdicts(file, make(), tmp, schemas) != ("refuses", "refuses"):
            problems.append(f"the {file} with {name}: the readers do not both refuse it")
    return problems


def test_every_typed_dict_is_a_closed_definition_with_its_required_keys():
    for file, root in (("document", ExperimentDocument), ("lock", Lock)):
        defs = SCHEMAS[f"{file}.schema.json"]["$defs"]
        for cls in reachable_typed_dicts(root):
            held = defs[cls.__name__]
            assert held["additionalProperties"] is False
            assert held["required"] == sorted(cls.__required_keys__)
            assert set(held["properties"]) == set(typed_dict_fields(cls))
    assert SCHEMAS["document.schema.json"]["$id"].endswith(DOCUMENT_SCHEMA)
    assert SCHEMAS["lock.schema.json"]["$id"].endswith(LOCK_SCHEMA)


def test_the_schemas_are_valid_json_schemas_and_name_every_recorded_gap_of_their_file():
    for file in ("document", "lock"):
        schema = SCHEMAS[f"{file}.schema.json"]
        jsonschema.Draft202012Validator.check_schema(schema)
        description = schema["description"]
        assert [gap.name for gap in GAPS if f"- {gap.name}: {gap.text}" in description] == \
            [gap.name for gap in GAPS if file in gap.files]


@pytest.mark.parametrize("folder", [CURRENT, *PREVIOUS], ids=lambda p: p.name)
def test_every_golden_document_and_lock_validates_as_the_loader_reads_it(folder):
    """A folder of an earlier form is read under the current one (its upgrade), which is what both readers judge."""
    document = upgraded_document(yaml.safe_load((folder / "experiment.mime.yaml").read_text(encoding="utf-8")))[0]
    lock = upgraded_lock(json.loads((folder / "mime.lock").read_text(encoding="utf-8")))[0]
    assert conforms(document, ExperimentDocument)[1] == [] and conforms(lock, Lock)[1] == []
    assert list(validator("document").iter_errors(document)) == []
    assert list(validator("lock").iter_errors(lock)) == []


def test_the_readers_disagree_exactly_on_the_recorded_gaps(tmp_path):
    assert set(GAP_PLANTS) == {gap.name for gap in GAPS}, "every recorded gap has its planted input, and only those"
    assert differential_problems(tmp_path) == []
    observed = {gap_id for gap_id, plants in gap_observations(tmp_path).items()
                if any(loader != schema for _, (loader, schema) in plants)}
    assert observed == {gap.name for gap in GAPS if gap.disagreement}
    assert len(GAP_PLANTS["earlier-form"]()) == 2 * len(PREVIOUS) == 20


@pytest.mark.parametrize("missing", [gap.name for gap in GAPS if gap.disagreement])
def test_control_n4_without_any_one_disagreement_recorded_the_differential_fails(tmp_path, missing):
    """N4: each recorded disagreement (the integral float, the repeated key in the document and in the lock, the key
    that is not a string, an earlier form) shows in the corpus, so a gap list without it fails, naming the plant; a
    deterministic corpus, so it fails on every run."""
    problems = differential_problems(tmp_path, tuple(gap for gap in GAPS if gap.name != missing))
    assert any(f"{missing!r}" in problem and "no recorded gap explains" in problem for problem in problems), problems


def test_control_a_schema_generated_wrong_disagrees_where_no_gap_is_recorded(tmp_path):
    """A generator that wrote an integer as any number: the planted seed of a fraction is then a disagreement no gap
    explains (the loader refuses it, the planted schema accepts it), and the gaps' verdicts move."""
    planted = copy.deepcopy(SCHEMAS)
    planted["document.schema.json"]["$defs"]["RunDoc"]["properties"]["seed"] = {"type": "number"}
    text = replaced(golden("document"), "seed: 0\n", "seed: 0.5\n")
    assert verdicts("document", text, tmp_path, planted) == ("refuses", "accepts")
    assert verdicts("document", text, tmp_path) == ("refuses", "refuses")


def test_control_a_gap_recorded_with_the_wrong_verdict_fails(tmp_path):
    flipped = tuple(Gap(g.name, g.files, g.loader, "refuses", g.text) if g.name == "integral-float" else g for g in GAPS)
    problems = differential_problems(tmp_path, flipped)
    assert any("'integral-float'" in problem and "the gap records refuses and refuses" in problem
               for problem in problems), problems


def drawn_text(file: str, value: object) -> str:
    return yaml.safe_dump(value, sort_keys=False) if file == "document" else json.dumps(value)


ROOTS = {"document": ExperimentDocument, "lock": Lock}


@pytest.mark.parametrize("file", ["document", "lock"])
@PROPERTY
@given(data=st.data())
def test_on_drawn_files_and_their_changes_the_readers_agree(file, data, tmp_path_factory):
    """Files drawn from the TypedDicts, as drawn and with a key added, a required key removed or a value replaced by
    another drawn value: the two readers agree on each. A disagreement here is one the recorded gaps do not hold."""
    tmp = tmp_path_factory.mktemp("drawn")
    value = data.draw(instances_of(ROOTS[file]))
    changed = [value, {**value, "added_key": 1}]
    required = sorted(ROOTS[file].__required_keys__)
    key = data.draw(st.sampled_from(required))
    changed.append({k: v for k, v in value.items() if k != key})
    other = data.draw(st.sampled_from(sorted(value)))
    changed.append({**value, other: data.draw(instances_of(ROOTS[file]))[data.draw(st.sampled_from(required))]})
    disagreements = {gap.name: (gap.loader, gap.schema) for gap in GAPS if gap.disagreement}
    for each in changed:
        text = drawn_text(file, each)
        seen = verdicts(file, text, tmp)
        if seen[0] != seen[1]:
            gap = explanation(file, text, tmp)
            assert gap in disagreements and disagreements[gap] == seen, (seen, reasons(file, text, tmp))


_NOT_AN_INTEGER = re.compile(r"\$[^ ]*: (?P<value>\S+) is not an integer")


def explanation(file: str, text: str, tmp: Path) -> str | None:
    """The recorded gap a disagreement on a drawn file shows, when its every refusal is one: a drawn number can land in
    an integer's place as a float with no fractional part (``integral-float``)."""
    loader, _ = reasons(file, text, tmp, limit=None)
    values = [_NOT_AN_INTEGER.fullmatch(problem) for problem in loader]
    if loader and all(match is not None and _integral_float(match.group("value")) for match in values):
        return "integral-float"
    return None


def _integral_float(written: str) -> bool:
    try:
        number = float(written)
    except ValueError:
        return False
    return not written.lstrip("-").isdigit() and math.isfinite(number) and number.is_integer()


def reasons(file: str, text: str, tmp: Path, limit: int | None = 3) -> tuple[list[str], list[str]]:
    """Why each reader refuses ``text``: the loader's problems and the schema's errors (the first ``limit`` of each)."""
    data = yaml.safe_load(text) if file == "document" else json.loads(text)
    if file == "document":
        path = tmp / "experiment.mime.yaml"
        path.write_text(text, encoding="utf-8")
        loader = [refusal.message.removeprefix(f"{path}: ") for refusal in read_document(path)[1]]
    else:
        loader = list(conforms(upgraded_lock(data)[0], Lock)[1])
    return loader[:limit], [error.message[:300] for error in validator(file).iter_errors(data)][:limit]


def test_the_schema_files_have_their_names_and_one_text_per_schema():
    assert sorted(SCHEMAS) == ["document.schema.json", "lock.schema.json"]
    assert json_schema(ExperimentDocument, DOCUMENT_SCHEMA, "document") == SCHEMAS["document.schema.json"]
