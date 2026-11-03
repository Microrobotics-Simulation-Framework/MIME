"""Evidence and claims: what a test id names once pytest has collected it, and the shape a claim must have.

**The resolver.** A claim's evidence is a pytest node id (``tests/oracles/test_drag.py::test_x``, or with its
parameters, ``...::test_x[a]``; without them it names every parametrisation) or a MADDENING ``@verification_benchmark``
id. Both resolve through pytest's own collection, run with ``-m ""`` so that no marker deselects anything: an id that is
not collected fails, whatever its file's text says, so a test renamed with its old name left in a comment, or a name
written only in a docstring, fails.

**Evidence is a test that ran and passed in a named lane.** Where a lane's JUnit reports are read (:func:`resolve`'s
``lanes``, each lane by its name), an id counts (:attr:`Resolution.counts`) when every node it names passed in one of
them and none is expected to fail: a skip, an xfail, a failure and a node the lane did not run (deselected, as a
``slow`` test is) each leave it uncounted, and a ``pytest.skip()`` in a test's body, which collection never sees, shows
there. Where no report is read (``mime docs check`` collects only), an id counts when every node it names runs in CI:
none is marked ``skip``, expected to fail (``xfail``, as pytest evaluates the mark), ``slow`` (no CI lane runs one) or
``gpu`` (CI has no CUDA). Each of those prints "not run in CI" and its reason, never "default lane". A ``skipif`` counts
as run, since CI's lanes allow no skipped test (``scripts/check_shards.py --max-skipped 0``), so its condition is false
on CI's runner; where it holds on the machine that collects, the lane says so ("skipped on this machine"). These rules
read pytest's own evaluation of the skip and xfail marks (``_pytest.skipping``), and fall back to the markers' names
where that module is gone.

What it does not see: a test that is collected and passes while it tests nothing (a same-name helper, a body of
``pass``, a threshold looser than its claim's) counts. A threshold is held by one authority per number (the test reads
the claim's value, :func:`claim`); an empty body and a helper are held by no rule here. Nor does anything tie a cited
test to the claim or statement that cites it: the id is a name the author chose.

Its interface, for the evidence report and ``scripts/check_anomalies.py``: :func:`collect` (one collection of a
checkout, or of the files the ids name), :func:`junit_outcomes` (one lane's reports), :func:`resolve` (each id against
both), :class:`Resolution` (its ``problem``, ``lane``, ``runs_in_ci``, ``outcomes`` and ``counts``, and ``text()``), and
:func:`claim_problems`, :func:`read_claims` and :func:`claim` for claims.

**Claims.** A claim is a field of the anomaly record (``claims``), not a second store: the quantity, its value and
unit, its **basis** (how it is known), its **role** (what it may be read as), the **oracle kind** it was compared
against, its evidence, its conditions (a point or a region), for a bound ``domain`` and ``n``, and ``says``, one line of
what it bounds and what it estimates. The basis decides the roles a claim may take:

========== ============================================================ ==========================================
basis      what it is                                                   roles
========== ============================================================ ==========================================
declared   stated, covered by no test                                   limit, rule
computed   against an analytic, manufactured or software oracle, or     bound (not with an implementation oracle),
           (kind ``implementation``) the code's own formula again       estimate, limit
measured   against a physical measurement                               bound, estimate, acceptance, limit
fitted     a fit's result, with the fit's dataset                       bound, estimate, limit
========== ============================================================ ==========================================

A ``bound`` is an upper bound on an error over its stated ``domain`` and carries ``domain``, ``n`` and its value. A
maximum over finitely many points is a lower bound on the supremum, not an upper bound over the range, so a claim of any
basis is a ``bound`` only with held-out evidence (``held_out``, an id that is not among its own evidence) that the
bound holds at points it was not taken from; without it, it is an ``estimate``, which bounds nothing. (The design also
lets a bound stand where ``n`` covers the domain; the shape has no field that says so, so that route is closed until it
has one.) A value is a finite number. An ``acceptance`` grounds sign, frame and units against a measurement and claims
no accuracy. An ``implementation`` oracle evaluates the code's own formula again: it is not accuracy evidence, and
prints so.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

BASES = ("declared", "computed", "measured", "fitted")
ROLES = ("bound", "estimate", "acceptance", "limit", "rule")
ORACLE_KINDS = ("analytic", "manufactured", "software oracle", "implementation", "measurement")
#: The roles each basis may take (the table above).
ROLES_BY_BASIS: Mapping[str, frozenset[str]] = {
    "declared": frozenset({"limit", "rule"}),
    "computed": frozenset({"bound", "estimate", "limit"}),
    "measured": frozenset({"bound", "estimate", "acceptance", "limit"}),
    "fitted": frozenset({"bound", "estimate", "limit"}),
}
#: The oracle kinds each basis is compared against: a declared claim names none.
KINDS_BY_BASIS: Mapping[str, frozenset[str]] = {
    "declared": frozenset(),
    "computed": frozenset({"analytic", "manufactured", "software oracle", "implementation"}),
    "measured": frozenset({"measurement"}),
    "fitted": frozenset({"analytic", "manufactured", "software oracle", "measurement"}),
}
CLAIM_FIELDS = ("id", "quantity", "value", "unit", "basis", "role", "oracle_kind", "evidence", "conditions", "domain",
                "n", "dataset", "held_out", "says")
_REQUIRED = ("id", "quantity", "basis", "role", "says")

DEFAULT_LANE = "default lane"
NOT_RUN = "not run in CI"
#: What each mark that keeps a test out of CI's lanes prints.
LANES = {"skip": f"{NOT_RUN} (marked skip)", "xfail": f"{NOT_RUN} (xfail: expected to fail)",
         "slow": f"{NOT_RUN} (slow)", "gpu": f"{NOT_RUN} (gpu: skipped without CUDA)"}
#: A MADDENING verification benchmark's id (``MADD-VER-001``, a lab's ``LABX-VER-002``).
BENCHMARK = re.compile(r"[A-Z][A-Z0-9]*-VER-\d+")
_NODE_ID = re.compile(r"(?P<file>[^:\s]+\.py)(?P<rest>(?:::[^:\s\[]+)+)(?P<params>\[.*\])?")
_CLAIM_ID = re.compile(r"[a-z][a-z0-9_]*")

# Run in a fresh interpreter: pytest's collection with every marker kept, each collected item's node id, its markers
# (module and class marks included), a benchmark id where the test carries one, and pytest's own evaluation of its skip
# and xfail marks on this machine (the reason, where one applies), written as JSON.
_COLLECTOR = r"""
import json, sys
import pytest
try:
    from _pytest.skipping import evaluate_skip_marks, evaluate_xfail_marks
except ImportError:
    evaluate_skip_marks = evaluate_xfail_marks = None


def evaluated(evaluate, item, marker):
    if evaluate is None:
        held = item.get_closest_marker(marker)
        return None if held is None else str(held.kwargs.get("reason", marker))
    try:
        found = evaluate(item)
    except Exception as error:
        return f"its condition cannot be evaluated: {error}"
    return None if found is None else (found.reason or marker)

class Collector:
    def __init__(self):
        self.items, self.errors = {}, []

    def pytest_collectreport(self, report):
        if report.failed:
            text = str(report.longrepr).strip().splitlines()
            self.errors.append(f"{report.nodeid or '<session>'}: {text[-1] if text else 'collection failed'}")

    def pytest_collection_finish(self, session):
        for item in session.items:
            held = getattr(getattr(item, "obj", None), "_benchmark", None)
            self.items[item.nodeid] = {"markers": sorted({mark.name for mark in item.iter_markers()}),
                                       "benchmark": getattr(held, "benchmark_id", None),
                                       "skipped": evaluated(evaluate_skip_marks, item, "skip"),
                                       "xfail": evaluated(evaluate_xfail_marks, item, "xfail")}

collector = Collector()
code = pytest.main(["--collect-only", "-q", "-m", "", "-p", "no:cacheprovider", *sys.argv[2:]], plugins=[collector])
with open(sys.argv[1], "w", encoding="utf-8") as out:
    json.dump({"exit": int(code), "items": collector.items, "errors": collector.errors}, out)
"""


@dataclass(frozen=True)
class Collection:
    """What one collection found: each node id with its markers, the benchmark ids by the node id that carries them,
    the collection's errors, the files it was given (none: the checkout's own ``testpaths``), and, by node id, the
    reason pytest gives where it would skip a test on this machine (``skipped``) or expects it to fail (``xfail``)."""

    items: Mapping[str, tuple[str, ...]]
    benchmarks: Mapping[str, str]
    errors: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    skipped: Mapping[str, str] = field(default_factory=dict[str, str])
    xfail: Mapping[str, str] = field(default_factory=dict[str, str])

    def to_json(self) -> str:
        return json.dumps({"items": {k: list(v) for k, v in sorted(self.items.items())},
                           "benchmarks": dict(sorted(self.benchmarks.items())), "errors": list(self.errors),
                           "files": list(self.files), "skipped": dict(sorted(self.skipped.items())),
                           "xfail": dict(sorted(self.xfail.items()))}, indent=1)

    @staticmethod
    def from_json(text: str) -> Collection:
        held = cast(dict[str, object], json.loads(text))
        items = cast(dict[str, list[str]], held["items"])
        return Collection({k: tuple(v) for k, v in items.items()}, cast(dict[str, str], held["benchmarks"]),
                          tuple(cast(list[str], held["errors"])), tuple(cast(list[str], held["files"])),
                          cast(dict[str, str], held["skipped"]), cast(dict[str, str], held["xfail"]))


def evidence_files(evidence: Iterable[str]) -> list[str]:
    """The test files the node ids among ``evidence`` name, each once, in order."""
    found: dict[str, None] = {}
    for held in evidence:
        match = _NODE_ID.fullmatch(held)
        if match is not None:
            found[match.group("file")] = None
    return list(found)


def collect(root: Path, files: Sequence[str] | None = None) -> Collection:
    """pytest's collection of the checkout at ``root`` with every marker kept (``-m ""``), in a fresh interpreter: of
    ``files`` (paths relative to ``root``; those that do not exist are left out, and resolve as missing), or of the
    whole suite when ``files`` is ``None``. A benchmark id resolves only in a collection of the file that holds it."""
    given = [f for f in files if (root / f).is_file()] if files is not None else []
    if files is not None and not given:
        return Collection({}, {}, (), tuple(files))
    with tempfile.TemporaryDirectory() as held:
        out = Path(held) / "collection.json"
        done = subprocess.run([sys.executable, "-c", _COLLECTOR, str(out), *given], cwd=root, capture_output=True,
                              text=True, check=False)
        if not out.is_file():
            tail = (done.stdout + done.stderr).strip().splitlines()[-5:]
            return Collection({}, {}, (f"pytest's collection did not run (exit {done.returncode}): "
                                       + " / ".join(tail),), tuple(files or ()))
        data = cast(dict[str, object], json.loads(out.read_text(encoding="utf-8")))
    raw = cast(dict[str, dict[str, object]], data["items"])
    items = {nodeid: tuple(cast(list[str], held["markers"])) for nodeid, held in raw.items()}
    benchmarks = {cast(str, held["benchmark"]): nodeid for nodeid, held in raw.items() if held["benchmark"]}
    skipped = {nodeid: cast(str, held["skipped"]) for nodeid, held in raw.items() if held["skipped"] is not None}
    xfail = {nodeid: cast(str, held["xfail"]) for nodeid, held in raw.items() if held["xfail"] is not None}
    return Collection(items, benchmarks, tuple(cast(list[str], data["errors"])), tuple(files or ()), skipped, xfail)


#: A test's outcomes in the order a repeated id takes the worse of: a node id in two cases of one report (pytest writes a
#: second ``testcase`` for a passed test whose teardown errored) or in two reports (an old and a new one, read
#: together) is never taken from the one that sorts last.
_OUTCOME_ORDER = ("error", "failed", "xfailed", "skipped", "passed")


def junit_outcomes(reports: Iterable[Path]) -> dict[str, str]:
    """Each test's outcome in one lane's JUnit reports (pytest's ``--junitxml``, either family; a sharded lane's
    reports together): ``passed``, ``failed``, ``error``, ``xfailed`` (an expected failure, which JUnit writes as a skip
    of type ``pytest.xfail``) or ``skipped``, by node id. A node id that appears more than once takes the worst of its
    outcomes."""
    found: dict[str, str] = {}
    for report in reports:
        for case in ElementTree.parse(report).iter("testcase"):
            nodeid = _junit_nodeid(case.attrib)
            children = {child.tag: child.attrib for child in case}
            outcome = ("error" if "error" in children else "failed" if "failure" in children
                       else "xfailed" if children.get("skipped", {}).get("type") == "pytest.xfail"
                       else "skipped" if "skipped" in children else "passed")
            before = found.get(nodeid)
            found[nodeid] = outcome if before is None else min(before, outcome, key=_OUTCOME_ORDER.index)
    return found


def _junit_nodeid(attributes: Mapping[str, str]) -> str:
    """A JUnit test case's node id: its file (xunit1's ``file``; in xunit2, the classname's parts before the first
    capitalised one, a class), its classes, and its name."""
    classname, name = attributes.get("classname", ""), attributes.get("name", "")
    parts = classname.split(".")
    cut = next((i for i, part in enumerate(parts) if part[:1].isupper()), len(parts))
    file = attributes.get("file") or "/".join(parts[:cut]) + ".py"
    module = file.removesuffix(".py").replace("/", ".")
    rest = classname[len(module) + 1:] if classname.startswith(f"{module}.") else ""
    return "::".join([file, *(rest.split(".") if rest else []), name])


@dataclass(frozen=True)
class Resolution:
    """One evidence id resolved: the node ids it names, its lane (where CI runs it, from its marks), whether CI's lanes
    run every node it names (``runs_in_ci``) and whether one is expected to fail, its outcome in each named lane whose
    report was read (``outcomes``, ``None`` where none was), and why it does not resolve (``problem``, ``None`` when it
    does)."""

    evidence: str
    nodeids: tuple[str, ...] = ()
    lane: str | None = None
    runs_in_ci: bool = False
    expected_to_fail: bool = False
    outcomes: tuple[tuple[str, str], ...] | None = None
    problem: str | None = None

    @property
    def counts(self) -> bool:
        """Whether the id is evidence: every node it names passed in a named lane whose report was read, and none is
        expected to fail; where no report was read, every node runs in CI's lanes."""
        if self.problem is not None or self.expected_to_fail:
            return False
        if self.outcomes is None:
            return self.runs_in_ci
        return any(outcome == "passed" for _, outcome in self.outcomes)

    def text(self) -> str:
        if self.problem is not None:
            return f"{self.evidence}: {self.problem}"
        if self.outcomes is None:
            return f"{self.evidence}: {self.lane}; outcome not read (no lane report)"
        return f"{self.evidence}: {self.lane}; " + "; ".join(f"{outcome} in {lane}" for lane, outcome in self.outcomes)


def lane_of(nodeid: str, collection: Collection) -> tuple[str, bool]:
    """Where CI runs the collected test ``nodeid``, in words, and whether it runs there at all."""
    markers = set(collection.items[nodeid])
    if "skip" in markers:
        return LANES["skip"], False
    if nodeid in collection.xfail:
        return LANES["xfail"], False
    for marker in ("slow", "gpu"):
        if marker in markers:
            return LANES[marker], False
    if nodeid in collection.skipped:
        return f"{DEFAULT_LANE}, skipped on this machine ({collection.skipped[nodeid]})", True
    return DEFAULT_LANE, True


def _outcome(nodeids: Sequence[str], outcomes: Mapping[str, str]) -> str:
    found = [outcomes.get(nodeid, "not in the lane's report") for nodeid in nodeids]
    for worst in ("error", "failed", "xfailed", "not in the lane's report", "skipped"):
        if worst in found:
            return worst
    return "passed"


def _resolved(evidence: str, nodeids: tuple[str, ...], collection: Collection,
              lanes: Mapping[str, Mapping[str, str]] | None) -> Resolution:
    each = [lane_of(nodeid, collection) for nodeid in nodeids]
    outcomes = (None if lanes is None else
                tuple((name, _outcome(nodeids, held)) for name, held in lanes.items()))
    return Resolution(evidence, nodeids, " and ".join(sorted({lane for lane, _ in each})),
                      all(runs for _, runs in each), any(nodeid in collection.xfail for nodeid in nodeids), outcomes)


def resolve_one(evidence: str, root: Path, collection: Collection,
                lanes: Mapping[str, Mapping[str, str]] | None = None) -> Resolution:
    """``evidence`` against a collection of the checkout at ``root``, and each named lane's outcomes where read."""
    if BENCHMARK.fullmatch(evidence):
        nodeid = collection.benchmarks.get(evidence)
        if nodeid is None:
            return Resolution(evidence, problem="no collected test carries this verification benchmark id")
        return _resolved(evidence, (nodeid,), collection, lanes)
    match = _NODE_ID.fullmatch(evidence)
    if match is None:
        return Resolution(evidence, problem="is neither a pytest node id ('tests/<file>.py::<test>') nor a "
                                            "verification benchmark id")
    if not (root / match.group("file")).is_file():
        return Resolution(evidence, problem=f"the file {match.group('file')} does not exist")
    if match.group("params"):
        nodeids = (evidence,) if evidence in collection.items else ()
    else:
        nodeids = tuple(n for n in collection.items if n == evidence or n.startswith(f"{evidence}["))
    if not nodeids:
        errors = [e for e in collection.errors if e.startswith(match.group("file"))]
        why = f" (its file did not collect: {errors[0]})" if errors else ""
        return Resolution(evidence, problem=f"is not collected: pytest finds no test of that id{why}")
    return _resolved(evidence, nodeids, collection, lanes)


def resolve(evidence: Iterable[str], root: Path, collection: Collection | None = None, *,
            lanes: Mapping[str, Sequence[Path]] | None = None) -> dict[str, Resolution]:
    """Each id of ``evidence`` resolved against ``collection``, or against a collection of the files the ids name when
    none is given, with the outcomes of each named lane's JUnit reports in ``lanes`` (lane name to its reports)."""
    held = list(dict.fromkeys(evidence))
    if collection is None:
        collection = collect(root, evidence_files(held)) if not any(BENCHMARK.fullmatch(e) for e in held) \
            else collect(root)
    outcomes = None if lanes is None else {name: junit_outcomes(reports) for name, reports in lanes.items()}
    return {each: resolve_one(each, root, collection, outcomes) for each in held}


def lane_counts(resolutions: Iterable[Resolution]) -> str:
    """The resolved ids by lane, in words, for a summary: ``default lane 3, not run in CI (slow) 1``."""
    lanes = Counter(r.lane for r in resolutions if r.problem is None)
    return ", ".join(f"{lane} {count}" for lane, count in sorted(lanes.items()))


@dataclass(frozen=True)
class Claim:
    """A claim as its record holds it, by its key (``<owner>.<id>``)."""

    key: str
    quantity: str
    basis: str
    role: str
    says: str
    value: float | None = None
    unit: str | None = None
    oracle_kind: str | None = None
    evidence: tuple[str, ...] = ()
    conditions: Mapping[str, str] = field(default_factory=dict[str, str])
    domain: str | None = None
    n: int | None = None
    dataset: str | None = None
    held_out: str | None = None

    def label(self) -> str:
        """What the claim may be read as, printed beside its value."""
        parts = [self.role, self.basis]
        if self.oracle_kind is not None:
            parts.append(f"against {_a(self.oracle_kind)} oracle")
        if self.oracle_kind == "implementation":
            parts.append("not accuracy evidence")
        if self.role == "estimate" and self.n is not None:
            parts.append(f"from {self.n} points{f' over {self.domain}' if self.domain else ''}, bounding nothing")
        if self.role == "acceptance":
            parts.append("an acceptance test, not an accuracy claim")
        return "; ".join(parts)


def claim_problems(raw: object, owner: str) -> list[str]:
    """Every way the claim ``raw`` (of the record ``owner``) breaks the claims' shape: its fields, its basis by role,
    its oracle kind by basis, and a bound's ``domain`` and ``n``. Its evidence is resolved apart (:func:`resolve`)."""
    if not isinstance(raw, dict):
        return [f"{owner}: a claim is a mapping of its fields, not {raw!r}"]
    held = cast(dict[str, object], raw)
    where = f"{owner}.{held.get('id', '<no id>')}"
    found = [f"{where}: unknown field {key!r} (a claim holds {', '.join(CLAIM_FIELDS)})"
             for key in held if key not in CLAIM_FIELDS]
    found += [f"{where}: missing field {key!r}" for key in _REQUIRED if key not in held]
    for key in ("id", "quantity", "says", "unit", "domain", "dataset", "held_out", "basis", "role", "oracle_kind"):
        if key in held and not (isinstance(held[key], str) and held[key]):
            found.append(f"{where}: {key} is {held[key]!r}, not a text")
    if isinstance(held.get("id"), str) and not _CLAIM_ID.fullmatch(cast(str, held["id"])):
        found.append(f"{where}: the id is lower-case letters, digits and underscores, starting with a letter")
    if isinstance(held.get("says"), str) and "\n" in cast(str, held["says"]).strip():
        found.append(f"{where}: says is one line of what the claim bounds and what it estimates")
    basis, role, kind = held.get("basis"), held.get("role"), held.get("oracle_kind")
    if basis is not None and basis not in BASES:
        found.append(f"{where}: basis {basis!r} is not one of {', '.join(BASES)}")
    if role is not None and role not in ROLES:
        found.append(f"{where}: role {role!r} is not one of {', '.join(ROLES)}")
    if kind is not None and kind not in ORACLE_KINDS:
        found.append(f"{where}: oracle_kind {kind!r} is not one of {', '.join(ORACLE_KINDS)}")
    evidence = held.get("evidence", [])
    if not (isinstance(evidence, list) and all(isinstance(e, str) and e for e in cast(list[object], evidence))):
        found.append(f"{where}: evidence is a list of test ids")
        evidence = []
    value = held.get("value")
    if "value" in held and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
        found.append(f"{where}: value is {value!r}, not a finite number")
    if "n" in held and (isinstance(held["n"], bool) or not isinstance(held["n"], int) or held["n"] < 1):
        found.append(f"{where}: n is {held['n']!r}, not a count of points")
    conditions = held.get("conditions")
    if conditions is not None and not (isinstance(conditions, dict) and len(cast(dict[object, object], conditions)) == 1
                                       and next(iter(cast(dict[object, object], conditions))) in ("point", "region")
                                       and isinstance(next(iter(cast(dict[object, object], conditions).values())),
                                                      str)):
        found.append(f"{where}: conditions are a point or a region, '{{point: <text>}}' or '{{region: <text>}}'")
    if basis in BASES and role in ROLES:
        found += _matrix_problems(where, cast(str, basis), cast(str, role), cast(str | None, kind), held,
                                  cast(list[str], evidence))
    return found


def _matrix_problems(where: str, basis: str, role: str, kind: str | None, held: Mapping[str, object],
                     evidence: Sequence[str]) -> list[str]:
    found: list[str] = []
    if role not in ROLES_BY_BASIS[basis]:
        allowed = ", ".join(sorted(ROLES_BY_BASIS[basis]))
        found.append(f"{where}: a {basis} claim is not {_a(role)} (a {basis} claim is {_a(allowed)})"
                     + ("; an acceptance grounds sign, frame and units against a measurement, so it is measured"
                        if role == "acceptance" else ""))
    if basis == "declared":
        if kind is not None or evidence:
            found.append(f"{where}: a declared claim is covered by no test, so it names no oracle and no evidence; "
                         f"a claim a test covers is computed, measured or fitted")
    else:
        if kind is None:
            found.append(f"{where}: a {basis} claim names its oracle_kind ({', '.join(sorted(KINDS_BY_BASIS[basis]))})")
        elif kind in ORACLE_KINDS and kind not in KINDS_BY_BASIS[basis]:
            found.append(f"{where}: a {basis} claim is not compared against {_a(kind)} oracle "
                         f"({', '.join(sorted(KINDS_BY_BASIS[basis]))})")
        if not evidence:
            found.append(f"{where}: a {basis} claim names its evidence, the tests that compute it")
    if kind == "implementation" and role in ("bound", "acceptance"):
        found.append(f"{where}: an implementation oracle evaluates the code's own formula again, which is not "
                     f"accuracy evidence, so it is never the oracle of a {role}")
    if role == "bound":
        missing = [key for key in ("domain", "n", "value") if key not in held]
        if missing:
            found.append(f"{where}: a bound carries the domain it holds over, n, the points it rests on, and its "
                         f"value; it lacks {', '.join(missing)}")
        if "held_out" not in held:
            found.append(f"{where}: a {basis} claim is a bound only on held-out evidence (held_out, an id not among "
                         f"its evidence): a maximum over its points bounds nothing beyond them, so without it the "
                         f"claim is an estimate")
    if isinstance(held.get("held_out"), str) and held["held_out"] in evidence:
        found.append(f"{where}: held_out {held['held_out']!r} is among its own evidence, so nothing is held out")
    if role == "estimate" and "value" not in held:
        found.append(f"{where}: an estimate carries its value")
    if basis == "fitted" and "dataset" not in held:
        found.append(f"{where}: a fitted claim names the fit's dataset")
    return found


def _a(word: str) -> str:
    return f"an {word}" if word[:1] in "aeiou" else f"a {word}"


def claim_of(raw: Mapping[str, object], owner: str) -> Claim:
    """The claim ``raw`` of the record ``owner``, read once :func:`claim_problems` finds nothing."""
    value = raw.get("value")
    return Claim(
        f"{owner}.{raw['id']}", cast(str, raw["quantity"]), cast(str, raw["basis"]), cast(str, raw["role"]),
        cast(str, raw["says"]).strip(), float(cast(float, value)) if value is not None else None,
        cast(str | None, raw.get("unit")), cast(str | None, raw.get("oracle_kind")),
        tuple(cast(list[str], raw.get("evidence", []))), cast(dict[str, str], raw.get("conditions", {})),
        cast(str | None, raw.get("domain")), cast(int | None, raw.get("n")), cast(str | None, raw.get("dataset")),
        cast(str | None, raw.get("held_out")))


def read_claims(registry: Path) -> dict[str, Claim]:
    """Every claim of the anomaly registry at ``registry``, by its id (``MIME-ANO-114.max_error``). Raises
    ``ValueError`` naming a malformed one."""
    data = cast(dict[str, object], yaml.safe_load(registry.read_text(encoding="utf-8")))
    found: dict[str, Claim] = {}
    for anomaly in cast(list[dict[str, object]], data.get("anomalies", [])):
        owner = str(anomaly.get("anomaly_id"))
        for raw in cast(list[object], anomaly.get("claims", [])):
            problems = claim_problems(raw, owner)
            if problems:
                raise ValueError("; ".join(problems))
            held = claim_of(cast(dict[str, object], raw), owner)
            found[held.key] = held
    return found


def claim(claim_id: str, registry: Path) -> Claim:
    """The claim ``claim_id`` of the registry at ``registry``: what a test asserts against, so that the number is held
    once, in the record. Raises ``KeyError`` naming the ids it holds."""
    held = read_claims(registry)
    if claim_id not in held:
        raise KeyError(f"no claim {claim_id!r} in {registry} (it holds {', '.join(sorted(held)) or 'none'})")
    return held[claim_id]


def claim_rows(claims: Iterable[Claim], resolutions: Mapping[str, Resolution]) -> list[tuple[str, ...]]:
    """The claims as a verification table, one row each, its labels as columns: id, quantity, value, role, basis,
    oracle kind, evidence (each with its lane and outcome, a held-out test last), and what it bounds and estimates."""
    rows: list[tuple[str, ...]] = []
    for held in sorted(claims, key=lambda c: (ROLES.index(c.role), c.key)):
        value = "" if held.value is None else f"{held.value:g}{f' {held.unit}' if held.unit else ''}"
        evidence = "; ".join(resolutions[e].text() if e in resolutions else e for e in held.evidence) or "none"
        if held.held_out is not None:
            out = resolutions[held.held_out].text() if held.held_out in resolutions else held.held_out
            evidence += f"; held out: {out}"
        rows.append((held.key, held.quantity, value, held.role, held.basis, held.oracle_kind or "none",
                     evidence, held.label() + f". {held.says}"))
    return rows


CLAIM_COLUMNS = ("Claim", "Quantity", "Value", "Role", "Basis", "Oracle", "Evidence", "What it bounds and estimates")


def components(records: Iterable[Any]) -> dict[str, Any]:
    """What an anomaly's ``affected_components`` can name, from the registered entries ``records``
    (``mime.registry.registered()``), as JSON: each entry's kind and name with its rungs, reserved ones included, the
    names a legacy spelling is held to (entries' names and classes, and the node classes they build), and those node
    classes."""
    entries: dict[str, set[str]] = {}
    names: set[str] = set()
    nodes: set[str] = set()
    for record in records:
        cls = record.cls
        held = entries.setdefault(f"{record.kind.value.replace(' ', '_')} {record.name}", set())
        held.update(getattr(cls, "rungs", ()) or ())
        held.update(getattr(cls, "reserved_rungs", ()) or ())
        names |= {record.name, cls.__name__}
        nodes |= {node.__name__ for built in dict(cls.builds).values() for node in built}
    return {"entries": {key: sorted(rungs) for key, rungs in sorted(entries.items())},
            "names": sorted(names | nodes), "nodes": sorted(nodes)}

__all__ = ["BASES", "BENCHMARK", "CLAIM_COLUMNS", "CLAIM_FIELDS", "DEFAULT_LANE", "KINDS_BY_BASIS", "LANES", "NOT_RUN",
           "ORACLE_KINDS", "ROLES", "ROLES_BY_BASIS", "Claim", "Collection", "Resolution", "claim", "claim_of",
           "claim_problems", "claim_rows", "collect", "components", "evidence_files", "junit_outcomes", "lane_counts",
           "lane_of", "read_claims", "resolve", "resolve_one"]
