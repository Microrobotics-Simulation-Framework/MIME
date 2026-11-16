"""Every unit label MIME writes parses, collected from where MIME writes them and never from a list kept by hand.

The sources: the symbol table and the typed-state ``Units`` of every node class a registered entry builds; the units of
every interface of the vocabulary; and every ``units=``, ``expected_units=``, ``output_units=`` keyword, ``Units(...)`` and
``node_input(...)`` call with a string in ``src/mime`` (a node's ``ParamSpec`` and input and output specs). The test prints how many labels
each source gave and how many distinct strings there are (run with ``-s`` to see them), and holds each count to the floor
it was first measured at, so that a source that quietly stops finding labels fails.

A label that is not a unit is declared, with its reason, in ``NOT_A_UNIT``: a registration's interface carries no values,
and a symbol row's units are prose where they depend on the command it sits in. A declared exemption that no source holds
any more fails too, so the list cannot outlive its rows. The control adds a typo to the collected set, and the check names it."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

import mime
from mime.core._field_kinds import _states_near, _typed, _units_of  # noqa: PLC2701 - the typed state's own reader
from mime.docs._check import node_classes
from mime.docs._symbols import rows_of
from mime.registry import registered
from mime.units import UnitError, parse_label
from mime.vocabulary import lookup, names

PACKAGE = Path(mime.__file__).parent
SPEC_KEYWORDS = frozenset({"units", "expected_units", "output_units"})
#: The calls whose first positional argument is a unit label: a state field's ``Units`` and a node's ``node_input``.
POSITIONAL_LABEL = frozenset({"Units", "node_input"})

#: Strings a symbol row holds as its units that are no unit label, each with why. A row's units are read by
#: ``mime.docs`` as a label where they parse and as text where they do not (``_same_units``).
NOT_A_UNIT = {
    "": "a registration's units (``flow.partners``, ``flow.resolved``): it carries no values, so no units",
    "the command's own": "a drive's value, in whatever units its command is declared in",
    "the command's own per second": "a ramp's rate, in its command's units per second",
}

#: What each source gave when this test was written (the first run of the collection, 2026-11-15; ``typed state`` counts
#: a TypedDict once for every node class that reads it, so it is the largest). A count that falls is a source that
#: stopped finding labels, or labels that were removed: lower the floor with the reason where it is.
FLOORS = {"symbol rows": 240, "typed state": 1377, "vocabulary": 31, "source": 417, "distinct": 48}


def rows_and_typed() -> tuple[list[str], list[str]]:
    rows: list[str] = []
    typed: list[str] = []
    for cls in sorted(node_classes(registered()), key=lambda cls: (cls.__module__, cls.__name__)):
        rows += [row.units for row in rows_of(cls)]
        for state in _states_near(cls):
            held = _typed(state)
            if held is not None:
                typed += [units for hint in held[0].values() if (units := _units_of(hint)) is not None]
    return rows, typed


def of_the_vocabulary() -> list[str]:
    return [interface.units for name in names() if (interface := lookup(name)) is not None]


def of_the_source() -> list[str]:
    found: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            named = isinstance(node.func, ast.Name) and node.func.id in POSITIONAL_LABEL and node.args
            held = [node.args[0]] if named else []
            held += [kw.value for kw in node.keywords if kw.arg in SPEC_KEYWORDS]
            found += [each.value for each in held if isinstance(each, ast.Constant) and isinstance(each.value, str)]
    return found


def unreadable(labels: set[str]) -> dict[str, str]:
    """Each label that does not parse, with what is said, apart from those ``NOT_A_UNIT`` declares."""
    said: dict[str, str] = {}
    for label in sorted(labels - NOT_A_UNIT.keys()):
        try:
            parse_label(label)
        except UnitError as error:
            said[label] = str(error)
    return said


@pytest.fixture(scope="module")
def collected() -> dict[str, list[str]]:
    rows, typed = rows_and_typed()
    held = {"symbol rows": rows, "typed state": typed, "vocabulary": of_the_vocabulary(), "source": of_the_source()}
    held["distinct"] = sorted({label for each in held.values() for label in each})
    print("\nunit labels collected: " + ", ".join(f"{source} {len(each)}" for source, each in held.items()))
    return held


def test_every_unit_label_mime_writes_parses_and_the_counts_are_printed(collected):
    counts = {source: len(each) for source, each in collected.items()}
    assert {source: count for source, count in counts.items() if count < FLOORS[source]} == {}, counts
    assert unreadable(set(collected["distinct"])) == {}


def test_each_declared_exemption_is_one_a_source_still_holds(collected):
    held = Counter(label for source in ("symbol rows", "vocabulary") for label in collected[source])
    assert {label: why for label, why in NOT_A_UNIT.items() if not held[label]} == {}


def test_the_control_a_typo_among_the_collected_labels_is_named(collected):
    for typo in ("m/s^", "kPa", "N s/m s", "m)"):
        said = unreadable({*collected["distinct"], typo})
        assert list(said) == [typo], (typo, said)
