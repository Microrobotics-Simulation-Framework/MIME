"""The Capabilities table of a geometry kind's entry page says what the kind implements.

The oracle is the kind: for each registered geometry kind, whether a class of its own (between it and ``Geometry``, and
for a solid's mass properties between it and ``Solid``) defines the method, read from the class dictionaries, and
whether its ``native`` is set. That is not ``mime.registry.implements``, which the export and so the page use; the
page is rendered from the export's record in this process and each row's last cell is read back.

The controls: the table with its yes and no swapped in the generator (the page's source, edited and run), a record
that exports the wrong answer for one capability, and a kind that gains a method each fail it, naming the row.
"""

import inspect
import textwrap

import pytest

from mime.docs import _entry_page
from mime.docs._context import Context
from mime.docs._export import entry_record, load_namespaces
from mime.registry import CAPABILITIES, EntryKind, Geometry, Solid, registered

EMPTY = Context((), {}, {}, {})


def defined_by_the_kind(cls: type[Geometry], name: str) -> bool:
    """Whether ``cls`` itself, rather than the base it inherits the unimplemented method from, gives ``name``."""
    base = Solid if name == "mass_properties" else Geometry
    method = "volume" if name == "mass_properties" else name
    if name == "mass_properties" and not issubclass(cls, Solid):
        return False
    for klass in cls.__mro__:
        if klass is base:
            return False
        if method in vars(klass):
            return getattr(klass, method) is not None if name == "native" else True
    return False


def capability_cells(text: str) -> dict[str, str]:
    """The Capabilities table's rows of a page: the capability's name against its last cell."""
    section = text.split("## Capabilities", 1)[1].split("\n## ", 1)[0]
    rows = [line.strip("|").split("|") for line in section.splitlines() if line.startswith("| `")]
    return {cells[0].strip().strip("`"): cells[-1].strip() for cells in rows}


@pytest.fixture(scope="module")
def kinds():
    load_namespaces()
    found = {record.name: (record.cls, entry_record(record)) for record in registered(EntryKind.GEOMETRY)}
    assert len(found) >= 6, "the walk would pass over the kinds"
    return found


def wrong_rows(kinds) -> list[str]:
    found: list[str] = []
    for name, (cls, record) in sorted(kinds.items()):
        cells = capability_cells(_entry_page.entry_page(record, EMPTY).text)
        if set(cells) != set(CAPABILITIES):
            found.append(f"{name}: rows {sorted(cells)}, not {sorted(CAPABILITIES)}")
        for capability in CAPABILITIES:
            expected = "yes" if defined_by_the_kind(cls, capability) else "no"
            if cells.get(capability) != expected:
                found.append(f"{name}: {capability} is {cells.get(capability)!r}, and the kind {expected!r}")
    return found


def test_every_geometry_kinds_capability_rows_say_which_methods_it_implements(kinds):
    assert wrong_rows(kinds) == []
    tables = {name: capability_cells(_entry_page.entry_page(record, EMPTY).text) for name, (_, record) in kinds.items()}
    held = {cell for cells in tables.values() for cell in cells.values()}
    assert held == {"yes", "no"}, "the kinds do not differ in a capability, so a swapped table would pass"
    print("\n" + "\n".join(f"{name}: " + ", ".join(c for c, v in cells.items() if v == "yes")
                           for name, cells in sorted(tables.items())))


def test_control_the_table_with_its_yes_and_no_swapped_in_the_generator_fails(kinds, monkeypatch):
    original = textwrap.dedent(inspect.getsource(_entry_page._capabilities))
    planted = original.replace('"yes" if held[name] else "no"', '"no" if held[name] else "yes"')
    assert planted != original
    scope = dict(vars(_entry_page))
    exec(planted, scope)  # noqa: S102 - the page generator's own source, edited
    monkeypatch.setattr(_entry_page, "_capabilities", scope["_capabilities"])
    problems = wrong_rows(kinds)
    print(f"\n{problems[:3]}")
    assert len(problems) >= len(kinds)


def test_control_an_export_that_gets_one_capability_wrong_fails(kinds, monkeypatch):
    name, (cls, record) = sorted(kinds.items())[0]
    wrong = {**kinds, name: (cls, {**record, "capabilities": {**record["capabilities"], "drawing": not
                                                              record["capabilities"]["drawing"]}})}
    problems = wrong_rows(wrong)
    print(f"\n{problems}")
    assert len(problems) == 1 and "drawing" in problems[0]


def test_control_a_kind_that_gains_a_method_is_seen_by_the_oracle():
    class Gains(Geometry):
        def tangent(self, parameters, point):
            return point

    class Inherits(Gains):
        pass

    assert defined_by_the_kind(Gains, "tangent") and defined_by_the_kind(Inherits, "tangent")
    assert not defined_by_the_kind(Gains, "samples") and not defined_by_the_kind(Inherits, "drawing")
