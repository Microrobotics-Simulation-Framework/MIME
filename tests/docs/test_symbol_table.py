"""The symbol-table checks, S1 to S6, and the ``hazard_hints`` harvest, N3, on the test-only nodes of
``planted_nodes.py``, since no MIME node carries ``NodeMeta`` yet. Every control runs the real check on a planted copy
of the clean node, against list files written to a temporary directory, and fails naming what is wrong; every passing
arm passes. The clean node passes every check, so a failure below is the plant's.

S1 renames a parameter; S2 changes a units string on an input, a parameter and a published state; S3 deletes the one
token of a single letter that a substring test would still find inside ``\\frac``; S4 points a row at its class; S5
holds ``implemented_by`` to the class that defines a method, among the node's own class and its bases, and holds the
reverse lint to braced subscripts; S6 holds the floor of pinned rows, node by node."""

from dataclasses import replace
from pathlib import Path

import pytest
from planted_nodes import EQUATIONS, META, ROWS, PlantedNode, planted, with_equations, with_row

from mime.core import MimeNode, SymbolKind
from mime.docs import (
    instance_problems,
    next_id,
    read_frozen_ids,
    read_list,
    row_key,
    static_problems,
    symbols_of,
    tokens,
)
from mime.docs._check import LISTS
from mime.registry import DimensionedSetting
from mime.units import DYNAMIC_VISCOSITY
from mime.vocabulary import APPLIED_WRENCH

FROZEN = {"MIME-NODE-012": "GravityNode", "MIME-NODE-101": "PermanentMagnetNode"}
SETTINGS = {"viscosity": DimensionedSetting(DYNAMIC_VISCOSITY)}
PUBLISHED = {"force": APPLIED_WRENCH.units}
#: Every row of the clean node pinned, as its table's landing pins it.
PINNED = "".join(f"{row_key(PlantedNode, row)}  # pinned at landing\n" for row in ROWS)
#: The rows whose units the code does not declare: a parameter with no units, two states no entry publishes, a constant.
UNCHECKED = "".join(f"{row_key(PlantedNode, ROWS[i])}  # declared, not checked\n" for i in (4, 5, 6, 7))


def lists(tmp_path: Path, rows: str = PINNED, indices: str = "", units: str = UNCHECKED):
    for name, text in (("symbol_rows", rows), ("symbol_indices", indices), ("symbol_units", units)):
        (tmp_path / f"{name}.txt").write_text(text, encoding="utf-8")
    return {name: read_list(tmp_path / f"{name}.txt", LISTS[name])
            for name in ("symbol_rows", "symbol_indices", "symbol_units")}


def static(tmp_path: Path, *nodes: type, rows: str = PINNED, indices: str = "") -> list[str]:
    held = lists(tmp_path, rows, indices)
    return list(static_problems(nodes, rows=held["symbol_rows"], indices=held["symbol_indices"],
                                frozen=FROZEN).problems)


def instance(tmp_path: Path, node: object, *, published=PUBLISHED, units: str = UNCHECKED) -> list[str]:
    held = lists(tmp_path, units=units)
    return instance_problems(node, settings=SETTINGS, published=published, units=held["symbol_units"])


def test_the_clean_node_passes_every_check(tmp_path):
    assert static(tmp_path, PlantedNode) == []
    assert instance(tmp_path, PlantedNode()) == []


# -- S1, S2: names and units ------------------------------------------------------------------------------------------

def test_s1_a_renamed_parameter_fails_naming_the_row(tmp_path):
    renamed = planted("PlantedNode", param_specs=lambda self: {"drag_coefficient": PlantedNode().param_specs()["drag"],
                                                                "length": PlantedNode().param_specs()["length"]})
    (problem,) = instance(tmp_path, renamed())
    assert problem.startswith("PlantedNode: the row 'c': parameter 'drag' is not declared")


def test_s2_a_changed_input_unit_fails_naming_both(tmp_path):
    moved = planted("PlantedNode", boundary_input_spec=lambda self: {
        "velocity": replace(PlantedNode().boundary_input_spec()["velocity"], expected_units="m")})
    assert instance(tmp_path, moved()) == [
        "PlantedNode: the row '\\\\mathbf{v}': units 'm/s', and the code declares 'm'"]


def test_s2_a_changed_parameter_unit_fails(tmp_path):
    moved = planted("PlantedNode", symbols=with_row(1, units="kg"))
    assert instance(tmp_path, moved()) == ["PlantedNode: the row 'c': units 'kg', and the code declares 'kg s^-1'"]


def test_s2_a_published_state_is_held_to_its_vocabulary_row(tmp_path):
    assert instance(tmp_path, PlantedNode(), published={"force": "N"}) == [
        "PlantedNode: the row '\\\\mathbf{F}': units 'N; N m', and the code declares 'N'"]


def test_s2_a_unit_the_code_does_not_declare_fails_until_listed_and_a_listed_one_now_checked_fails(tmp_path):
    (unlisted,) = instance(tmp_path, PlantedNode(), units=UNCHECKED.replace(f"{row_key(PlantedNode, ROWS[5])}  # "
                                                                            f"declared, not checked\n", ""))
    assert unlisted.startswith("PlantedNode L: a row whose units the code does not declare")
    (stale,) = instance(tmp_path, PlantedNode(), units=UNCHECKED + f"{row_key(PlantedNode, ROWS[2])}  # stale\n")
    assert stale.endswith("'PlantedNode \\\\mathbf{v}' now passes; delete this line")


# -- S3, S4 -------------------------------------------------------------------------------------------------------

def test_s3_a_single_letter_deleted_from_the_equation_fails_where_a_substring_test_would_pass(tmp_path):
    doctored = EQUATIONS.replace(r"\, a \,", r"\,")
    assert doctored != EQUATIONS and "a" in doctored, "the letter must survive, inside \\frac, for the plant to count"
    assert [t.text for t in symbols_of(doctored)].count("a") == 0
    (problem,) = static(tmp_path, planted("PlantedNode", meta=with_equations(doctored)))
    assert problem == "PlantedNode: the row 'a': a does not occur in governing_equations"


def test_s4_a_row_pointed_at_its_class_fails(tmp_path):
    pointed = planted("PlantedNode", symbols=with_row(0, implemented_by="planted_nodes.PlantedNode"))
    (problem,) = static(tmp_path, pointed)
    assert "resolves to the class PlantedNode, not to a function, method or property" in problem


# -- S5: the path, and the reverse lint ------------------------------------------------------------------------------

@pytest.mark.parametrize(("path", "failure"), [
    ("planted_nodes.PlantedNode.base_helper", "'base_helper' is not defined on PlantedNode; it resolves only through "
                                              "base class PlantedBase"),
    ("planted_nodes.Unrelated.helper",
     "names a method of Unrelated, which is neither PlantedNode nor one of its bases"),
    ("planted_nodes.PlantedSubNode.extra", "names a method of PlantedSubNode, which is neither PlantedNode nor one of "
                                           "its bases"),
    ("planted_nodes.Unrelated", "resolves to the class Unrelated, not to a function, method or property"),
], ids=["a base's method by the subclass's path", "an unrelated class's method", "a subclass's own method",
        "a module-level class"])
def test_s5_a_row_naming_a_function_where_it_is_not_defined_fails(tmp_path, path, failure):
    (problem,) = static(tmp_path, planted("PlantedNode", symbols=with_row(5, implemented_by=path)))
    assert problem.startswith("PlantedNode: the row 'L''s implemented_by: ") and failure in problem


@pytest.mark.parametrize("path", ["planted_nodes.PlantedBase.base_helper", "planted_nodes.drag_law",
                                  "planted_nodes.PlantedNode.advance"],
                         ids=["a base's method by the base's path", "a module-level function", "its own method"])
def test_s5_a_row_naming_a_function_where_it_is_defined_passes(tmp_path, path):
    assert static(tmp_path, planted("PlantedNode", symbols=with_row(5, implemented_by=path))) == []


def test_s5_a_braced_subscript_needs_no_line(tmp_path):
    assert "x_{i}" in EQUATIONS
    assert static(tmp_path, PlantedNode, indices="") == []
    assert [(t.text, t.subscript) for t in tokens("x_{i}")] == [("x", False), ("i", True)]


def test_s5_an_unbraced_subscript_fails_without_its_line_and_passes_with_it(tmp_path):
    node = planted("PlantedNode", meta=with_equations(EQUATIONS.replace("x_{i}", "x_i")))
    (problem,) = static(tmp_path, node)
    assert problem.startswith("PlantedNode i: an equation symbol with no row")
    assert static(tmp_path, node, indices="PlantedNode i  # the index of x, unbraced\n") == []


@pytest.mark.parametrize(("extra", "missing"), [("^{z}", ["z"]), (r"\frac{z}{w}", ["w", "z"])],
                         ids=["a superscript group", "a fraction's groups"])
def test_s5_a_symbol_in_a_braced_group_that_is_not_a_subscript_needs_a_row(tmp_path, extra, missing):
    problems = static(tmp_path, planted("PlantedNode", meta=with_equations(f"{EQUATIONS} + y{extra}")))
    assert sorted(p.split(":")[0] for p in problems) == sorted(f"PlantedNode {s}" for s in ["y", *missing])


def test_s5_a_listed_index_that_now_has_a_row_fails(tmp_path):
    (problem,) = static(tmp_path, PlantedNode, indices="PlantedNode x  # once an index\n")
    assert problem.endswith("'PlantedNode x' now passes; delete this line")


# -- S6: the floor --------------------------------------------------------------------------------------------------

#: What deleting the row of ``x`` from the clean node's table fails on: the floor, and the reverse lint, since ``x``
#: stays in the equations with no row.
X_DELETED = ["the pinned symbol row 'PlantedNode x' is missing; restore it, or delete the line with its reason logged",
             "PlantedNode x: an equation symbol with no row"]


def matches(problems: list[str], expected: list[str]) -> bool:
    return len(problems) == len(expected) and all(any(e in p for p in problems) for e in expected)


def test_s6_a_pinned_row_deleted_from_its_table_fails_naming_the_node_and_the_symbol(tmp_path):
    assert matches(static(tmp_path, planted("PlantedNode", symbols=ROWS[:-1])), X_DELETED)


def test_s6_the_row_pinned_and_kept_passes(tmp_path):
    assert static(tmp_path, PlantedNode) == []


def test_s6_another_node_holding_the_symbol_does_not_hold_the_floor(tmp_path):
    other = planted("OtherNode", base=PlantedNode, meta=replace(META, algorithm_id="MIME-NODE-902"))
    pinned = PINNED + "".join(f"{row_key(other, row)}  # pinned at landing\n" for row in ROWS)
    assert static(tmp_path, PlantedNode, other, rows=pinned) == []
    assert matches(static(tmp_path, planted("PlantedNode", symbols=ROWS[:-1]), other, rows=pinned), X_DELETED)


def test_s6_a_pinned_row_replaced_by_a_filler_fails(tmp_path):
    filler = (*ROWS[:-1], replace(ROWS[-1], latex="X"))
    (problem,) = static(tmp_path, planted("PlantedNode", symbols=filler,
                                          meta=with_equations(EQUATIONS.replace("x_{i}", "X_{i}"))))
    assert "the pinned symbol row 'PlantedNode x' is missing" in problem


def test_s6_a_renamed_symbol_is_a_deletion_and_a_pin(tmp_path):
    renamed = planted("PlantedNode", symbols=(*ROWS[:-1], replace(ROWS[-1], latex="y")),
                      meta=with_equations(EQUATIONS.replace("x_{i}", "y_{i}")))
    (problem,) = static(tmp_path, renamed)
    assert "the pinned symbol row 'PlantedNode x' is missing" in problem
    repinned = PINNED.replace("PlantedNode x  # pinned at landing", "PlantedNode y  # x renamed y")
    assert static(tmp_path, renamed, rows=repinned) == []


def test_s6_a_table_with_no_row_pinned_fails(tmp_path):
    (problem,) = static(tmp_path, PlantedNode, rows="")
    assert problem.startswith("PlantedNode: none of its symbol rows is pinned")


# -- NodeMeta -----------------------------------------------------------------------------------------------------

def test_n3_hazard_hints_set_on_a_node_fails_the_harvest(tmp_path):
    (problem,) = static(tmp_path, planted("PlantedNode", meta=replace(META, hazard_hints=("CFL > 1",))))
    assert problem == "PlantedNode: NodeMeta.hazard_hints is set (['CFL > 1']); MIME's nodes leave it empty"


@pytest.mark.parametrize(("algorithm_id", "failure"), [
    ("", "NodeMeta.algorithm_id is empty; give it the next id, MIME-NODE-102"),
    ("MIME-NODE-012", "algorithm_id MIME-NODE-012 was the 0.2 line's GravityNode's"),
], ids=["empty", "a 0.2 node's"])
def test_an_algorithm_id_missing_or_taken_from_another_node_fails(tmp_path, algorithm_id, failure):
    (problem,) = static(tmp_path, planted("PlantedNode", meta=replace(META, algorithm_id=algorithm_id)))
    assert failure in problem


def test_a_node_taking_its_own_0_2_id_passes_and_two_nodes_sharing_an_id_fail(tmp_path):
    gravity = planted("GravityNode", meta=replace(META, algorithm_id="MIME-NODE-012"))
    pinned = PINNED + "".join(f"{row_key(gravity, row)}  # pinned at landing\n" for row in ROWS)
    assert static(tmp_path, PlantedNode, gravity, rows=pinned) == []
    twin = planted("TwinNode", meta=META)
    pinned = PINNED + "".join(f"{row_key(twin, row)}  # pinned at landing\n" for row in ROWS)
    (problem,) = static(tmp_path, PlantedNode, twin, rows=pinned)
    assert problem == "TwinNode: algorithm_id MIME-NODE-901 is PlantedNode's too"


def test_an_order_declared_by_the_hook_and_in_nodemeta_fails(tmp_path):
    from maddening.core.compliance.metadata import DiscretizationOrder

    twice = planted("PlantedNode", meta=replace(META, discretization_order=DiscretizationOrder(temporal=1.0)),
                    discretization_order=lambda self: DiscretizationOrder(temporal=1.0))
    (problem,) = static(tmp_path, twice)
    assert "the order is declared by its discretization_order() and again in NodeMeta" in problem


def test_a_table_without_nodemeta_fails(tmp_path):
    (problem,) = static(tmp_path, planted("PlantedNode", meta=None))
    assert problem == "PlantedNode: it has a symbol table and no NodeMeta, so no equations for the rows to name"


def test_a_node_with_neither_is_counted_undocumented(tmp_path):
    held = lists(tmp_path, rows="")
    report = static_problems([MimeNode], rows=held["symbol_rows"], indices=held["symbol_indices"], frozen=FROZEN)
    assert report.problems == () and report.undocumented == ("MimeNode",)


def test_the_tokeniser_reads_macros_then_letters_outside_words():
    assert [t.text for t in symbols_of(r"\mathbf{F} = -6 \pi \mu a \, \text{drag} \frac{\partial u}{\partial t}")] \
        == ["F", "\\mu", "a", "u", "t"]
    assert [t.text for t in symbols_of(r"\begin{pmatrix} x \end{pmatrix} \mathrm{Re}")] == ["x"]


def test_the_constant_row_names_a_module_constant(tmp_path):
    bad = planted("PlantedNode", symbols=with_row(4, name="mime.units.si"))
    assert instance(tmp_path, bad()) == ["PlantedNode: the row 'a': 'mime.units.si' is not a constant"]
    assert SymbolKind.CONSTANT is ROWS[4].kind


def test_the_frozen_list_holds_the_0_2_lines_twenty_ids_and_the_next_is_computable():
    frozen = read_frozen_ids(Path(__file__).resolve().parents[2] / "docs/data/node_ids_0.2.yaml")
    assert len(frozen) == len(set(frozen.values())) == 20
    assert frozen["MIME-NODE-012"] == "GravityNode" and next_id(frozen, ()) == "MIME-NODE-104"
