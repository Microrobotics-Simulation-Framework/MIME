"""OpenMaterica's table-evaluation cases (its read-side contract, section 7; ``CASES.md`` beside the files) against
``mime.core.tables.evaluate``, each case's table converted through the materials seam (``evaluator_table``), as a
record's table reaches the evaluator.

**Configurations.** Every case that applies runs in binary64 (with ``jax_enable_x64``) and in binary32 (without it,
the case's numbers first rounded to the nearest float32), each compared with its own relative tolerance (1e-12 and the
file's), with each side declared at an interior knot, and both with and without a declared constant extrapolation.
Without one a read outside the readable range returns no number (NaN, which the cases accept, and MIME requires); with
one it returns the end value and a zero derivative, which the cases then compare. The derivative is JAX's, of the value
in the query.

**Skipped**: none of the 35 files. The two whose readable range is one point (TE-007 ``one_point_range_end`` and
``one_point_range_inside``) are read as the specification says: the value there, no derivative (NaN). No case is
TE-001 (undefined) or TE-009 (a power law, which no 0.1.0 field takes), so the files are all 35.

**The controls**, each a planted fault run against every case in every configuration: the side not declared (the
derivative at an interior knot from the other side); the 0.2 line's clip (the query clipped into the knots, and every
read reported in range); the cell one below the right one; a slope kept where the range is one point; and OpenMaterica's
edge faults (a read at an edge unrefused, a refusal that returns a number or an infinity, an unrefused read across an
edge that returns a number, a refusal where none is due) with a value shifted by a constant, which only the value
comparison sees. Each must fail at least one case file in every configuration, the files OpenMaterica names for that
fault among them: the cases are fixed, so each run detects it. Between them they reach every stage of ``problems``:
the refusal expected, the refusal not due, ``in_range``, the value, the slope, and the rule for outside the range.
A last fault no case can catch, since the cases accept a number or NaN outside the range: the clipped end value
returned, and reported out of range, with no extrapolation declared. MIME's own rule catches it."""

import dataclasses
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import materica
import numpy as np
import pytest

import mime.core.tables._evaluate as evaluate_module
from mime.core import switch
from mime.core.tables import ConstantExtrapolation, Evaluation, Table, evaluate
from mime.materials import evaluator_table

CASES = Path(materica.__file__).parent / "data" / "conformance" / "table_evaluation" / "cases"
FILES = sorted(CASES.glob("TE-*.json"))
RUN = FILES
OWNER = "toy.table"
PRECISIONS = {"binary64": np.float64, "binary32": np.float32}
SIDES = ("right", "left")
EXTRAPOLATIONS = (False, True)
CONFIGURATIONS = [(precision, side, extrapolated) for precision in PRECISIONS for side in SIDES
                  for extrapolated in EXTRAPOLATIONS]

type Evaluator = Callable[[Table, np.ndarray], tuple[Evaluation, np.ndarray]]


def case(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def precision(name: str) -> Iterator[type]:
    """``jax_enable_x64`` on for binary64 and off for binary32, restored after."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", name == "binary64")
    try:
        yield PRECISIONS[name]
    finally:
        jax.config.update("jax_enable_x64", held)


def table_of(request: dict, dtype: type, side: str, extrapolated: bool) -> Table:
    """The case's table through the seam, its numbers first rounded to ``dtype``."""
    held = dict(request["table"])
    for member in ("x", "y", "edges", "source_range"):
        if member in held:
            held[member] = [float(dtype(v)) for v in held[member]]
    found = evaluator_table(held, side=side)
    if extrapolated:
        found = dataclasses.replace(found, extrapolation=ConstantExtrapolation("the cases' end value"))
    return found


def evaluated(table: Table, at: np.ndarray) -> tuple[Evaluation, np.ndarray]:
    """The evaluation at each point and its derivative in the query."""
    query = jnp.asarray(at)

    def value(q):
        return evaluate(table, q, owner=OWNER).value

    found = evaluate(table, query, owner=OWNER)
    _, slope = jax.jvp(value, (query,), (jnp.ones_like(query),))
    return found, np.asarray(slope)


def close(found: float, expected: float, tolerance: float) -> bool:
    if expected == 0.0:
        return found == 0.0
    return abs(found - expected) <= tolerance * abs(expected)


def problems(path: Path, name: str, side: str, extrapolated: bool, evaluator: Evaluator = evaluated) -> list[str]:
    """Each point of the case where ``evaluator`` misses the expected answer by the case's rules."""
    held = case(path)
    request = held["request"]
    tolerance = held["compare"]["relative_tolerance"][name]
    with precision(name) as dtype:
        at = np.asarray([dtype(v) for v in request["at"]], dtype=dtype)
        found, slope = evaluator(table_of(request, dtype, side, extrapolated), at)
        values, in_range, refused = (np.asarray(found.value), np.asarray(found.in_range), np.asarray(found.refused))
        assert values.dtype == dtype, f"the read ran in {values.dtype}, not {dtype.__name__}"
    missed: list[str] = []
    for i, expected in enumerate(held["expected"]["evaluations"]):
        where = f"{path.name} at {request['at'][i]!r}"
        if expected.get("refused"):
            if not refused[i] or not np.isnan(values[i]) or not np.isnan(slope[i]):
                missed.append(f"{where}: refused expected; refusal {refused[i]}, value {values[i]}, slope {slope[i]}")
            continue
        if refused[i]:
            missed.append(f"{where}: a refusal where none is due")
            continue
        if bool(in_range[i]) != expected["in_range"]:
            missed.append(f"{where}: in_range {in_range[i]}, expected {expected['in_range']}")
        outside = not expected["in_range"]
        if not (outside and np.isnan(values[i])) and not close(float(values[i]), expected["value"], tolerance):
            missed.append(f"{where}: value {values[i]!r}, expected {expected['value']!r}")
        derivative = expected["derivative"]
        if derivative == "none":
            if not np.isnan(slope[i]):
                missed.append(f"{where}: a derivative {slope[i]!r} where none is due")
            continue
        wanted = derivative[side] if isinstance(derivative, dict) else derivative
        if not (outside and np.isnan(slope[i])) and not close(float(slope[i]), wanted, tolerance):
            missed.append(f"{where}: derivative {slope[i]!r}, expected {wanted!r} ({side})")
        if outside and not extrapolated and not (np.isnan(values[i]) and np.isnan(slope[i])):
            missed.append(f"{where}: outside the readable range with no extrapolation declared, a number "
                          f"({values[i]!r}, {slope[i]!r}): a clip")
        if outside and extrapolated and (np.isnan(values[i]) or np.isnan(slope[i])):
            missed.append(f"{where}: outside the readable range under a declared extrapolation, no number")
    return missed


def test_the_files_are_the_thirty_five_cases_md_lists():
    listed = (CASES.parent / "CASES.md").read_text(encoding="utf-8")
    assert len(FILES) == 35
    assert all(f"`{path.name.split('.')[1]}`" in listed for path in FILES)


@pytest.mark.parametrize(("name", "side", "extrapolated"), CONFIGURATIONS,
                         ids=[f"{n}-{s}-{'extrapolated' if e else 'refused_outside'}" for n, s, e in CONFIGURATIONS])
@pytest.mark.parametrize("path", RUN, ids=[path.name.removesuffix(".json") for path in RUN])
def test_each_case_passes(path, name, side, extrapolated):
    assert problems(path, name, side, extrapolated) == []


# -- the controls --------------------------------------------------------------------------------------------------

def other_side(table: Table, at: np.ndarray) -> tuple[Evaluation, np.ndarray]:
    """The derivative at an interior knot from the side the table does not declare."""
    return evaluated(dataclasses.replace(table, side="left" if table.side == "right" else "right"), at)


def clipped(table: Table, at: np.ndarray) -> tuple[Evaluation, np.ndarray]:
    """The 0.2 lookup's behaviour: the query clipped into the knots, and every read reported in range."""
    (axis,) = table.axes
    found, slope = evaluated(table, np.clip(at, axis.x[0], axis.x[-1]).astype(at.dtype))
    inside = np.ones_like(np.asarray(found.in_range))
    return found._replace(in_range=inside, accepted=inside), slope * ((at >= axis.x[0]) & (at <= axis.x[-1]))


class _OneBelow:
    """``mime.core.switch`` as the evaluator sees it, its location one interval below the right one."""

    def __getattr__(self, name):
        return getattr(switch, name)

    @staticmethod
    def searchsorted(breaks, x, **settings):
        return jnp.maximum(switch.searchsorted(breaks, x, **settings) - 1, 0)


def one_below(monkeypatch) -> Evaluator:
    def planted(table: Table, at: np.ndarray) -> tuple[Evaluation, np.ndarray]:
        with monkeypatch.context() as patched:
            patched.setattr(evaluate_module, "switch", _OneBelow())
            return evaluated(table, at)

    return planted


class _SlopeKept:
    """``mime.core.switch`` as the evaluator sees it, a read at a readable range of one point keeping its slope."""

    def __getattr__(self, name):
        return getattr(switch, name)

    @staticmethod
    def without_slope(x, mask):
        return x


def slope_kept(monkeypatch) -> Evaluator:
    def planted(table: Table, at: np.ndarray) -> tuple[Evaluation, np.ndarray]:
        with monkeypatch.context() as patched:
            patched.setattr(evaluate_module, "switch", _SlopeKept())
            return evaluated(table, at)

    return planted


def at_the_refusals(change: Callable[[Evaluation, np.ndarray], tuple[Evaluation, np.ndarray]]) -> Evaluator:
    """The evaluator with ``change`` applied to its read: OpenMaterica's edge faults, each of the flags and numbers."""
    def planted(table: Table, at: np.ndarray) -> tuple[Evaluation, np.ndarray]:
        return change(*evaluated(table, at))

    return planted


def unrefused(found: Evaluation, slope: np.ndarray) -> tuple[Evaluation, np.ndarray]:
    """A read at an edge that reports no refusal."""
    return found._replace(refused=np.zeros_like(np.asarray(found.refused))), slope


def refusal_with_a_number(found: Evaluation, slope: np.ndarray) -> tuple[Evaluation, np.ndarray]:
    """A refusal that returns a finite number."""
    value = np.where(np.asarray(found.refused).reshape(-1, *[1] * (np.ndim(found.value) - 1)), 1.0, found.value)
    return found._replace(value=value), np.where(np.asarray(found.refused), 0.0, slope)


def refusal_with_an_infinity(found: Evaluation, slope: np.ndarray) -> tuple[Evaluation, np.ndarray]:
    """A refusal that returns an infinite number."""
    value = np.where(np.asarray(found.refused).reshape(-1, *[1] * (np.ndim(found.value) - 1)), np.inf, found.value)
    return found._replace(value=value), slope


def unrefused_number(found: Evaluation, slope: np.ndarray) -> tuple[Evaluation, np.ndarray]:
    """A read across an edge, or at one, that returns a number and reports no refusal (extrapolation across an edge)."""
    flags = np.asarray(found.refused)
    value = np.where(flags.reshape(-1, *[1] * (np.ndim(found.value) - 1)), 1.0, found.value)
    return found._replace(value=value, refused=np.zeros_like(flags)), np.where(flags, 0.0, slope)


def refusal_not_due(found: Evaluation, slope: np.ndarray) -> tuple[Evaluation, np.ndarray]:
    """A refusal wherever a read is out of range."""
    return found._replace(refused=np.asarray(found.refused) | ~np.asarray(found.in_range)), slope


def value_shifted(found: Evaluation, slope: np.ndarray) -> tuple[Evaluation, np.ndarray]:
    """Every value off by a constant, so that the slope is right and only the value comparison can see it."""
    value = np.asarray(found.value)
    return found._replace(value=value + 1e-3 * (np.abs(value) + 1.0)), slope


#: Each fault's files that OpenMaterica's README names as catching it ("The broken evaluators").
CATCHING = {
    "other_side": {"TE-006.close_sides.json", "TE-006.far_sides.json"},
    "clipped": {"TE-002.linear.json", "TE-003.just_outside_linear.json"},
    "slope_kept": {"TE-007.one_point_range_end.json", "TE-007.one_point_range_inside.json"},
    "unrefused": {"TE-010.linear_edge.json"},
    "refusal_with_a_number": {"TE-010.linear_edge.json"},
    "refusal_with_an_infinity": {"TE-010.linear_edge.json"},
    "unrefused_number": {"TE-010.edge_outside_readable_range.json", "TE-010.edge_outside_readable_range_above.json",
                         "TE-010.edge_on_readable_range_end.json", "TE-010.edge_on_readable_range_end_above.json"},
    "refusal_not_due": {"TE-003.just_outside_linear.json"},
    "value_shifted": set(),
}


@pytest.mark.parametrize(("name", "side", "extrapolated"), CONFIGURATIONS,
                         ids=[f"{n}-{s}-{'extrapolated' if e else 'refused_outside'}" for n, s, e in CONFIGURATIONS])
@pytest.mark.parametrize("fault", ["one_below", *CATCHING])
def test_control_each_planted_fault_fails_cases_in_every_configuration(fault, name, side, extrapolated, monkeypatch):
    evaluator = {"other_side": other_side, "clipped": clipped, "one_below": one_below(monkeypatch),
                 "slope_kept": slope_kept(monkeypatch), "unrefused": at_the_refusals(unrefused),
                 "refusal_with_a_number": at_the_refusals(refusal_with_a_number),
                 "refusal_with_an_infinity": at_the_refusals(refusal_with_an_infinity),
                 "unrefused_number": at_the_refusals(unrefused_number),
                 "refusal_not_due": at_the_refusals(refusal_not_due),
                 "value_shifted": at_the_refusals(value_shifted)}[fault]
    failing = {path.name for path in RUN if problems(path, name, side, extrapolated, evaluator)}
    print(f"\n{fault}, {name}, {side}, {'extrapolated' if extrapolated else 'refused outside'}: fails {len(failing)} "
          f"of {len(RUN)} files")
    assert failing, f"{fault} passed every case"
    assert CATCHING.get(fault, set()) <= failing, CATCHING[fault] - failing


def test_control_the_end_value_returned_outside_with_no_extrapolation_passes_the_cases_and_fails_mimes_rule():
    """The cases accept a number or NaN outside the range; MIME requires NaN there with no extrapolation declared,
    so that an unchecked flag cannot pass as a silent clip."""
    def end_value(table: Table, at: np.ndarray) -> tuple[Evaluation, np.ndarray]:
        declared = dataclasses.replace(table, extrapolation=ConstantExtrapolation("planted"))
        found, slope = evaluated(declared, at)
        return found._replace(accepted=found.in_range), slope

    path = CASES / "TE-002.linear.json"
    found = problems(path, "binary64", "right", False, end_value)
    assert found and all("a clip" in line for line in found), found
