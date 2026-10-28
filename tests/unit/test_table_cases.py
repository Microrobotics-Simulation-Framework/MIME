"""OpenMaterica's table-evaluation cases (its read-side contract, section 7; ``CASES.md`` beside the files) against
``mime.core.tables.evaluate``, each case's table converted through the materials seam (``evaluator_table``), as a
record's table reaches the evaluator.

**Configurations.** Every case that applies runs in binary64 (with ``jax_enable_x64``) and in binary32 (without it,
the case's numbers first rounded to the nearest float32), each compared with its own relative tolerance (1e-12 and the
file's), with each side declared at an interior knot, and both with and without a declared constant extrapolation.
Without one a read outside the readable range returns no number (NaN, which the cases accept, and MIME requires); with
one it returns the end value and a zero derivative, which the cases then compare. The derivative is JAX's, of the value
in the query.

**Skipped**, two files: TE-007 ``one_point_range_end`` and ``one_point_range_inside``, whose readable range is one
point. MIME's evaluator refuses such a table when it is made (it has no inside, so no derivative), and that refusal is
tested here instead. No case is TE-001 (undefined) or TE-009 (a power law, which no 0.1.0 field takes).

**The controls**, each a planted fault run against every case in every configuration: the side not declared (the
derivative at an interior knot from the other side); the 0.2 line's clip (the query clipped into the knots, and every
read reported in range); the cell one below the right one. Each must fail at least one case file in every
configuration, the files OpenMaterica names for that fault among them: the cases are fixed, so each run detects it.
A fourth fault no case can catch, since the cases accept a number or NaN outside the range: the clipped end value
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
from mime.core.tables import ConstantExtrapolation, Evaluation, Table, TableInvalid, evaluate
from mime.materials import evaluator_table

CASES = Path(materica.__file__).parent / "data" / "conformance" / "table_evaluation" / "cases"
FILES = sorted(CASES.glob("TE-*.json"))
SKIPPED = {"TE-007.one_point_range_end.json": "a readable range of one point, which MIME refuses",
           "TE-007.one_point_range_inside.json": "a readable range of one point, which MIME refuses"}
RUN = [path for path in FILES if path.name not in SKIPPED]
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
    query = jnp.asarray(at)[:, None]

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
    assert set(SKIPPED) <= {path.name for path in FILES}


@pytest.mark.parametrize(("name", "side", "extrapolated"), CONFIGURATIONS,
                         ids=[f"{n}-{s}-{'extrapolated' if e else 'refused_outside'}" for n, s, e in CONFIGURATIONS])
@pytest.mark.parametrize("path", RUN, ids=[path.name.removesuffix(".json") for path in RUN])
def test_each_case_passes(path, name, side, extrapolated):
    assert problems(path, name, side, extrapolated) == []


@pytest.mark.parametrize("path", [CASES / name for name in SKIPPED], ids=list(SKIPPED))
def test_a_skipped_cases_one_point_range_is_refused_when_the_table_is_made(path):
    with pytest.raises(TableInvalid, match="readable range is the one point"):
        evaluator_table(case(path)["request"]["table"])


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


#: Each fault's files that OpenMaterica's README names as catching it ("The broken evaluators").
CATCHING = {
    "other_side": {"TE-006.close_sides.json", "TE-006.far_sides.json"},
    "clipped": {"TE-002.linear.json", "TE-003.just_outside_linear.json"},
}


@pytest.mark.parametrize(("name", "side", "extrapolated"), CONFIGURATIONS,
                         ids=[f"{n}-{s}-{'extrapolated' if e else 'refused_outside'}" for n, s, e in CONFIGURATIONS])
@pytest.mark.parametrize("fault", ["other_side", "clipped", "one_below"])
def test_control_each_planted_fault_fails_cases_in_every_configuration(fault, name, side, extrapolated, monkeypatch):
    evaluator = {"other_side": other_side, "clipped": clipped, "one_below": one_below(monkeypatch)}[fault]
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
