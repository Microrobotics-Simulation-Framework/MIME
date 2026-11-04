"""The tallied scan (``scan_chunk_tallied``) and the plain scan (``scan_chunk``) refuse the same graphs (from #129's
review).

The tallied scan has to repeat what MADDENING's ``run_scan`` does around its scan, because that has nowhere to carry
the tallies: the recovery from an escaped tracer, the static-data and parameter check, compiling a graph that is not,
the XLA loop hazard refusal, and the resolution of the external inputs. A guard left out of the copy would let the
tallied scan run a graph the plain scan refuses, or run it stale, only in a run that records flags. This holds the two
to one set of refusals in two ways.

**By behaviour:** each situation is run through both scans and what they raise (the type and the message), or where
neither raises, the state they end in (bit for bit) and the warnings they give, are compared. The situations are an
undeclared input, a node that is not declared, a step that cannot be traced, a graph that runs, an uncompiled graph, a
graph whose static data and one whose parameters change after the scan compiled it (the scan runs again), and a graph
left holding tracers by a differentiated run.

**By source:** the guard calls in ``scan_chunk_tallied``, in order, are those in MADDENING's ``run_scan``, read from
both functions' source. A guard MADDENING adds to ``run_scan`` changes that list, so the ``maddening-main`` lane fails on
it before any user meets a graph the two scans treat differently.

**Controls.** Each guard removed from the tallied scan alone is caught on every run by the situation that answers it
(the escaped-tracer recovery, the static-data check, the compile, the resolution of the external inputs, the hazard
refusal), and a call missing from, added to or moved in the source comparison is caught. The situations are shown to
be refusals or differences at all by the plain scan, so an arm where both ran proves nothing about a guard."""

import ast
import inspect
import textwrap
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from maddening import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.static_data import StaticArray

from mime.build._maddening import FlagTest, scan_chunk, scan_chunk_tallied

DT = 0.02
STEPS = 4
TESTS = [FlagTest("body", "x", limits=(1.0,))]


class _Body(SimulationNode):
    """A node with one state and one external input, ``push``, that moves it by ``dt * gain * sum(table) * push``: its
    ``gain`` is a parameter and its ``table`` static data, both read when the step is traced."""

    def __init__(self, name: str, timestep: float) -> None:
        super().__init__(name, timestep)
        self.params = {"gain": 1.0}
        self.table = jnp.ones(2)

    @property
    def static_data(self) -> dict[str, Any]:
        return {"table": StaticArray(self.table)}

    def initial_state(self) -> dict[str, Any]:
        return {"x": jnp.zeros(3), "push": jnp.zeros(3)}

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"push": BoundaryInputSpec(shape=(3,), default=jnp.zeros(3), coupling_type="additive")}

    def update(self, state: dict[str, Any], boundary_inputs: dict[str, Any], dt: float) -> dict[str, Any]:
        push = jnp.asarray(boundary_inputs.get("push", jnp.zeros(3)), dtype=state["x"].dtype)
        return {"x": state["x"] + dt * self.params["gain"] * jnp.sum(self.table) * push, "push": push}


class _Failing(_Body):
    """A node whose step cannot be traced."""

    def update(self, state: dict[str, Any], boundary_inputs: dict[str, Any], dt: float) -> dict[str, Any]:
        raise RuntimeError("a step that cannot be traced")


def graph(node: type[_Body] = _Body, compiled: bool = True) -> GraphManager:
    held = GraphManager()
    held.add_node(node("body", DT))
    held.add_external_input("body", "push", shape=(3,))
    if compiled:
        held.compile()
    return held


def plain(held: GraphManager, fed: dict[str, Any]) -> Any:
    return scan_chunk(held, STEPS, fed)


def tallied(held: GraphManager, fed: dict[str, Any]) -> Any:
    return scan_chunk_tallied(held, STEPS, fed, TESTS)[0]


FED = {"body": {"push": np.ones(3)}}
Scan = Callable[[GraphManager, dict[str, Any]], Any]


@contextmanager
def hazard_refused(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """MADDENING's loop-hazard refusal, which needs a multi-device mesh to fire for real, made to fire."""
    def refuse(self: GraphManager, entry: str, *, scan: bool) -> None:
        raise RuntimeError(f"{entry}: refused (a planted hazard, scan={scan})")

    monkeypatch.setattr(GraphManager, "_refuse_xla_loop_hazards", refuse)
    yield


def change_static_data(held: GraphManager) -> None:
    held._nodes["body"].node.table = jnp.ones(5)  # type: ignore[attr-defined]  # noqa: SLF001


def change_a_parameter(held: GraphManager) -> None:
    held._nodes["body"].node.params["gain"] = 3.0  # type: ignore[attr-defined]  # noqa: SLF001


def leave_tracers(held: GraphManager) -> None:
    """A differentiated run: ``jax.grad`` returns with the graph holding the tracers of the run it differentiated."""
    def loss(push: Any) -> Any:
        return jnp.sum(scan_chunk(held, STEPS, {"body": {"push": push}})["body"]["x"])

    jax.grad(loss)(jnp.ones(3))


@dataclass(frozen=True)
class Situation:
    """A graph and what is done to it before the scan under test runs: ``warm`` runs that scan once first, so that a
    program it cached is there to go stale; ``before`` then changes the graph."""

    node: type[_Body] = _Body
    fed: dict[str, Any] = None  # type: ignore[assignment]
    compiled: bool = True
    warm: bool = False
    before: Callable[[GraphManager], None] = lambda held: None  # noqa: E731


#: The guard each situation answers, by the name of MADDENING's method.
SITUATIONS: dict[str, Situation] = {
    "an input no node declares": Situation(fed={"body": {"nope": np.ones(3)}}),
    "a node that is not declared": Situation(fed={"ghost": {"push": np.ones(3)}}),
    "a step that cannot be traced": Situation(_Failing, FED),
    "a graph that runs": Situation(fed=FED),
    "a graph that runs with nothing fed": Situation(fed={}),
    "a graph that is not compiled": Situation(fed=FED, compiled=False),
    "static data changed after the scan compiled it": Situation(fed=FED, warm=True, before=change_static_data),
    "a parameter changed after the scan compiled it": Situation(fed=FED, warm=True, before=change_a_parameter),
    "a graph left holding tracers by a differentiated run": Situation(fed=FED, before=leave_tracers),
}
ANSWERED_BY = {
    "_resolve_external_inputs": ("an input no node declares", "a node that is not declared"),
    "compile": ("a graph that is not compiled",),
    "_check_static_data_dirty": ("static data changed after the scan compiled it",
                                 "a parameter changed after the scan compiled it"),
    "_recover_from_escaped_tracers": ("a graph left holding tracers by a differentiated run",),
}


def outcome(scan: Scan, situation: Situation) -> tuple[str, Any]:
    """What a scan does in a situation: refused (the error's type and message), or the state it ended in, with the
    categories of the warnings it gave."""
    held = graph(situation.node, situation.compiled)
    fed = situation.fed
    with warnings.catch_warnings(record=True) as given:
        warnings.simplefilter("always")
        try:
            if situation.warm:
                scan(held, fed)
            situation.before(held)
            final = scan(held, fed)
        except Exception as error:  # noqa: BLE001 - what the two scans raise is what is compared
            return "refused", (type(error).__name__, str(error))
    return "ran", ({node: {name: np.asarray(value).tolist() for name, value in fields.items()}
                    for node, fields in final.items()}, sorted({each.category.__name__ for each in given}))


def disagreements(tallied_scan: Scan = tallied) -> dict[str, tuple[Any, Any]]:
    """The situations in which the two scans do not do the same thing, each with what each did."""
    found: dict[str, tuple[Any, Any]] = {}
    for name, situation in SITUATIONS.items():
        first, second = outcome(plain, situation), outcome(tallied_scan, situation)
        if first != second:
            found[name] = (first, second)
    return found


def test_the_two_scans_refuse_the_same_graphs_and_run_the_others_to_the_same_state():
    assert disagreements() == {}


def test_the_situations_include_real_refusals_and_differences_by_the_plain_scan_so_the_comparison_can_see_a_guard():
    kinds = {name: outcome(plain, SITUATIONS[name])[0] for name in SITUATIONS}
    assert kinds["an input no node declares"] == kinds["a step that cannot be traced"] == "refused"
    assert {kinds[name] for name in SITUATIONS if name not in
            ("an input no node declares", "a node that is not declared", "a step that cannot be traced")} == {"ran"}
    # What the plain scan does on the situations that run is not what an unguarded scan would do: the stale program, the
    # uncompiled step and the tracers each change the state or the warnings.
    ran = outcome(plain, SITUATIONS["a graph that runs"])
    assert outcome(plain, SITUATIONS["static data changed after the scan compiled it"])[1][0] != ran[1][0]
    assert outcome(plain, SITUATIONS["a parameter changed after the scan compiled it"])[1][0] != ran[1][0]
    assert "RuntimeWarning" in outcome(plain, SITUATIONS["a graph left holding tracers by a differentiated run"])[1][1]


def test_the_hazard_refusal_is_asked_of_both_scans_alike(monkeypatch: pytest.MonkeyPatch):
    with hazard_refused(monkeypatch):
        first, second = outcome(plain, SITUATIONS["a graph that runs"]), outcome(tallied, SITUATIONS["a graph that runs"])
    assert first[0] == "refused" and "a planted hazard, scan=True" in first[1][1]
    assert first == second


def without(guard: str, replacement: Callable[..., Any], monkeypatch: pytest.MonkeyPatch) -> Scan:
    """The tallied scan with one of MADDENING's methods replaced for the length of the call: the copy that left it out."""
    def planted(held: GraphManager, fed: dict[str, Any]) -> Any:
        with monkeypatch.context() as local:
            local.setattr(GraphManager, guard, replacement)
            return tallied(held, fed)

    return planted


@pytest.mark.parametrize("guard", sorted(ANSWERED_BY))
def test_the_control_a_tallied_scan_without_a_guard_is_caught_by_the_situation_that_answers_it(
        guard: str, monkeypatch: pytest.MonkeyPatch):
    original = GraphManager._resolve_external_inputs  # noqa: SLF001
    replacements: dict[str, Callable[..., Any]] = {
        "_resolve_external_inputs": lambda self, external: original(self, {}) if external else original(self, external),
        "compile": lambda self, *args, **kwargs: None,
        "_check_static_data_dirty": lambda self: False,
        "_recover_from_escaped_tracers": lambda self, *, warn=True: None,
    }
    found = disagreements(without(guard, replacements[guard], monkeypatch))
    assert set(ANSWERED_BY[guard]) <= set(found), f"{guard} left out and not seen: {sorted(found)}"
    assert all(first != second for first, second in found.values())


def test_the_control_a_tallied_scan_without_the_hazard_guard_is_caught(monkeypatch: pytest.MonkeyPatch):
    def unguarded(held: GraphManager, fed: dict[str, Any]) -> Any:
        with monkeypatch.context() as local:
            local.setattr(GraphManager, "_refuse_xla_loop_hazards", lambda self, entry, *, scan: None)
            return tallied(held, fed)

    monkeypatch.setattr(GraphManager, "_refuse_xla_loop_hazards",
                        lambda self, entry, *, scan: (_ for _ in ()).throw(RuntimeError("a planted hazard")))
    found = disagreements(unguarded)
    planted = ("refused", ("RuntimeError", "a planted hazard"))
    assert found and all(first == planted and second != planted for first, second in found.values())
    assert any(second[0] == "ran" for _, second in found.values()), "and it ran a graph the plain scan refuses"


# --- the guard calls, from the source -------------------------------------------------------------------------------


def guard_calls(source: str, receiver: str) -> list[str]:
    """The calls on ``receiver`` that a function makes before it defines its scan, in order, each as the method's name
    with its string and boolean arguments (the entry point a refusal is asked for): what a function does around its scan, ahead of building it. Read from its source."""
    function = ast.parse(textwrap.dedent(source)).body[0]
    assert isinstance(function, ast.FunctionDef)
    found: list[tuple[int, int, str]] = []
    for statement in function.body:
        if isinstance(statement, ast.FunctionDef):
            break
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == receiver:
                literals = [ast.unparse(each) for each in [*node.args, *(k.value for k in node.keywords)]
                            if isinstance(each, ast.Constant) and isinstance(each.value, str | bool)]
                found.append((node.lineno, node.col_offset, f"{node.func.attr}({', '.join(literals)})"))
    return [name for *_, name in sorted(found)]


def madden_guards() -> list[str]:
    return guard_calls(inspect.getsource(GraphManager.run_scan), "self")


def mime_guards(source: str | None = None) -> list[str]:
    return guard_calls(source or inspect.getsource(scan_chunk_tallied), "held")


def test_the_tallied_scan_makes_the_guard_calls_of_maddenings_run_scan_in_the_same_order():
    wanted = madden_guards()
    assert len(wanted) >= 6, f"run_scan's guards were not read: {wanted}"
    assert mime_guards() == wanted


def test_the_control_a_guard_call_missing_added_or_moved_is_caught_in_the_comparison():
    source = inspect.getsource(scan_chunk_tallied)
    wanted = madden_guards()
    for line in ("    held._check_static_data_dirty()  # noqa: SLF001\n", "    held._recover_from_escaped_tracers()  # noqa: SLF001\n",
                 '    held._refuse_xla_loop_hazards("run_scan", scan=True)  # noqa: SLF001\n',
                 "    external = held._resolve_external_inputs(dict(fed))  # noqa: SLF001\n",
                 "    params = held._params_or_default(None)  # noqa: SLF001\n"):
        assert source.count(line) == 1, f"the control's line is not in the source: {line!r}"
        assert mime_guards(source.replace(line, "")) != wanted, f"a missing call was not caught: {line!r}"
    moved = source.replace("    held._check_static_data_dirty()  # noqa: SLF001\n", "", 1).replace(
        "    held._refuse_xla_loop_hazards(", "    held._check_static_data_dirty()  # noqa: SLF001\n    held._refuse_xla_loop_hazards(", 1)
    assert mime_guards(moved) != wanted, "a moved call was not caught"
    added = inspect.getsource(GraphManager.run_scan).replace(
        "        self._recover_from_escaped_tracers()\n", "        self._recover_from_escaped_tracers()\n        self._a_guard_added_later()\n", 1)
    assert guard_calls(added, "self") != mime_guards() and "_a_guard_added_later()" in guard_calls(added, "self")
    assert mime_guards(source) == wanted, "the control's own reading of the unchanged source agrees"
