"""The trace form on an array held as built, in a real node's step: a ``MimeNode`` runs its step inside ``jit``, and an
array a nested ``jit`` closes over is the constant of that sub-program, in neither the printed text nor the top-level
program's constants (a Python float prints as a literal and is seen anywhere). The form compares the constants of
every nested sub-program besides (``nested_program``). Review of PR #208 found the form blind to this: a
3-vector of the magnetic torque's moment, half of it taken from the build, gave ``quiet`` below 1e-13 and ``outside-step``
above it, at every move.

- The constructs, alone: an array closed over by a ``jit``, ``cond``, ``scan``, ``while``, ``custom_jvp``,
  ``custom_vjp`` or ``custom_linear_solve``, inside two ``jit`` levels, a NumPy or a JAX array, differs by
  ``nested_program`` and not by ``program_of`` (the control: the old comparison is blind to every one of them, so the
  test can fail), and the same array twice does not differ.
- An array a ``jit`` closes over *inside* a ``cond`` branch or ``custom_linear_solve``'s matvec or solve, a NumPy or a
  JAX array: JAX hoists an array closed over by the branch or the function itself out to the enclosing program's
  constants, so the cases above never need the walk through a tuple-valued parameter (``branches``, ``jaxprs``); this
  array lives only in the closed program of a ``jit`` inside that tuple, and the same three assertions hold.
- A real node (``MagneticTorqueNode``, in ``rotating_screw``; the builder planted none there): half the moment taken
  from the build as a NumPy or a JAX array, in the step's ``jit`` alone, inside a ``scan`` and inside a ``cond``, with
  the moment moved by a relative 1e-1 to 1e-15: the trace form fires as a ``bake`` at every move (and the run form, on
  the NumPy plant, with four times its line to spare from 1e-1 to 1e-12); the twin, the same mix with the live moment,
  fires at none, and is ``quiet`` (a leaf moved).
- A run form that fires where the trace form (made blind on purpose: MIME refuses a host callback, so no plant of a
  value outside both exists in its own nodes) sees nothing and no input or state moved: the check refuses
  (``two-builds.undetected``) rather than call it ``outside-step``; the twin is quiet.
- A pair that moves nothing says so: ``not-discriminating``, never ``quiet``."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from range_cases import rotating_screw

from mime.build import InternalError
from mime.build._in_graph import program_of
from mime.nodes.magnetic import _torque as torque
from mime.testing import _two_builds as two_builds_module
from mime.testing import two_builds_of
from mime.testing._two_builds import nested_program, programs_differ

MOVES = (1e-1, 1e-3, 1e-6, 1e-9, 1e-11, 1e-12, 1e-13, 3e-14, 1e-14, 3e-15, 1e-15)
#: The run form's gating line: four times k resolutions (the plants file's ``MARGIN``), down to this move.
RUN_GATED = 1e-12
MARGIN = 4.0
VALUE = (1.0, 2.0, 3.0)


def constructs(held: Any) -> dict[str, Callable[[Any], Any]]:
    """Each construct as a function of ``x`` that closes over ``held``."""

    @jax.custom_jvp
    def by_jvp(y: Any) -> Any:
        return y * held

    by_jvp.defjvp(lambda primals, tangents: (by_jvp(primals[0]), tangents[0]))

    @jax.custom_vjp
    def by_vjp(y: Any) -> Any:
        return y * held

    by_vjp.defvjp(lambda y: (by_vjp(y), None), lambda residual, cotangent: (cotangent,))
    return {
        "jit": lambda x: jax.jit(lambda y: y * held)(x),
        "cond": lambda x: jax.lax.cond(x[0] > 0, lambda y: y * held, lambda y: y, x),
        "scan": lambda x: jax.lax.scan(lambda c, _: (c * held, None), x, None, length=2)[0],
        "while": lambda x: jax.lax.while_loop(lambda c: c[0] < 100, lambda c: c * held + 1.0, x),
        "custom_jvp": by_jvp,
        "custom_vjp": by_vjp,
        "custom_linear_solve": lambda x: jax.lax.custom_linear_solve(lambda y: held * y, x, lambda _, b: b / held,
                                                                     symmetric=True),
    }


def traced(function: Callable[[Any], Any]) -> Any:
    """``function`` two ``jit`` levels down, as a node's step runs inside the chunk's."""
    return jax.make_jaxpr(lambda x: jax.jit(lambda y: jnp.sum(jax.jit(function)(y)))(x))(jnp.ones(3))


@pytest.mark.parametrize("make", [np.asarray, jnp.asarray], ids=["numpy", "jax"])
@pytest.mark.parametrize("construct", list(constructs(np.asarray(VALUE))))
def test_an_array_held_inside_a_nested_construct_is_a_constant_the_form_compares(construct: str, make: Any):
    def program(scale: float) -> Any:
        return traced(constructs(make(VALUE) * scale)[construct])

    moved, same, again = program(1.0 + 1e-12), program(1.0), program(1.0)
    assert programs_differ(program_of(same), program_of(moved)) == "", "the control: the old comparison is blind"
    assert programs_differ(nested_program(same), nested_program(moved)) == "consts-value"
    assert programs_differ(nested_program(same), nested_program(again)) == "", "the same array twice is the same"


def inside_a_tuple_parameter(held: Any) -> dict[str, Callable[[Any], Any]]:
    """Each construct whose sub-programs sit in a tuple-valued parameter, with ``held`` closed over by a ``jit`` inside
    them: the array is in the closed program of that inner ``jit`` and nowhere else (not hoisted to the enclosing one)."""
    return {
        "cond": lambda x: jax.lax.cond(x[0] > 0, lambda y: jax.jit(lambda z: z * held)(y), lambda y: y, x),
        "custom_linear_solve": lambda x: jax.lax.custom_linear_solve(
            lambda y: jax.jit(lambda z: held * z)(y), x, lambda _, b: jax.jit(lambda z: z / held)(b), symmetric=True),
    }


@pytest.mark.parametrize("make", [np.asarray, jnp.asarray], ids=["numpy", "jax"])
@pytest.mark.parametrize("construct", list(inside_a_tuple_parameter(np.asarray(VALUE))))
def test_an_array_a_jit_closes_inside_a_tuple_valued_parameter_is_a_constant_the_form_compares(
        construct: str, make: Any):
    def program(scale: float) -> Any:
        return traced(inside_a_tuple_parameter(make(VALUE) * scale)[construct])

    moved, same, again = program(1.0 + 1e-12), program(1.0), program(1.0)
    assert programs_differ(program_of(same), program_of(moved)) == "", "the control: the old comparison is blind"
    assert programs_differ(nested_program(same), nested_program(moved)) == "consts-value"
    assert programs_differ(nested_program(same), nested_program(again)) == "", "the same array twice is the same"


def _mixed(live: Any, held: Any, nesting: str) -> Any:
    """Half the live value, half ``held``, built inside a ``jit`` alone, a ``scan`` or a ``cond``."""
    if nesting == "scan":
        return jax.lax.scan(lambda carry, _: (0.5 * carry + 0.5 * held, None), live, None, length=1)[0]
    if nesting == "cond":
        return jax.lax.cond(jnp.sum(live) > -jnp.inf, lambda y: 0.5 * y + 0.5 * held, lambda y: y, live)
    return 0.5 * live + 0.5 * held


@pytest.fixture
def moment(request: pytest.FixtureRequest) -> Iterator[tuple[str, str]]:
    """The magnetic torque with half its moment taken from the build (``plant``), or from the live leaf (``twin``), as
    the kind (``numpy`` or ``jax``) of array, built in the nesting."""
    arm, kind, nesting = request.param
    original = torque.MagneticTorqueNode.advance

    def advance(self: Any, p: Any, state: Any, boundary_inputs: Any, dt: Any) -> Any:
        if arm == "twin":
            held = p["moment"]
        else:
            built = np.asarray(self.constants()["moment"], dtype=np.float64)
            held = built if kind == "numpy" else jnp.asarray(built)
        return original(self, {**p, "moment": _mixed(jnp.asarray(p["moment"]), held, nesting)}, state,
                        boundary_inputs, dt)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torque.MagneticTorqueNode, "advance", advance)
        yield arm, nesting


NESTINGS = ("jit", "scan", "cond")
PLANTS = [("plant", kind, nesting) for nesting in NESTINGS for kind in ("numpy", "jax")]
TWINS = [("twin", "numpy", nesting) for nesting in NESTINGS]


def _ladder(run: bool) -> list[Any]:
    exp = rotating_screw()
    built = exp.build()
    return [two_builds_of(exp, "umr.magnetic_moment", relative=move, built=built, run=run) for move in MOVES]


@pytest.mark.parametrize("moment", PLANTS, indirect=True, ids=["-".join(case) for case in PLANTS])
def test_an_array_plant_in_a_real_nodes_step_is_caught_at_every_move(moment: tuple[str, str], record_property: Any):
    _, nesting = moment
    with_run = nesting == "jit"
    found = _ladder(run=with_run)
    assert all(each.moved for each in found), "the pair must move a leaf, or it shows nothing"
    assert [(move, each.verdict) for move, each in zip(MOVES, found, strict=True) if each.verdict != "bake"] == []
    if with_run:
        short = [move for move, each in zip(MOVES, found, strict=True)
                 if move >= RUN_GATED and not (each.run is not None and each.run.ratio >= MARGIN)]
        assert not short, "a gating plant stands less than four times k resolutions above the run form's line"
        for move, each in zip(MOVES, found, strict=True):
            if move < RUN_GATED and each.run is not None:
                record_property(f"probe_ratio_at_{move:.0e}", each.run.ratio)


@pytest.mark.parametrize("moment", TWINS, indirect=True, ids=["-".join(case) for case in TWINS])
def test_the_twin_of_an_array_plant_is_quiet_at_every_move(moment: tuple[str, str]):
    found = _ladder(run=moment[1] == "jit")
    assert all(each.moved and not each.trace_fired and not each.fired for each in found)
    assert all(each.verdict == "quiet" for each in found)
    assert all(each.run is None or each.run.ratio == 0.0 for each in found)


@pytest.fixture
def blind_trace_form() -> Iterator[None]:
    """The trace form made blind: it compares nothing (what it was to an array in a nested ``jit``, before the walk of
    sub-programs). MIME refuses a host callback in a built graph (``post-build.host-callback``), so a value reaching the
    run by a path neither form compares has no plant in MIME's own nodes; the blind form is that case made on purpose."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(two_builds_module, "programs_differ", lambda a, b: "")
        yield


@pytest.mark.parametrize("moment", [("plant", "numpy", "jit")], indirect=True, ids=["plant-numpy-jit"])
def test_a_run_form_fire_the_trace_form_does_not_see_is_refused_not_named_outside_the_step(
        moment: tuple[str, str], blind_trace_form: None):
    """The program is (as far as the form sees) the same, no state and no input moved, and the run form fires: the
    check's own gap, refused and named, never ``outside-step``."""
    exp = rotating_screw()
    built = exp.build()
    with pytest.raises(InternalError) as raised:
        two_builds_of(exp, "umr.magnetic_moment", relative=1e-3, built=built)
    assert raised.value.codes() == ["two-builds.undetected"]


@pytest.mark.parametrize("moment", [("twin", "numpy", "jit")], indirect=True, ids=["twin-numpy-jit"])
def test_control_the_blind_forms_twin_is_quiet(moment: tuple[str, str], blind_trace_form: None):
    exp = rotating_screw()
    found = two_builds_of(exp, "umr.magnetic_moment", relative=1e-3, built=exp.build())
    assert found.verdict == "quiet" and found.moved and found.run is not None and found.run.ratio == 0.0


def test_a_pair_that_moves_nothing_is_not_discriminating_never_quiet():
    exp = rotating_screw()
    built = exp.build()
    same = two_builds_of(exp, "umr.magnetic_moment", relative=0.0, built=built)
    assert not same.moved and not same.state_moved and not same.inputs_moved and not same.trace_fired
    assert same.run is not None and same.run.ratio == 0.0
    assert same.verdict == "not-discriminating" and not same.fired
    moved = two_builds_of(exp, "umr.magnetic_moment", relative=1e-6, built=built)
    assert moved.verdict == "quiet" and moved.moved
