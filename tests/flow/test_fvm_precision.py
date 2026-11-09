"""Gate 14, matmul precision: MADDENING pins the step builders it compiles at "highest", but a dot MIME traces itself
runs at the backend's default, which on a GPU's TF32 is 15 % off a float32 drag (CLAUDE.md's precision line). So every
dot_general in the FVM's programs, IMEX Euler's step, its steady limit's and RK3's, states the highest precision, seen
on a CPU from the traced program. The control: the wall's signed distance traced without its precision context (the
one place the FVM multiplies matrices) has dots the walk finds unpinned."""

from __future__ import annotations

import contextlib

import jax
import pytest
from fvm_cases import step_of, tube_node

import mime.nodes.flow.fvm._band as band
from mime.nodes.flow.fvm import ExplicitFlowNode, periodic_tube_layout

HIGHEST = jax.lax.Precision.HIGHEST


def dots(jaxpr: object, found: list[object]) -> list[object]:
    """Every dot_general's precision in ``jaxpr`` and the programs it holds (loops, branches, solves)."""
    for eqn in jaxpr.eqns:  # type: ignore[attr-defined]
        if eqn.primitive.name == "dot_general":
            found.append(eqn.params.get("precision"))
        for value in eqn.params.values():
            for held in value if isinstance(value, (list, tuple)) else (value,):
                inner = getattr(held, "jaxpr", held)
                if hasattr(inner, "eqns"):
                    dots(inner, found)
    return found


def programs() -> dict[str, object]:
    held: dict[str, object] = {}
    for scheme in ("imex_euler", "steady"):
        node = tube_node(8, scheme=scheme, convection=scheme == "imex_euler")
        held[scheme] = jax.make_jaxpr(step_of(node, node._step_constants(None)))(dict(node.initial_state()))  # noqa: SLF001
    layout = periodic_tube_layout(0.5, 4.0 / 8, cells_across=8)
    rk3 = ExplicitFlowNode("flow", 0.1 * layout.spacing ** 2, cells_x=layout.cells[0], cells_y=layout.cells[1],
                           cells_z=layout.cells[2], spacing=layout.spacing, viscosity=1.0, density=1.0,
                           tube_centre=layout.centre, tube_axis=layout.axis, tube_radius=layout.radius, scheme="rk3")
    held["rk3"] = jax.make_jaxpr(step_of(rk3, rk3._step_constants(None)))(dict(rk3.initial_state()))  # noqa: SLF001
    return held


@pytest.mark.parametrize("planted", [False, True])
def test_every_dot_in_the_fvms_programs_states_the_highest_precision(dtype, planted):
    with pytest.MonkeyPatch.context() as patch:
        if planted:
            patch.setattr(band.jax, "default_matmul_precision", lambda _: contextlib.nullcontext())
        held = programs()
    for name, program in held.items():
        found = dots(program.jaxpr, [])  # type: ignore[attr-defined]
        assert found, name  # the wall's signed distance multiplies matrices in every program
        pinned = [precision is not None and all(p == HIGHEST for p in (precision if isinstance(precision, tuple)
                                                                     else (precision,))) for precision in found]
        assert all(pinned) != planted, (name, found)
