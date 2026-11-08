"""Gate 9, no per-step remake in reverse mode: the joint node's held free parts are its run-constants (S-F0-1's seam,
``MimeNode.run_constants``), made once per chunk. Spheres of 1 and 0.7 mm at 0.6 mm spacing, 0.4 mm apart, float64,
the larger one's radius moved 1 % (so its free part is made from the moved leaf): the value and gradient of a loss of
the final poses through MIME's chunked run (``chunk_function``: each free part made before the loop) equal those
through MADDENING's scan of the step (``held_or_made`` making it again in every step, forward and backward) to 1e-12
of their size at 10 steps; in the chunk's compiled program, value and gradient, no instruction of the free part's own
kernel (its force points against its own quadrature, a shape no other block has, the two bodies' counts differing)
sits inside the step loop, and some sit outside it. The control: in the per-step form's compiled program they sit
inside the step loop, every run. The two-builds check on the joint node's step is the registry walk's case ``mrs_near_field`` in
``tests/two_builds/test_two_builds_walk.py`` (every declared float of the case moved in turn, the programs compared)."""

from __future__ import annotations

import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from near_field_cases import NODE, bodies, joint_node, mrs, precision

from mime.scene import Sphere
from mime.units import m

from mime.build._maddening import chunk_function, fed_inputs, full_state, graph_params, step_function

STEPS = 10
SPACING = 0.6e-3
LEAF = "m0_shape_radius"
WEIGHTS = jnp.asarray([0.3, -0.5, 0.7])


def _with_radius(params: Any, radius: Any) -> Any:
    nodes = dict(params["nodes"])
    held = dict(nodes[NODE])
    held[LEAF] = jnp.asarray(radius, dtype=jnp.asarray(held[LEAF]).dtype)
    nodes[NODE] = held
    return {**params, "nodes": nodes}


def _loss(final: Any) -> Any:
    return sum(1e3 * jnp.dot(final[body]["pose"][:3], WEIGHTS) + jnp.sum(final[body]["pose"][3:])
               for body in ("s0", "s1"))


def seam_loss(built: Any) -> Any:
    graph = built.graph
    run = chunk_function(graph, STEPS, built.held_inputs())
    state, params = full_state(graph), graph_params(graph)
    return lambda radius: _loss(run(state, _with_radius(params, radius)))


def per_step_loss(built: Any) -> Any:
    graph = built.graph
    step = step_function(graph)
    state, params = full_state(graph), graph_params(graph)
    external = fed_inputs(graph, built.held_inputs())

    def loss(radius: Any) -> Any:
        moved = _with_radius(params, radius)

        def body(carry: Any, _: Any) -> tuple[Any, None]:
            return step(carry, external, moved), None

        final, _ = jax.lax.scan(body, state, None, length=STEPS)
        return _loss(final)

    return loss


_CALL = re.compile(r"(?:body|condition|to_apply|calls|true_computation|false_computation)=%?([\w.\-]+)")
_BRANCHES = re.compile(r"branch_computations=\{([^}]*)\}")
_HEAD = re.compile(r"^(ENTRY )?%?([\w.\-]+) .*\{\s*$")


def kernel_places(text: str, steps: int, shape: re.Pattern[str]) -> tuple[int, int]:
    """How many instructions of an optimised HLO module whose type matches ``shape`` lie inside a loop of ``steps``
    trips (reachable from its body), and how many outside every such loop."""
    comps: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        head = _HEAD.match(line)
        if head and not line.startswith(" "):
            current = head.group(2)
            comps[current] = []
            continue
        if line.startswith("}"):
            current = None
        elif current is not None:
            comps[current].append(line)
    calls = {name: set[str]() for name in comps}
    bodies_of: list[str] = []
    for name, lines in comps.items():
        for line in lines:
            calls[name].update(found for found in _CALL.findall(line) if found in comps)
            for group in _BRANCHES.findall(line):
                calls[name].update(each.strip().lstrip("%") for each in group.split(",")
                                   if each.strip().lstrip("%") in comps)
            if " while(" in line and f'"n":"{steps}"' in line:
                found = re.search(r"body=%?([\w.\-]+)", line)
                if found:
                    bodies_of.append(found.group(1))
    inside: set[str] = set()
    todo = list(bodies_of)
    while todo:
        name = todo.pop()
        if name not in inside:
            inside.add(name)
            todo.extend(calls.get(name, ()))
    assert bodies_of, "the program holds the step loop"
    counts = [0, 0]
    for name, lines in comps.items():
        for line in lines:
            if shape.search(line.split("=", 1)[-1].split("(", 1)[0]):
                counts[0 if name in inside else 1] += 1
    return counts[0], counts[1]


@pytest.fixture(scope="module")
def gradients() -> dict[str, Any]:
    with precision("float64"):
        placed = {"s0": (Sphere(radius=m(1e-3)), (0.0, 0.0, 0.0)), "s1": (Sphere(radius=m(0.7e-3)), (2.1e-3, 0.0, 0.0))}
        built = bodies("seam", placed, mrs(SPACING), velocity=(1e-3, 0.0, 0.0)).build()
        node = joint_node(built)
        own = node.member_nodes[0]
        cells, points = own.sample_count, int(own.chunks.points.shape[1])
        assert cells != node.member_nodes[1].sample_count, "the two bodies' counts differ"
        shape = re.compile(rf"\[(?:{cells},{points}|{points},{cells})[,\]]")
        radius = float(np.asarray(graph_params(built.graph)["nodes"][NODE][LEAF])) * 1.01
        found: dict[str, Any] = {"shape": shape, "cells": cells, "points": points}
        for name, make in (("seam", seam_loss), ("per step", per_step_loss)):
            program = jax.jit(jax.value_and_grad(make(built)))
            compiled = program.lower(radius).compile()
            found[name] = (compiled(radius), compiled.as_text())
    return found


def test_reverse_mode_through_the_chunk_equals_the_per_step_form(gradients: dict[str, Any]):
    (value, gradient), _ = gradients["seam"]
    (reference, slope), _ = gradients["per step"]
    print(f"\nthe loss {float(value):.12g} against {float(reference):.12g}; its gradient in s0's radius "
          f"{float(gradient):.12g} against {float(slope):.12g}")
    assert abs(float(value) - float(reference)) <= 1e-12 * abs(float(reference))
    assert abs(float(gradient) - float(slope)) <= 1e-12 * abs(float(slope))
    assert float(slope) != 0.0, "the radius moves the loss, so equal gradients are evidence"


def test_each_free_part_is_made_outside_the_chunks_step_loop(gradients: dict[str, Any]):
    _, text = gradients["seam"]
    inside, outside = kernel_places(text, STEPS, gradients["shape"])
    print(f"\nthe free part's kernel ({gradients['cells']} force points against {gradients['points']} quadrature "
          f"points): {inside} instructions inside the step loop, {outside} outside")
    assert inside == 0
    assert outside >= 1, "the free part is made, once"


def test_control_the_per_step_forms_free_part_is_made_inside_its_step_loop(gradients: dict[str, Any]):
    _, text = gradients["per step"]
    inside, _ = kernel_places(text, STEPS, gradients["shape"])
    assert inside >= 1
