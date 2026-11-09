"""Gate 12 (carried from round 1, L2-05): every matrix product of the joint node's step taken at the highest precision,
as dev1's per-body node's are (``tests/oracles/test_mrs.py``): on a GPU the default is a reduced one (TF32-like), which
the 0.2 line found moved its results by 15 %; on either route (past the takeover line, GMRES's products, its
preconditioner's and the joint matrix's on its columns). The control: the near field's MRS body model placing its points at the
default precision, which the check catches."""

import jax
import jax.numpy as jnp
import pytest
from near_field_cases import NODE, bodies, joint_node, mrs, precision, spheres, start_inputs

from mime.build._maddening import parameter_leaves
from mime.nodes.hydro.near_field import _mrs


def products(jaxpr) -> list:
    found = []
    for equation in jaxpr.eqns:
        if equation.primitive.name == "dot_general":
            found.append(equation.params["precision"])
        for value in equation.params.values():
            for inner in value if isinstance(value, tuple | list) else (value,):
                held = getattr(inner, "jaxpr", inner)
                if hasattr(held, "eqns"):
                    found += products(held)
    return found


def step_products(**settings) -> list:
    with precision("float64"):
        built = bodies("precision", spheres(0.3e-3, 2), mrs(0.6e-3, **settings)).build()
        node = joint_node(built)
        inputs = {key: jnp.asarray(value) for key, value in start_inputs(built).items()}
        step = jax.make_jaxpr(lambda p: node.advance(p, node.initial_state(), inputs, 1e-3))
        return products(step(node._step_constants(parameter_leaves(built.graph)[NODE])).jaxpr)  # noqa: SLF001


@pytest.mark.parametrize("route", ["dense", "iterative"])
def test_every_matrix_product_of_the_joint_nodes_step_is_taken_at_the_highest_precision(route):
    found = step_products(**({"near_field_dense": 0} if route == "iterative" else {}))
    assert len(found) >= 8 and all(held == (jax.lax.Precision.HIGHEST,) * 2 for held in found), found


def test_control_the_body_model_placing_its_points_at_the_default_precision_is_caught(monkeypatch):
    monkeypatch.setattr(_mrs, "PRECISION", None)
    assert None in step_products()


def test_control_gmres_taking_its_products_at_the_default_precision_is_caught(monkeypatch):
    from mime.nodes.hydro.near_field import _gmres

    monkeypatch.setattr(_gmres, "PRECISION", None)
    assert None in step_products(near_field_dense=0)
