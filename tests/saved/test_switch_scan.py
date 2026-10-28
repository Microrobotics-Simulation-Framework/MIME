"""The switch scan (``mime.docs._switches``): every registered node's step, traced in the switch configurations, writes
every non-smooth operation through ``mime.core.switch`` under a declared owner, or the site is listed in
``docs/ratchets/raw_switches.txt``; no switch function is evaluated inside a transform; every owner passed is
declared and every declared owner is passed.

The scope is measured: the functions the traces reach hold every census site a node's step reaches. Each family is
planted in a package of its own (one case per operation name, plain, aliased and through ``jax.numpy.``, and one per
other family and per transform, ``vmap`` batched and unbatched; a custom rule in JAX's documented forms; an operation
passed uncalled or through a local alias; a comparison cast by a call, in arithmetic or through a tuple's names; a
raw operation in the second lambda of a line or in a generator expression) and each must fail; a square root of a
domain guard, a NumPy call on a host array, a raw ``abs`` in a constructor, a raw ``where`` in planning code and a
clean generator expression must not. The controls of the list and the owner pass: a converted site still listed, a
listed ``stop_gradient`` naming neither a status nor a test, an owner no entry declares, a declared owner no code
passes, an owner passed by position to a helper, and a registered node class no configuration traces, each fail;
and with JAX's source information gone, the scan says so first and says nothing that rests on it."""

import dataclasses
import importlib.util
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from switch_configurations import PACKAGE, configurations, traced_steps, tree_scan

import mime.docs._switches as switch_scan_module
from mime.build._maddening import make_jaxpr, node_state, parameter_leaves, resolved_inputs, zero_inputs
from mime.core import switch
from mime.docs import load_namespaces, read_list
from mime.docs._check import LISTS, list_path
from mime.docs._switches import (
    SwitchScan,
    TracedStep,
    declaration_problems,
    declared_owners,
    scan_problems,
    scan_switches,
    traced_step,
)
from mime.registry import EntryKind, GradientRecord, Registered, lookup
from mime.testing import installed_entries

ROOT = Path(__file__).resolve().parents[2]
LIST = list_path(ROOT, "raw_switches")


@pytest.fixture(scope="module")
def records() -> list[Registered]:
    load_namespaces()
    found = [lookup(EntryKind(kind), name, version) for kind, name, version in installed_entries()]
    assert all(record is not None for record in found)
    return [record for record in found if record is not None]


@pytest.fixture(scope="module")
def steps() -> tuple[TracedStep, ...]:
    return traced_steps()


@pytest.fixture(scope="module")
def scan() -> SwitchScan:
    return tree_scan()


def node_classes(records) -> set[type]:
    return {node for record in records for nodes in dict(record.cls.builds).values() for node in nodes}


def test_the_tree_passes_the_scan_with_its_list_and_its_owners(scan, records, steps):
    problems = scan_problems(scan, read_list(LIST, LISTS["raw_switches"]), declared_owners(records),
                             node_classes(records), ROOT / "tests")
    assert problems == [], "\n".join(problems)
    print(f"\nconfigurations: {sorted({step.configuration for step in steps})}")
    traced = sorted(scan.traced.items(), key=lambda item: item[0].__name__)
    print("node classes traced: " + ", ".join(f"{cls.__name__} ({len(found)})" for cls, found in traced))
    print(f"functions in scope: {len(scan.scope)}; raw sites: {len(scan.hits)}; owners passed: {len(scan.owners)}")


#: The functions holding the census's sites (counted at 8b3db94) that a node's step reaches: every one but the
#: multibody library's, 12 to 18, which no node runs yet.
CENSUS = {
    (1, 2): ("core/_quaternion.py", "normalized"),
    (3, 4, 5): ("core/_quaternion.py", "exponential"),
    (6,): ("core/_geometry.py", "_unit"),
    (7, 8): ("core/_geometry.py", "Cylinder._radial"),
    (9, 10): ("core/_geometry.py", "Cylinder.outward_normal"),
    (11,): ("core/_geometry.py", "remove_outward"),
    (19, 20, 21, 22, 23, 24): ("nodes/body/_rigid_inertial.py", "RigidInertialNode.held"),
    (25, 26): ("nodes/body/_rigid_inertial.py", "RigidInertialNode._confined"),
    (27, 28, 29, 30): ("nodes/body/_swim_speed.py", "SwimSpeedNode.advance"),
    (31, 32, 33, 34, 35, 36): ("nodes/hydro/_surrogate_drag.py", "SurrogateDragNode.frame"),
    (37, 38, 39, 40): ("nodes/hydro/_surrogate_drag.py", "SurrogateDragNode.canonical_resistance"),
    (41, 42): ("nodes/hydro/_surrogate_drag.py", "SurrogateDragNode.lubrication"),
    (43,): ("nodes/hydro/_surrogate_drag.py", "SurrogateDragNode.end"),
    (44, 45): ("surrogates/_cholesky_mlp.py", "cholesky_resistance"),
    (46,): ("nodes/actuation/_uniform_field.py", "_unit"),
}


def test_the_traced_scope_holds_every_census_site_a_nodes_step_reaches(scan):
    reached = {(function.path, function.name) for function in scan.scope}
    missing = {sites: function for sites, function in CENSUS.items() if function not in reached}
    assert missing == {}
    assert ("core/multibody/_ik.py", "inverse_kinematics") not in reached, "no node runs the inverse kinematics yet"


def test_every_owner_passed_is_one_a_collector_records_in_a_fresh_trace_of_the_steps(scan, steps):
    recorded: set[str] = set()
    for configuration, make in configurations().items():
        graph = make().build().graph
        for name in graph.schedule:
            node = graph.get_node(name)
            given = {**zero_inputs(node), **resolved_inputs(graph, name)}
            with switch.collecting() as rows:
                traced_step(configuration, name, node, node_state(graph, name), given,
                            parameter_leaves(graph).get(name))
            recorded |= {row.owner for row in rows}
    assert recorded == set(scan.owners)


# -- planted packages ----------------------------------------------------------------------------------------------

def scanned(tmp_path: Path, source: str, *, call: str = "step", argument=None) -> SwitchScan:
    """``source`` as a module of a package of its own, its function ``call`` traced, and the package scanned."""
    package = tmp_path / "plant"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    index = len(list(package.glob("planted_*.py")))
    path = package / f"planted_{index}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"plant.planted_{index}_{id(tmp_path)}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    held = jnp.asarray([0.5, -1.5, 2.0]) if argument is None else argument
    step = TracedStep("planted", "planted", object, make_jaxpr(getattr(module, call), held))
    return scan_switches([step], package)


def families_found(found: SwitchScan) -> list[str]:
    return sorted(site.split()[2] for site in found.hits)


PRELUDE = """
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from mime.core import switch
from mime.core._quaternion import safe_norm
"""

#: Each raw operation as the step writes it, on a vector ``x``: ``{f}`` is the callable as each form names it.
OPERATIONS = {
    "where": "{f}(x > 0, x, 0.0)", "maximum": "{f}(x, 0.0)", "minimum": "{f}(x, 0.0)", "abs": "{f}(x)",
    "absolute": "{f}(x)", "clip": "{f}(x, 0.0, 1.0)", "sign": "{f}(x)", "floor": "{f}(x)", "ceil": "{f}(x)",
    "round": "{f}(x)", "rint": "{f}(x)", "trunc": "{f}(x)", "argmax": "{f}(x) * 1.0",
    "argmin": "{f}(x) * 1.0", "argsort": "{f}(x) * 1.0", "sort": "{f}(x)", "max": "{f}(x)", "min": "{f}(x)",
    "amax": "{f}(x)", "amin": "{f}(x)", "fmax": "{f}(x, 0.0)", "fmin": "{f}(x, 0.0)", "heaviside": "{f}(x, 0.5)",
    "select": "{f}([x > 0], [x], 0.0)", "piecewise": "{f}(x, [x > 0], [lambda v: v, lambda v: -v])",
    "fabs": "{f}(x)", "copysign": "{f}(x, -1.0)", "around": "{f}(x)", "mod": "{f}(x, 1.0)",
    "remainder": "{f}(x, 1.0)", "fmod": "{f}(x, 1.0)", "nanmax": "{f}(x)", "nanmin": "{f}(x)", "median": "{f}(x)",
    "hypot": "{f}(x, x)", "searchsorted": "{f}(x, 0.5) * 1.0", "digitize": "{f}(x, jnp.asarray([0.0, 1.0])) * 1.0",
    "interp": "{f}(x, jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 2.0]))", "floor_divide": "{f}(x, 2.0)",
    "histogram": "{f}(x, bins=3)[0] * 1.0", "bincount": "{f}(jnp.arange(3), length=3) * 1.0",
    "unique": "{f}(x, size=3)", "argwhere": "{f}(x, size=1) * 1.0",
}
LAX_OPERATIONS = {
    "select": "{f}(x > 0, x, -x)", "select_n": "{f}(x > 0, -x, x)", "max": "{f}(x, x * 0.0)",
    "min": "{f}(x, x * 0.0)", "abs": "{f}(x)", "sign": "{f}(x)", "clamp": "{f}(0.0, x, 1.0)", "floor": "{f}(x)",
    "ceil": "{f}(x)", "round": "{f}(x)", "cond": "{f}(x[0] > 0, lambda: x, lambda: -x)",
    "switch": "{f}(0, [lambda: x, lambda: -x])", "rem": "{f}(x, x * 0.0 + 1.0)", "top_k": "{f}(x, 1)[0]",
    "sort": "{f}(x)",
}
NN_OPERATIONS = ("relu", "relu6", "hard_tanh", "leaky_relu", "hard_sigmoid", "hard_silu", "hard_swish")


def forms(module: str, name: str) -> dict[str, tuple[str, str]]:
    """The plain, aliased and full forms of ``module.name``: each an import line and the callable as it is written."""
    plain = {"jax.numpy": "jnp", "jax.lax": "lax", "jax.nn": "jax.nn", "jax.numpy.linalg": "jnp.linalg"}[module]
    return {"plain": ("", f"{plain}.{name}"), "aliased": (f"from {module} import {name} as renamed", "renamed"),
            "full": ("", f"{module}.{name}")}


CASES = [(f"jax.numpy.{name}", form, written, template) for name, template in OPERATIONS.items()
         for form, written in forms("jax.numpy", name).items()]
CASES += [(f"jax.lax.{name}", form, written, template) for name, template in LAX_OPERATIONS.items()
          for form, written in forms("jax.lax", name).items()]
CASES += [("jax.numpy.linalg.norm", form, written, "{f}(x)")
          for form, written in forms("jax.numpy.linalg", "norm").items()]
CASES += [(f"jax.nn.{name}", form, written, "{f}(x)") for name in NN_OPERATIONS
          for form, written in forms("jax.nn", name).items()]


@pytest.mark.parametrize(("operation", "form", "written", "template"), CASES,
                         ids=[f"{operation} {form}" for operation, form, *_ in CASES])
def test_a_planted_raw_operation_fails_in_every_form(tmp_path, operation, form, written, template):
    imports, callable_name = written
    source = PRELUDE + imports + "\n\n\ndef step(x):\n    return " + template.format(f=callable_name) + "\n"
    found = scanned(tmp_path, source)
    assert families_found(found) == [operation.rsplit(".", 1)[-1]], (operation, form, found.hits)


@pytest.mark.parametrize(("family", "body"), [
    ("abs", "return abs(x)"),
    ("round", "return round(x)"),
    ("safe_norm", "return safe_norm(x)"),
    ("safe_norm", "from mime.core._quaternion import safe_norm as sn\n    return sn(x)"),
    ("safe_norm", "from mime.core import quaternion\n    return quaternion.safe_norm(x)"),
    ("_safe_norm", "return switch._safe_norm(x)"),
    ("clip", "return x.clip(0.0, 1.0)"),
    ("max", "return x.max()"),
    ("min", "return x.min()"),
    ("argmax", "return x.argmax() * 1.0"),
    ("argmin", "return x.argmin() * 1.0"),
    ("argsort", "return x.argsort() * 1.0"),
    ("sort", "return x.sort()"),
    ("round", "return x.round()"),
    ("abs", "return x.__abs__()"),
    ("sqrt", "return jnp.sqrt(x * x)"),
    ("sqrt", "return jax.numpy.sqrt(x * x)"),
    ("sqrt", "return (x * x) ** 0.5"),
    ("sqrt", "return lax.sqrt(x * x)"),
    ("comparison_cast", "return (x > 0).astype(x.dtype)"),
    ("comparison_cast", "positive = x > 0\n    return positive.astype(x.dtype)"),
    ("comparison_cast", "return x * (x > 0)"),
    ("comparison_cast", "positive = x > 0\n    return positive * x"),
    ("comparison_cast", "held = switch.predicate(x > 0, owner='planted.p', resolution=0.0, indicator=x)\n"
                        "    return held.astype(x.dtype)"),
    # a cell index by division or truncation: a uniform grid's lookup
    ("floor_divide", "return x // 0.5"),
    ("floor_divide", "cell = x // 0.5\n    return cell * 0.5"),
    ("truncation", "return (x / 0.5).astype(jnp.int32) * 1.0"),
    ("truncation", "return x.astype('int32') * 1.0"),
    ("truncation", "return x.astype(int) * 1.0"),
    ("truncation", "return x.astype(np.int32) * 1.0"),
    ("truncation", "return x.astype(jax.numpy.int32) * 1.0"),
    ("truncation", "return jnp.int32(x / 0.5) * 1.0"),
    ("truncation", "return jnp.asarray(x / 0.5, dtype=jnp.int32) * 1.0"),
    ("truncation", "return jnp.asarray(x / 0.5, jnp.int32) * 1.0"),
    ("truncation", "return jnp.array(x / 0.5, dtype=jnp.int32) * 1.0"),
    ("truncation", "return lax.convert_element_type(x / 0.5, jnp.int32) * 1.0"),
    ("truncation", "from jax.numpy import int32 as integer\n    return x.astype(integer) * 1.0"),
    ("stop_gradient", "return x - lax.stop_gradient(x)"),
    ("stop_gradient", "return x - jax.lax.stop_gradient(x)"),
    # operations passed uncalled, through a function's own alias, and powers of a half
    ("max", "return lax.reduce(x, -jnp.inf, lax.max, (0,))"),
    ("maximum", "import functools\n    return functools.partial(jnp.maximum, 0.0)(x)"),
    ("maximum", "mx = jnp.maximum\n    return mx(x, 0.0)"),
    ("sqrt", "return jnp.power(x * x, 0.5)"),
    ("sqrt", "return jnp.float_power(x * x, 0.5)"),
    ("sqrt", "return lax.pow(x * x, 0.5)"),
    ("sqrt", "return (x * x) ** (1 / 2)"),
    # a comparison cast by a call, in arithmetic, or through a tuple's names
    ("comparison_cast", "return x + (x > 0)"),
    ("comparison_cast", "return x * jnp.asarray(x > 0, x.dtype)"),
    ("comparison_cast", "positive = x > 0\n    return x * jnp.asarray(positive, x.dtype)"),
    ("comparison_cast", "return x * jnp.array(x > 0)"),
    ("comparison_cast", "return x * jnp.float32(x > 0)"),
    ("comparison_cast", "return x * lax.convert_element_type(x > 0, x.dtype)"),
    ("comparison_cast", "return (x > 0) @ x"),
    ("comparison_cast", "return jnp.sum(x > 0) * 1.0"),
    ("comparison_cast", "above, below = x > 0, x < 1\n    return above.astype(x.dtype) * x"),
])
def test_a_planted_member_of_every_other_family_fails(tmp_path, family, body):
    found = scanned(tmp_path, PRELUDE + f"\n\ndef step(x):\n    {body}\n")
    assert family.strip("_") in [f.strip("_") for f in families_found(found)], found.hits


@pytest.mark.parametrize(("family", "decorator"), [("custom_jvp", "jax.custom_jvp"), ("custom_vjp", "jax.custom_vjp")])
def test_a_planted_custom_rule_fails(tmp_path, family, decorator):
    rule = ("@twisted.defjvp\ndef twisted_jvp(primals, tangents):\n    return twisted(primals[0]), 2 * tangents[0]\n"
            if family == "custom_jvp" else
            "twisted.defvjp(lambda x: (x * 1.0, None), lambda _, g: (2 * g,))\n")
    source = PRELUDE + f"\n\n@{decorator}\ndef twisted(x):\n    return x * 1.0\n\n\n{rule}\n\ndef step(x):\n" \
                       f"    return twisted(x)\n"
    assert family in families_found(scanned(tmp_path, source))


CUSTOM_FORMS = {
    "custom_jvp through partial, as a decorator": ("custom_jvp", """
from functools import partial


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def twisted(x, k):
    return x * k


@twisted.defjvp
def twisted_jvp(k, primals, tangents):
    return twisted(primals[0], k), 2 * tangents[0]


def step(x):
    return twisted(x, 2.0)
"""),
    "custom_vjp through partial, as a decorator": ("custom_vjp", """
from functools import partial


@partial(jax.custom_vjp, nondiff_argnums=())
def twisted(x):
    return x * 1.0


twisted.defvjp(lambda x: (x * 1.0, None), lambda _, g: (2 * g,))


def step(x):
    return twisted(x)
"""),
    "custom_jvp called among the module's statements": ("custom_jvp", """
def _raw(x):
    return x * 1.0


twisted = jax.custom_jvp(_raw)
twisted.defjvp(lambda primals, tangents: (_raw(primals[0]), 2 * tangents[0]))


def step(x):
    return twisted(x)
"""),
}


@pytest.mark.parametrize("form", sorted(CUSTOM_FORMS))
def test_a_planted_custom_rule_fails_in_jaxs_documented_forms(tmp_path, form):
    family, source = CUSTOM_FORMS[form]
    found = scanned(tmp_path, PRELUDE + source)
    assert families_found(found) == [family], found.hits


def test_every_lambda_on_a_line_is_scanned(tmp_path):
    """A frame names a lambda by its first line, so two on one line are both scanned."""
    for body, hits in (("(lambda v: v * 1.0)(x) + (lambda v: jnp.maximum(v, 0.0))(x)", ["maximum"]),
                       ("(lambda v: jnp.maximum(v, 0.0))(x) + (lambda v: v * 1.0)(x)", ["maximum"]),
                       ("(lambda v: v * 1.0)(x) + (lambda v: v * 2.0)(x)", [])):
        found = scanned(tmp_path, PRELUDE + f"\n\ndef step(x):\n    return {body}\n")
        assert (families_found(found), found.unmatched) == (hits, []), body


def test_a_generator_expression_is_matched_and_scanned(tmp_path):
    clean = scanned(tmp_path, PRELUDE + "\n\ndef step(x):\n    return sum(t * 1.0 for t in [x])\n")
    assert (clean.hits, clean.unmatched) == ({}, [])
    assert any(function.name.endswith("<genexpr>") for function in clean.scope), clean.scope
    raw = scanned(tmp_path, PRELUDE + "\n\ndef step(x):\n    return sum(jnp.maximum(t, 0.0) for t in [x])\n")
    assert (list(raw.hits), raw.unmatched) == (["planted_1.py step.<locals>.<genexpr> maximum 1"], [])


def test_an_owner_passed_by_position_to_a_helper_fails(tmp_path, scan, records):
    helper = "\n\ndef helper(v, owner):\n    return switch.maximum(v, 0.0, owner=owner, resolution=0.0, indicator=v)\n"
    positional = scanned(tmp_path, PRELUDE + helper + "\n\ndef step(x):\n    return helper(x, 'planted.m')\n")
    assert positional.owners == {} and positional.positional_owners, positional.positional_owners
    by_keyword = scanned(tmp_path, PRELUDE + helper + "\n\ndef step(x):\n    return helper(x, owner='planted.m')\n")
    assert list(by_keyword.owners) == ["planted.m"] and by_keyword.positional_owners == []
    real = dataclasses.replace(scan, positional_owners=positional.positional_owners)
    (problem,) = problems_with(real, records, real_list(), tmp_path)
    assert "an owner passed by position to plant.planted_0.helper" in problem


def test_control_steps_whose_equations_carry_no_frame_say_so_first_and_nothing_else(steps, records, tmp_path,
                                                                                    monkeypatch):
    """With JAX's source information gone, the scan's scope is empty: every list line would read as stale and every
    owner as unpassed. The scan says the information did not hold, and nothing that rests on it."""
    monkeypatch.setattr(switch_scan_module, "equation_frames", lambda equation: ())
    blind = scan_switches(steps, PACKAGE)
    found = problems_with(blind, records, real_list(), tmp_path)
    assert len(found) == 1 and "none of which has a frame" in found[0], found


@pytest.mark.parametrize("body", [
    "return jnp.sqrt(switch.domain_guard(x > 0, x, 1.0))",
    "guarded = switch.domain_guard(x > 0, x, 1.0)\n    return jnp.sqrt(guarded)",
    "return switch.domain_guard(x > 0, x, 1.0) ** 0.5",
    "return x * np.abs(HOST).sum() + np.where(HOST > 0, 1.0, 2.0).sum()",
    "return x * CONSTRUCTED.scale + PLANNED",
    "return x ** 2 + jnp.exp(x)",
    "return x.astype(jnp.float32) * 2.0 + x * (7 // 2) + x * jnp.int32(3) + x * int(np.prod((2, 3)))",
    "return switch.where(x > 0, x, 0.0, owner='planted.w', resolution=lambda: jnp.abs(x).max(), "
    "indicator=lambda: jnp.where(x > 0, x, -x))",
])
def test_guarded_square_roots_host_code_and_lazy_indicators_are_not_flagged(tmp_path, body):
    """A constructor and a plan run before the step is traced: their raw ``abs`` and ``where`` produce no equation of
    the step, so no function of theirs is in scope. A lazy indicator or resolution runs only inside a collector."""
    source = PRELUDE + """
HOST = np.array([1.0, -2.0])


class Constructed:
    def __init__(self, scale):
        self.scale = abs(scale) if np.where(scale > 0, True, False) else float(jnp.where(scale < 0, -scale, 0.0))


def plan(held):
    return float(np.where(held > 0, held, 0.0).sum() + jnp.maximum(held, 0.0).sum())


CONSTRUCTED = Constructed(-2.0)
PLANNED = plan(HOST)
""" + f"\n\ndef step(x):\n    {body}\n"
    found = scanned(tmp_path, source)
    assert found.hits == {}, found.hits


# -- nesting -------------------------------------------------------------------------------------------------------

NESTED = {
    "cond": ("lax.cond(x[0] > 0, helper, lambda v: v, x)", "cond"),
    "switch": ("lax.switch(1, [helper, lambda v: v], x)", "cond"),
    "scan": ("lax.scan(lambda c, _: (helper(c), None), x, None, length=2)[0]", "scan"),
    "fori_loop": ("lax.fori_loop(0, 2, lambda i, c: helper(c), x)", "scan"),
    "while_loop": ("lax.while_loop(lambda c: c[0] < 3.0, lambda c: helper(c) + 1.0, x)", "while"),
    "vmap batched": ("jax.vmap(helper)(jnp.stack([x, x]))[0]", "vmap"),
    "vmap unbatched": ("jax.vmap(helper, in_axes=None, axis_size=2)(x)[0]", "vmap"),
    "checkpoint": ("jax.checkpoint(helper)(x)", "remat2"),
    "remat": ("jax.remat(helper)(x)", "remat2"),
    "jit": ("jax.jit(helper)(x)", "jit"),
}


@pytest.mark.parametrize("transform", sorted(NESTED))
@pytest.mark.parametrize("through", ["a helper", "directly"])
def test_a_switch_function_inside_a_transform_is_refused(tmp_path, transform, through):
    call, primitive = NESTED[transform]
    raw = [transform] if transform in ("cond", "switch") else []  # the transform itself, raw: lax.cond and lax.switch
    helper = ("def helper(v):\n    return switch.maximum(v, 0.0, owner='planted.m', resolution=0.0, indicator=v)\n"
              if through == "a helper" else
              "helper = lambda v: switch.maximum(v, 0.0, owner='planted.m', resolution=0.0, indicator=v)\n")
    found = scanned(tmp_path, PRELUDE + f"\n\n{helper}\n\ndef step(x):\n    return {call}\n")
    assert families_found(found) == sorted([*raw, f"nested_{primitive}"]), found.hits


def test_control_a_planted_raw_maximum_in_a_helper_inside_a_scan_body_is_found(tmp_path):
    source = PRELUDE + """

def helper(v):
    return jnp.maximum(v, 0.0)


def step(x):
    return lax.scan(lambda c, _: (helper(c), None), x, None, length=2)[0]
"""
    found = scanned(tmp_path, source)
    assert [site.split(maxsplit=1)[1] for site in found.hits] == ["helper maximum 1"]


def test_a_switch_function_at_the_steps_top_level_is_not_nesting(tmp_path):
    source = PRELUDE + "\n\ndef step(x):\n    return switch.maximum(x, 0.0, owner='planted.m', resolution=0.0, " \
                       "indicator=x) + jax.vmap(jnp.exp)(x)\n"
    found = scanned(tmp_path, source)
    assert found.hits == {} and found.owners == {"planted.m": [found.owners["planted.m"][0]]}


# -- the list and the owners ---------------------------------------------------------------------------------------

def problems_with(scan, records, listed: str, tmp_path, owners=None, classes=None) -> list[str]:
    path = tmp_path / "raw_switches.txt"
    path.write_text(listed, encoding="utf-8")
    return scan_problems(scan, read_list(path, LISTS["raw_switches"]),
                         declared_owners(records) if owners is None else owners,
                         node_classes(records) if classes is None else classes, ROOT / "tests")


def real_list() -> str:
    return LIST.read_text(encoding="utf-8")


def test_control_a_converted_site_still_listed_fails(scan, records, tmp_path):
    stale = "nodes/hydro/_surrogate_drag.py SurrogateDragNode.end where 1  # planted: drag.end is converted\n"
    (problem,) = problems_with(scan, records, real_list() + stale, tmp_path)
    assert "is no longer a raw site a node's step runs; delete this line" in problem


def test_control_an_unlisted_raw_site_fails(scan, records, tmp_path):
    held = real_list().replace("nodes/actuation/_uniform_field.py _unit sqrt 1", "# ")
    (problem,) = problems_with(scan, records, held, tmp_path)
    assert "sqrt in _unit, which a node's step runs, is not written through mime.core.switch" in problem


def test_control_a_listed_stop_gradient_naming_neither_a_status_nor_a_test_fails(scan, records, tmp_path):
    held = real_list().replace(" caps geometry tube 0.1.0 unverified", "")
    found = problems_with(scan, records, held, tmp_path)
    assert any("names the status it caps its entry at" in problem for problem in found), found
    named = real_list().replace(" caps geometry tube 0.1.0 unverified",
                                " cleared_by tests/unit/test_geometry.py::test_no_such_test")
    assert any("is no test of" in problem for problem in problems_with(scan, records, named, tmp_path))
    cleared = real_list().replace(
        " caps geometry tube 0.1.0 unverified",
        " cleared_by tests/saved/test_switch_scan.py::test_control_a_converted_site_still_listed_fails")
    assert problems_with(scan, records, cleared, tmp_path) == []


def test_control_an_undeclared_owner_and_a_declared_owner_nothing_passes_each_fail(scan, records, tmp_path):
    owners = dict(declared_owners(records))
    owners.pop("drag.end")
    owners["drag.unused"] = "interaction drag 0.2.0"
    found = problems_with(scan, records, real_list(), tmp_path, owners=owners)
    assert any("the owner 'drag.end' is passed and no entry or shape declares it" in p for p in found), found
    assert any("declares the switch 'drag.unused', and nothing a node's step runs passes it" in p for p in found)
    assert len(found) == 2


def test_control_a_registered_node_class_no_configuration_traces_fails_naming_it(scan, records, tmp_path):
    class UntracedNode:
        pass

    (problem,) = problems_with(scan, records, real_list(), tmp_path, classes={*node_classes(records), UntracedNode})
    assert "UntracedNode: a registered node class that no configuration traces" in problem


def test_an_owner_the_scan_cannot_read_fails(tmp_path):
    source = PRELUDE + "\n\nNAME = 'planted.m'\n\n\ndef step(x):\n    return switch.maximum(x, 0.0, owner=NAME, " \
                       "resolution=0.0, indicator=x)\n"
    found = scanned(tmp_path, source)
    assert found.unreadable_owners and found.owners == {}


# -- a lab's owners ------------------------------------------------------------------------------------------------

def test_a_labs_owner_carries_its_packages_name(records):
    nudge = next(record for record in records if record.name == "labx.nudge")

    def declaring(owner: str) -> Registered:
        declared = switch.Declaration(owner=owner, indicator="a planted gap, m", orders={"load": 0},
                                      route="none: planted")
        planted = type("Nudge", (nudge.cls,), {"__module__": nudge.cls.__module__,
                                                "gradient": GradientRecord(switches=(declared,))})
        return Registered(nudge.kind, nudge.name, nudge.version, planted)

    assert declaration_problems([declaring("mime_lab_fixture.nudge.cap")]) == []
    (problem,) = declaration_problems([declaring("nudge.cap")])
    assert "does not start with its package's name, mime_lab_fixture." in problem
    twice = declaration_problems([declaring("contact.band"), *[r for r in records if r.name == "contact"]])
    assert len(twice) == 2 and "does not start with its package's name" in twice[0], twice
    assert "interaction contact 0.1.0: the switch 'contact.band' is declared again, first by interaction labx.nudge" \
        in twice[1]


def test_the_labs_package_is_scanned_by_the_same_rule_over_its_own_functions_alone(steps, tmp_path):
    """The lab fixture's node writes a constant, so its step has no equation of its own; MIME's toys beside it are
    traced in the same configuration and none of their functions is the lab's to scan."""
    lab = Path(sys.modules["mime_lab_fixture"].__file__).parent
    found = scan_switches([step for step in steps if step.configuration == "case/nudged_toy"], lab)
    assert sys.modules["mime_lab_fixture"].NudgeNode in found.traced
    assert found.scope == set() and found.hits == {} and found.owners == {}
    empty = tmp_path / "raw_switches.txt"
    empty.write_text("", encoding="utf-8")
    nudge = sys.modules["mime_lab_fixture"].NudgeNode
    assert scan_problems(found, read_list(empty, LISTS["raw_switches"]), {}, {nudge}, ROOT / "tests") == []


def test_jax_records_the_source_of_a_helper_inside_a_scan_body(tmp_path):
    """The source information the scope rests on, which is internal to JAX: a helper called inside a ``lax.scan``
    body is attributed to the helper. An upgrade that breaks it fails here before it shrinks the scope."""
    source = PRELUDE + "\n\ndef helper(v):\n    return jnp.exp(v)\n\n\ndef step(x):\n" \
                       "    return lax.scan(lambda c, _: (helper(c), None), x, None, length=2)[0]\n"
    found = scanned(tmp_path, source)
    assert {function.name for function in found.scope} == {"helper", "step", "step.<locals>.<lambda>"}


def test_jax_is_the_version_the_source_information_was_measured_on():
    assert jax.__version__.startswith("0.12."), "re-measure the scan's source information on this JAX"
