"""``mime.core.switch``: each function returns its ``jnp`` operation's value and derivative, ties included, and with
no context open its traced program is exactly the operation's; the module imports only JAX, jaxtyping and the
standard library; a collector around a fresh trace records each switch with its owner; and the test-only setting acts
only on a trace made inside it.

The controls: a wrapper with another tie convention, and a raw ``jnp.linalg.norm``, fail the comparison; an indicator
evaluated eagerly adds equations; a planted ``mime`` or NumPy import is found; a planted switch appears in the
collector's rows and a removed one vanishes; and a step traced before the setting opened, reused inside it, reaches
nothing, which :func:`assert_reached` refuses."""

import ast
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mime.core import switch


@pytest.fixture(autouse=True, scope="module")
def x64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


OWNED = {"owner": "toy.switch", "resolution": 0.0, "indicator": 0.0}

#: Each function beside the operation it is, as a function of one scalar, and the points it is compared at: its ties
#: (where JAX's convention decides the derivative) and points either side.
PAIRS = {
    "maximum": (lambda x: switch.maximum(x, 0.0, **OWNED), lambda x: jnp.maximum(x, 0.0), (0.0, -0.5, 0.5)),
    "minimum": (lambda x: switch.minimum(x, 0.0, **OWNED), lambda x: jnp.minimum(x, 0.0), (0.0, -0.5, 0.5)),
    "absolute": (lambda x: switch.absolute(x, **OWNED), jnp.abs, (0.0, -0.5, 0.5)),
    "clip": (lambda x: switch.clip(x, 0.0, 1.0, **OWNED), lambda x: jnp.clip(x, 0.0, 1.0), (0.0, 1.0, -0.5, 0.5, 1.5)),
    "where": (lambda x: switch.where(x > 0, x, 0.0, **OWNED), lambda x: jnp.where(x > 0, x, 0.0), (0.0, -0.5, 0.5)),
    "predicate": (lambda x: jnp.where(switch.predicate(x > 0, **OWNED), x, 0.0), lambda x: jnp.where(x > 0, x, 0.0),
                  (0.0, -0.5, 0.5)),
    "domain_guard": (lambda x: jnp.sqrt(switch.domain_guard(x > 0, x, 1.0)),
                     lambda x: jnp.sqrt(jnp.where(x > 0, x, 1.0)), (0.0, -0.5, 0.5)),
    "static_select": (lambda x: switch.static_select(np.array(True), x, 0.0), lambda x: jnp.where(True, x, 0.0),
                      (0.0, 0.5)),
}

#: The derivatives at the ties, as JAX gives them (jax 0.12): pinned, since a change would move gradients there.
TIES = {"maximum": 0.5, "minimum": 0.5, "absolute": 1.0, "clip": 0.5, "where": 0.0, "predicate": 0.0}


def safe_norm_formula(v):
    """MIME's safe norm as it was written before the switch module held it."""
    squared = jnp.sum(v * v)
    positive = squared > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 0.0)


def differences(switched, raw, points) -> list[str]:
    """Every point where ``switched`` and ``raw`` differ in value or derivative, bit for bit (NaN equal to NaN)."""
    found = []
    for point in points:
        x = jnp.asarray(point)
        ours, theirs = jax.value_and_grad(switched)(x), jax.value_and_grad(raw)(x)
        for what, a, b in zip(("value", "derivative"), ours, theirs, strict=True):
            if not np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True):
                found.append(f"{what} at {point}: {np.asarray(a)} against {np.asarray(b)}")
    return found


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_each_function_has_its_operations_value_and_derivative_ties_included(name):
    switched, raw, points = PAIRS[name]
    assert differences(switched, raw, points) == []
    if name in TIES:
        assert float(jax.grad(switched)(jnp.asarray(0.0))) == TIES[name]


def test_clip_takes_half_the_derivative_at_its_upper_bound_too():
    assert float(jax.grad(PAIRS["clip"][0])(jnp.asarray(1.0))) == 0.5


def test_norm_is_the_safe_norm_with_derivative_zero_at_zero():
    owned = lambda v: switch.norm(v, **OWNED)  # noqa: E731 - one expression
    for v in (np.zeros(3), np.array([3.0, -4.0, 0.0]), np.array([1e-200, 0.0, 0.0])):
        x = jnp.asarray(v)
        assert np.array_equal(np.asarray(owned(x)), np.asarray(safe_norm_formula(x)))
        assert np.array_equal(np.asarray(jax.grad(owned)(x)), np.asarray(jax.grad(safe_norm_formula)(x)))
    assert np.array_equal(np.asarray(jax.grad(owned)(jnp.zeros(3))), np.zeros(3))


@given(st.floats(-1e3, 1e3), st.one_of(st.just(None), st.floats(-1e3, 1e3)))
def test_the_binary_functions_equal_their_operations_over_draws_ties_among_them(x, y):
    y = x if y is None else y
    for switched, raw in ((switch.maximum, jnp.maximum), (switch.minimum, jnp.minimum)):
        assert differences(lambda a, s=switched: s(a, y, **OWNED), lambda a, r=raw: r(a, y), (x,)) == []


def test_control_another_tie_convention_or_a_raw_norm_fails_the_comparison():
    other_tie = lambda x: jnp.where(x >= 0.0, x, 0.0)  # noqa: E731 - maximum with the derivative 1 at its tie
    assert differences(other_tie, PAIRS["maximum"][1], (0.0,)) != []
    assert differences(lambda x: jnp.where(x > 0, x, -x), jnp.abs, (0.0,)) != []
    norm = lambda v: switch.norm(v, **OWNED)  # noqa: E731
    found = differences(lambda x: jnp.linalg.norm(jnp.stack([x, 0.0 * x])),
                        lambda x: norm(jnp.stack([x, 0.0 * x])), (0.0,))
    assert found == ["derivative at 0.0: nan against 0.0"]


# -- the program with no context open ------------------------------------------------------------------------------

def test_with_no_context_open_the_traced_program_is_the_operations():
    def expensive():
        return jnp.sin(jnp.cos(jnp.ones(3))).sum()

    def owned(x):
        return switch.maximum(x, 0.0, owner="toy.max", resolution=expensive, indicator=lambda: jnp.exp(x))

    assert str(jax.make_jaxpr(owned)(1.0)) == str(jax.make_jaxpr(lambda x: jnp.maximum(x, 0.0))(1.0))

    def eager(x):  # the control: an indicator computed in the step, though only a collector reads it
        return switch.maximum(x, 0.0, owner="toy.max", resolution=0.0, indicator=jnp.exp(x))

    assert str(jax.make_jaxpr(eager)(1.0)) != str(jax.make_jaxpr(lambda x: jnp.maximum(x, 0.0))(1.0))


# -- the module's imports ------------------------------------------------------------------------------------------

ALLOWED = frozenset({"jax", "jaxtyping"}) | frozenset(sys.stdlib_module_names)


def outside_imports(source: str) -> set[str]:
    """The top-level packages ``source`` imports that are neither JAX, jaxtyping nor the standard library's."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            found.add(".")
    return found - ALLOWED


def test_the_module_imports_only_jax_jaxtyping_and_the_standard_library():
    source = Path(switch.__file__).read_text(encoding="utf-8")
    assert "import jax" in source, "the scan read no import at all"
    assert outside_imports(source) == set()


@pytest.mark.parametrize("planted", ["from mime.core import jnp", "import mime", "import numpy as np",
                                     "from . import _node", "from maddening.core.node import SimulationNode"])
def test_control_a_planted_import_is_found(planted):
    source = Path(switch.__file__).read_text(encoding="utf-8") + f"\n{planted}\n"
    assert outside_imports(source)


# -- the collector -------------------------------------------------------------------------------------------------

def toy_step(x, y):
    a = switch.maximum(x, y, owner="toy.a", resolution=0.0, indicator=lambda: x - y)
    return switch.where(a > 1.0, a, -a, owner="toy.b", resolution=0.0, indicator=lambda: a - 1.0)


def planted_step(x, y):
    return switch.absolute(toy_step(x, y), owner="toy.c", resolution=0.0, indicator=lambda: toy_step(x, y))


def removed_step(x, y):
    a = jnp.maximum(x, y)
    return switch.where(a > 1.0, a, -a, owner="toy.b", resolution=0.0, indicator=lambda: a - 1.0)


def owners_collected(step) -> list[str]:
    with switch.collecting() as rows:
        jax.make_jaxpr(step)(2.0, 0.5)
    return [row.owner for row in rows]


def test_a_collector_around_a_fresh_trace_records_each_switch_with_its_owner_and_indicator():
    assert owners_collected(toy_step) == ["toy.a", "toy.b"]
    with switch.collecting() as rows:
        toy_step(jnp.asarray(2.0), jnp.asarray(0.5))
    assert [(row.owner, row.function, float(row.indicator), bool(row.branch)) for row in rows] == [
        ("toy.a", "maximum", 1.5, True), ("toy.b", "where", 1.0, True)]


def test_control_a_planted_switch_appears_and_a_removed_one_vanishes():
    assert owners_collected(planted_step)[-1] == "toy.c"
    assert "toy.c" not in owners_collected(toy_step)
    assert owners_collected(removed_step) == ["toy.b"]


def test_nothing_is_collected_outside_a_collector_and_a_collector_adds_no_derivative():
    with switch.collecting() as rows:
        value, slope = jax.value_and_grad(toy_step)(2.0, 0.5)
    assert len(rows) == 2
    assert (float(value), float(slope)) == (float(toy_step(2.0, 0.5)), float(jax.grad(toy_step)(2.0, 0.5)))


# -- the test-only setting -----------------------------------------------------------------------------------------

def assert_reached(reached: switch.Reached, owners: set[str]) -> None:
    """That a setting reached every owner it forced: a step traced before it opened reaches none."""
    assert owners <= reached.owners, f"the setting never reached {sorted(owners - reached.owners)}: the step that " \
                                     f"ran was traced before the setting opened"


def test_a_forced_branch_acts_on_a_trace_made_inside_the_setting():
    with switch.forcing({"toy.b": False}) as reached:
        forced = jax.jit(lambda x, y: toy_step(x, y))(2.0, 0.5)
    assert_reached(reached, {"toy.b"})
    assert float(forced) == -2.0 and float(toy_step(2.0, 0.5)) == 2.0


def test_control_a_step_traced_before_the_setting_opened_and_reused_inside_it_fails():
    compiled = jax.jit(toy_step)
    compiled(2.0, 0.5)
    with switch.forcing({"toy.b": False}) as reached:
        reused = compiled(2.0, 0.5)
    assert float(reused) == 2.0
    with pytest.raises(AssertionError, match="never reached"):
        assert_reached(reached, {"toy.b"})


@pytest.mark.parametrize(("call", "first", "second"), [
    (lambda: switch.maximum(1.0, 3.0, **OWNED), 1.0, 3.0),
    (lambda: switch.minimum(1.0, 3.0, **OWNED), 1.0, 3.0),
    (lambda: switch.absolute(-2.0, **OWNED), -2.0, 2.0),
    (lambda: switch.norm(jnp.asarray([3.0, 4.0]), **OWNED), 5.0, 0.0),
    (lambda: switch.where(False, 1.0, 3.0, **OWNED), 1.0, 3.0),
    (lambda: jnp.where(switch.predicate(jnp.asarray(False), **OWNED), 1.0, 3.0), 1.0, 3.0),
])
def test_each_function_takes_the_forced_branch_either_way(call, first, second):
    for branch, expected in ((True, first), (False, second)):
        with switch.forcing({"toy.switch": branch}) as reached:
            assert float(call()) == expected
        assert reached.owners == {"toy.switch"}


def test_clip_forces_each_bound_as_its_own_switch():
    clip = lambda: float(switch.clip(-1.0, 0.0, 2.0, owner="toy.clip", resolution=0.0, indicator=-1.0))  # noqa: E731
    with switch.forcing({"toy.clip.low": True}):
        assert clip() == -1.0
    with switch.forcing({"toy.clip.high": False}):
        assert clip() == 2.0
    assert clip() == 0.0


def test_replaced_fallbacks_scale_every_domain_guard_in_a_fresh_trace():
    guarded = lambda x: switch.domain_guard(x > 0, x, 1.0)  # noqa: E731
    with switch.replacing_fallbacks(2.0) as reached:
        assert float(guarded(-1.0)) == 2.0 and float(guarded(3.0)) == 3.0
    assert reached.guards == 2 and float(guarded(-1.0)) == 1.0


def test_a_zero_fallback_is_replaced_by_the_scale_itself_and_each_guard_records_its_call_site():
    with switch.replacing_fallbacks(0.5) as reached:
        assert float(switch.domain_guard(jnp.asarray(-1.0) > 0, 3.0, 0.0)) == 0.5
        assert float(switch.domain_guard(jnp.asarray(1.0) > 0, 3.0, 0.0)) == 3.0
    sites = sorted(reached.guard_sites.items())
    assert [(Path(site.rpartition(":")[0]).name, false) for site, false in sites] == [
        ("test_switch.py", True), ("test_switch.py", False)]


def test_nonzero_norm_is_the_safe_norm_and_records_its_zero_only_in_the_setting():
    """No switch: with no context open its program is the safe norm's; in the test-only setting its call site is
    recorded with whether its vector was zero, and the guards inside it (the switch module's own) are not."""
    for v in (np.array([3.0, -4.0, 0.0]), np.zeros(3)):
        assert str(jax.make_jaxpr(switch.nonzero_norm)(v)) == str(jax.make_jaxpr(safe_norm_formula)(v))
    with switch.replacing_fallbacks(2.0) as reached:
        switch.nonzero_norm(jnp.asarray([3.0, 4.0]))
        switch.nonzero_norm(jnp.zeros(2))
    assert sorted(reached.nonzero_norms.values()) == [False, True] and reached.guard_sites == {}
    assert reached.guards == 4


def test_static_select_refuses_a_traced_mask():
    assert float(switch.static_select(np.array(False), 1.0, 2.0)) == 2.0
    with pytest.raises(TypeError, match="fixed at build time"):
        jax.jit(lambda mask: switch.static_select(mask, 1.0, 2.0))(True)


@pytest.mark.parametrize(("settings", "complaint"), [
    ({"owner": "Contact.Band"}, "dotted lower-case"),
    ({"owner": "band"}, "dotted lower-case"),
    ({"orders": {}}, "orders"),
    ({"orders": {"twist": -2}}, "orders"),
    ({"orders": {"twist": True}}, "orders"),
    ({"route": " "}, "a sentence"),
])
def test_a_declaration_refuses_a_malformed_owner_order_or_route(settings, complaint):
    declared = {"owner": "toy.band", "indicator": "the gap, m", "orders": {"twist": -1}, "route": "none: a toy",
                **settings}
    with pytest.raises(ValueError, match=complaint):
        switch.Declaration(**declared)
    assert switch.Declaration(owner="toy.band", indicator="the gap, m", orders={"twist": -1, "pose": 0},
                              route="none: a toy").lowest_order == -1
