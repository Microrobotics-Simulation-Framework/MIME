"""The switch harness (``mime.testing``) over every switch MIME declares: each owner has a row, placed a quarter of a
step from its switch on script A's nodes (the body confined in its tube, the drag, the swim speed) or on the cylinder
itself, and measures within its declared order, its indicator at each placed point being that point's offset. Every
row runs in float64: the forced-branch difference needs no finite difference, and every placed point lies outside its
band (the contact band, about 1e-16 m, and the axis band, 3e-15 m) by at least 1e-8 m, or inside it on purpose.

The kink limit is measured: fitted on ``drag.clamp`` (a kink) and ``drag.end`` (a jump), then checked on two
independent planted rows, an analytic kink and an analytic jump. The controls: ``drag.end`` declared a kink fails,
``drag.clamp``'s continuity passes, branches that coincide away from the switch measure continuous, and a step
traced before the forced setting opened fails. The domain guards: at points where each guard's condition is false,
replacing every fallback with twice and half its value changes no output and no derivative; a planted guard whose
fallback reaches an output fails, and so does ``_unit``'s old selection, whose fallback reaches the derivative."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from script_a import script_a

from mime.build._maddening import node_state, parameter_leaves
from mime.core import CYLINDER, CYLINDER_SWITCHES, quaternion, switch
from mime.docs import load_namespaces
from mime.registry import EntryKind, declared_switches, lookup
from mime.testing import KINK_LIMIT, SwitchRow, classify, fallback_problems

DT = 5e-4
#: Where the body sits along the tube's axis (+x): script A's start.
ALONG = 0.01
RADIUS, EXTENT, MARGIN = 3.175e-3, 2.0748e-3, 1e-4
#: The radius of the surface the body's centre is confined to: the tube's, less its extent and margin.
SHRUNK = RADIUS - EXTENT - MARGIN
REFERENCE, OUTER = 1.56e-3, 1.33
#: A body's outward speed where a row needs the hold to engage, m/s.
OUTWARD = 1e-3


@pytest.fixture(scope="module")
def graph():
    return script_a().build().graph


def declarations() -> dict[str, switch.Declaration]:
    load_namespaces()
    found = {d.owner: d for d in CYLINDER_SWITCHES}
    for kind, name, version in (("interaction", "contact", "0.1.0"), ("interaction", "drag", "0.2.0"),
                                ("observation", "swim_speed", "0.1.0"), ("dynamics", "rigid_inertial", "0.2.0")):
        record = lookup(EntryKind(kind), name, version)
        assert record is not None
        found.update({d.owner: d for d in declared_switches(record.cls)})
    return found


def leaves(graph, name: str, **changed) -> dict:
    return {**parameter_leaves(graph)[name], **{key: jnp.asarray(value) for key, value in changed.items()}}


# -- the body, confined to script A's tube -------------------------------------------------------------------------

def body_at(graph, position, velocity=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0), coupled=False):
    """The body's outputs after one step from ``position`` and ``velocity`` with no load, and with the drag's
    resistance at that pose where ``coupled`` (a resistance coupling translation and rotation, so that holding the
    contact changes more than the normal speed). ``velocity`` is the speed the solve leaves the body with, from which
    the starting speed is found by the step's own equation, ``V = V' + dt M^-1 R V'``."""
    node = graph.get_node("umr")
    pose = jnp.asarray((*position, *orientation))
    resistance = jnp.zeros((6, 6))
    if coupled:
        drag = graph.get_node("drag:umr")
        resistance = drag.fresh_update(node_state(graph, "drag:umr"), {"pose": pose}, DT,
                                       params=leaves(graph, "drag:umr"))["resistance"]
    after = jnp.asarray((*velocity, 0.0, 0.0, 0.0))
    mass = parameter_leaves(graph)["umr"]["mass"]
    inertia = parameter_leaves(graph)["umr"]["inertia"]
    inverse_mass = jnp.concatenate([jnp.full(3, 1.0 / mass), 1.0 / inertia])
    start = after + DT * inverse_mass * (resistance @ after)
    out = node.fresh_update({"pose": pose, "twist": start},
                            {"applied_wrench": jnp.zeros(6), "resistance": resistance, "background_load": jnp.zeros(6)},
                            DT, params=leaves(graph, "umr"))
    return {"pose": out["pose"], "twist": out["twist"]}


def below(depth: float) -> tuple[float, float, float]:
    """A point ``depth`` past the shrunk surface (negative: inside it), below the axis, where the outward normal is
    -z."""
    return (ALONG, 0.0, -(SHRUNK + depth))


def tilted(angle: float) -> tuple[float, float, float, float]:
    """The body turned by ``angle`` about z: its x axis at ``cos(angle)`` to the tube's."""
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


def rolled_then_tilted(angle: float) -> tuple[float, float, float, float]:
    """The body turned a quarter turn about x, then by ``angle`` about z: its x axis at ``cos(angle)`` to the tube's
    and its y axis along z, so that the two axes' parts across the tube are perpendicular."""
    quarter = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
    return tuple(float(c) for c in quaternion.multiply(jnp.asarray(tilted(angle)), jnp.asarray(quarter)))


#: The hold ends the step on the surface, where the rule's own switch sits within its rounding: a row that engages
#: the hold keeps the rule's branch fixed, so a rounding flip of it does not read as this switch's.
ON_THE_SURFACE = {"contact.rule": False}


def body_rows(graph, declared) -> list[SwitchRow]:
    down = (0.0, 0.0, -OUTWARD)
    return [
        SwitchRow(declared["contact.band"], lambda o: body_at(graph, below(o), down, coupled=True), 1e-8, side=-1,
                  held=ON_THE_SURFACE),
        SwitchRow(declared["contact.outward"], lambda o: body_at(graph, below(0.0), (0.0, 0.0, -o), coupled=True),
                  1e-4, held=ON_THE_SURFACE),
        SwitchRow(declared["contact.hold"], lambda o: body_at(graph, below(0.0), (0.0, 0.0, -o), coupled=True), 1e-4,
                  held=ON_THE_SURFACE),
        SwitchRow(declared["contact.closing"], lambda o: body_at(graph, below(-o), down, coupled=True), 1e-8,
                  held={**ON_THE_SURFACE, "contact.band": True}),
        SwitchRow(declared["contact.scale"], lambda o: body_at(graph, (0.0, 0.0, -o)), 1e-6),
        SwitchRow(declared["contact.project"], lambda o: body_at(graph, below(o)), 1e-6),
        SwitchRow(declared["contact.rule"], lambda o: body_at(graph, below(o - OUTWARD * DT), down), 1e-6),
        SwitchRow(declared["contact.rule_outward"],
                  lambda o: body_at(graph, below(1e-7 - o * DT), (0.0, 0.0, -o)), 1e-4),
        SwitchRow(declared["quaternion.norm"], lambda o: body_at(graph, below(-5e-4), orientation=(o, 0.0, 0.0, 0.0)),
                  1e-3),
    ]


# -- the cylinder --------------------------------------------------------------------------------------------------

def cylinder_at(point, axis=(1.0, 0.0, 0.0)):
    parameters = {"centre": jnp.zeros(3), "axis": jnp.asarray(axis), "radius": jnp.asarray(RADIUS)}
    held = jnp.asarray(point)
    return {"distance": CYLINDER.signed_distance(parameters, held), "normal": CYLINDER.outward_normal(parameters, held)}


def cylinder_rows(declared) -> list[SwitchRow]:
    return [
        SwitchRow(declared["cylinder.axis"], lambda o: cylinder_at((ALONG, 0.0, -o)), 1e-6),
        SwitchRow(declared["cylinder.axis_norm"], lambda o: cylinder_at((ALONG, 3e-4, -1e-3), axis=(o, 0.0, 0.0)),
                  1e-3),
    ]


# -- the drag ------------------------------------------------------------------------------------------------------

def drag_at(graph, position, orientation=(1.0, 0.0, 0.0, 0.0), **changed):
    node = graph.get_node("drag:umr")
    return node.fresh_update(node_state(graph, "drag:umr"), {"pose": jnp.asarray((*position, *orientation))}, DT,
                             params=leaves(graph, "drag:umr", **changed))


def largest(tube: float) -> float:
    """The largest offset the surrogate is read at, in the body's radius, for a tube of ``tube`` body radii."""
    return (tube - OUTER) * 0.95


def drag_rows(graph, declared) -> list[SwitchRow]:
    tube = RADIUS / REFERENCE
    outer = OUTER * REFERENCE
    wide = 2 * outer + 5e-4

    def room(o):
        return drag_at(graph, (ALONG, 0.0, -2 * o * REFERENCE), vessel_radius=REFERENCE * (OUTER + o / 0.95))

    def feature_floor(o):
        narrow = OUTER + (1e-3 + o) / 0.05
        return drag_at(graph, (ALONG, 0.0, -2 * largest(narrow) * REFERENCE), vessel_radius=REFERENCE * narrow)

    def log_floor(o):
        return drag_at(graph, (ALONG, 0.0, -(wide - outer - outer * math.exp(-o))), vessel_radius=wide,
                       lubrication_epsilon=1e-2)

    return [
        SwitchRow(declared["drag.axis"], lambda o: drag_at(graph, (ALONG, 0.0, -o)), 1e-6),
        SwitchRow(declared["drag.across"], lambda o: drag_at(graph, (ALONG, 0.0, 0.0),
                                                             rolled_then_tilted(math.asin(math.sqrt(0.25 + o)))),
                  1e-3),
        SwitchRow(declared["drag.end"], lambda o: drag_at(graph, (ALONG, 0.0, -5e-4), tilted(math.acos(o - 1e-3))),
                  1e-3),
        SwitchRow(declared["drag.room"], room, 1e-3),
        SwitchRow(declared["drag.clamp"], lambda o: drag_at(graph, (ALONG, 0.0, -(largest(tube) + o) * REFERENCE)),
                  1e-3),
        SwitchRow(declared["drag.clamp_divisor"], lambda o: drag_at(graph, (ALONG, 0.0, -(1e-10 + o) * REFERENCE)),
                  4e-11),
        SwitchRow(declared["drag.feature_gap_floor"], feature_floor, 1e-4),
        SwitchRow(declared["drag.gap_floor"], lambda o: drag_at(graph, (ALONG, 0.0, -(RADIUS - outer - 1e-9 - o))),
                  4e-10),
        SwitchRow(declared["drag.log_floor"], log_floor, 1e-3),
        SwitchRow(declared["drag.axis_norm"], lambda o: drag_at(graph, (ALONG, 0.0, -5e-4), vessel_axis=(o, 0.0, 0.0)),
                  1e-3),
        SwitchRow(declared["drag.direction_norm"], lambda o: drag_at(graph, (ALONG, 0.0, -o)), 1e-6),
    ]


# -- the swim speed ------------------------------------------------------------------------------------------------

def swim_at(graph, *, steps=10, distance=2e-6, observed=1e-3, after=1.0):
    node = graph.get_node("swim_speed:umr")
    state = {"speed": jnp.asarray(0.0), "mean_speed": jnp.asarray(0.0), "distance": jnp.asarray(distance),
             "observed": jnp.asarray(observed), "steps": jnp.asarray(steps, dtype=jnp.int32)}
    return node.fresh_update(state, {"twist": jnp.asarray((1e-3, 0.0, 0.0, 0.0, 0.0, 0.0))}, DT,
                             params=leaves(graph, "swim_speed:umr", after=after))


def swim_rows(graph, declared) -> list[SwitchRow]:
    return [
        SwitchRow(declared["swim.window"], lambda o: swim_at(graph, after=11 * DT - o), 1e-5),
        SwitchRow(declared["swim.mean"], lambda o: swim_at(graph, observed=DT / 2 + o), 1e-5),
    ]


def all_rows(graph) -> dict[str, SwitchRow]:
    declared = declarations()
    rows = [*body_rows(graph, declared), *cylinder_rows(declared), *drag_rows(graph, declared),
            *swim_rows(graph, declared)]
    return {row.declaration.owner: row for row in rows}


OWNERS = sorted(["contact.band", "contact.outward", "contact.hold", "contact.closing", "contact.scale",
                 "contact.project", "contact.rule", "contact.rule_outward", "quaternion.norm", "cylinder.axis",
                 "cylinder.axis_norm", "drag.axis", "drag.across", "drag.end", "drag.room", "drag.clamp",
                 "drag.clamp_divisor", "drag.feature_gap_floor", "drag.gap_floor", "drag.log_floor", "drag.axis_norm",
                 "drag.direction_norm", "swim.window", "swim.mean"])


def test_every_owner_mime_declares_has_a_row():
    assert sorted(declarations()) == OWNERS


@pytest.mark.parametrize("owner", OWNERS)
def test_each_switch_measures_within_its_declared_order(graph, owner):
    found = classify(all_rows(graph)[owner], KINK_LIMIT)
    print(f"\n{found.line()}; indicators {found.indicators}")
    assert found.problems == [], "\n".join(found.problems)
    declared = all_rows(graph)[owner].declaration
    jumps = {group for group, order in declared.orders.items() if order < 0 and found.measures[group].continuous}
    if jumps:
        print(f"{owner}: declared a jump in {sorted(jumps)}, measured continuous at this point")


# -- the kink limit ------------------------------------------------------------------------------------------------

def test_the_kink_limit_lies_between_drag_clamps_kink_and_drag_ends_jump(graph):
    rows = all_rows(graph)
    kink = classify(rows["drag.clamp"], KINK_LIMIT).measures["resistance"]
    jump = classify(rows["drag.end"], KINK_LIMIT).measures["resistance"]
    fitted = math.sqrt(kink.ratio * jump.ratio)
    print(f"\ndrag.clamp's ratio {kink.ratio:.4g}, drag.end's {jump.ratio:.4g}: their geometric mean {fitted:.4g}; "
          f"KINK_LIMIT {KINK_LIMIT}; rounding floor {kink.floor:.3g} and {jump.floor:.3g} (64 units in the last place "
          f"of the resistance's largest entry)")
    assert 2 * kink.ratio < KINK_LIMIT < jump.ratio / 2
    assert 0.5 < fitted / KINK_LIMIT < 2


PLANTED = switch.Declaration(owner="planted.switch", indicator="a planted offset", orders={"value": 0},
                             route="none: planted")


def analytic(value):
    return lambda o: {"value": value(jnp.asarray(1.0 + o), jnp.asarray(1.0))}


@pytest.mark.parametrize(("name", "value", "continuous"), [
    ("an analytic kink", lambda x, y: 3.0 * switch.maximum(x, y, owner="planted.switch", resolution=0.0,
                                                           indicator=x - y) + 2.0 * x, True),
    ("an analytic jump", lambda x, y: switch.where(x > y, x + 0.5, x, owner="planted.switch", resolution=0.0,
                                                   indicator=x - y), False),
    ("branches that coincide away from the switch", lambda x, y: switch.where(x > y, 2.0 * x, 2.0 * x,
                                                                                owner="planted.switch",
                                                                                resolution=0.0, indicator=x - y), True),
])
def test_control_the_kink_limit_classifies_independent_planted_rows(name, value, continuous):
    found = classify(SwitchRow(PLANTED, analytic(value), 1e-3), KINK_LIMIT)
    assert found.measures["value"].continuous is continuous, found.line()


def test_control_drag_end_declared_a_kink_fails_and_drag_clamps_continuity_passes(graph):
    rows = all_rows(graph)
    as_kink = switch.Declaration(owner="drag.end", indicator="planted", orders={"resistance": 0}, route="none: planted")
    found = classify(SwitchRow(as_kink, rows["drag.end"].evaluate, rows["drag.end"].delta), KINK_LIMIT)
    assert any("declared order 0 in resistance" in problem and "it is a jump" in problem for problem in found.problems)
    assert classify(rows["drag.clamp"], KINK_LIMIT).problems == []


def test_control_a_misplaced_point_or_a_flipped_indicator_fails(graph):
    rows = all_rows(graph)
    flipped = SwitchRow(rows["contact.project"].declaration, lambda o: rows["contact.project"].evaluate(-o), 1e-6)
    displaced = SwitchRow(rows["drag.clamp"].declaration, lambda o: rows["drag.clamp"].evaluate(1e3 * o), 1e-3)
    for row in (flipped, displaced):
        assert any("a sign or units error" in problem for problem in classify(row, KINK_LIMIT).problems)


def test_control_a_step_traced_before_the_setting_opened_fails(graph):
    node = graph.get_node("drag:umr")
    node.update(node_state(graph, "drag:umr"), {"pose": jnp.asarray((ALONG, 0.0, -5e-4, 1.0, 0.0, 0.0, 0.0))}, DT)

    def cached(o):
        pose = jnp.asarray((ALONG, 0.0, -5e-4, *tilted(math.acos(o - 1e-3))))
        return node.update(node_state(graph, "drag:umr"), {"pose": pose}, DT)

    found = classify(SwitchRow(declarations()["drag.end"], cached, 1e-3), KINK_LIMIT)
    assert any("the forced branch never reached it" in problem for problem in found.problems)


# -- the domain guards ---------------------------------------------------------------------------------------------

def guard_points(graph):
    """Points where each domain guard of a node's step has its condition false: the body at the world's origin, on
    the tube's axis and free (the norm's, the cylinder's two and the hold's); the drag on the axis; the swim speed
    before it has observed."""

    def body(state):
        unloaded = {"applied_wrench": jnp.zeros(6), "resistance": jnp.zeros((6, 6)), "background_load": jnp.zeros(6)}
        return graph.get_node("umr").fresh_update(state, unloaded, DT, params=leaves(graph, "umr"))

    body_state = {"pose": jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)), "twist": jnp.asarray((1e-4,) * 6)}

    def drag(pose):
        return graph.get_node("drag:umr").fresh_update(node_state(graph, "drag:umr"), {"pose": pose}, DT,
                                                       params=leaves(graph, "drag:umr"))

    drag_pose = jnp.asarray((ALONG, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))

    def swim(twist):
        return swim_at_twist(graph, twist)

    swim_twist = jnp.asarray((1e-3, 0.0, 0.0, 0.0, 0.0, 0.0))
    return {
        "the body at the origin, on the axis, free": (lambda: body(body_state), lambda: jax.jacfwd(body)(body_state)),
        "the drag on the axis": (lambda: drag(drag_pose), lambda: jax.jacfwd(drag)(drag_pose)),
        "the swim speed before observing": (lambda: swim(swim_twist), lambda: jax.jacfwd(swim)(swim_twist)),
    }


def swim_at_twist(graph, twist):
    node = graph.get_node("swim_speed:umr")
    state = {"speed": jnp.asarray(0.0), "mean_speed": jnp.asarray(0.0), "distance": jnp.asarray(0.0),
             "observed": jnp.asarray(0.0), "steps": jnp.asarray(0, dtype=jnp.int32)}
    return node.fresh_update(state, {"twist": twist}, DT, params=leaves(graph, "swim_speed:umr", after=1.0))


@pytest.mark.parametrize("where", ["the body at the origin, on the axis, free", "the drag on the axis",
                                   "the swim speed before observing"])
def test_replacing_every_fallback_changes_nothing_where_its_condition_is_false(graph, where):
    evaluate, derivative = guard_points(graph)[where]
    assert fallback_problems(evaluate, derivative, where) == []


@pytest.mark.parametrize(("name", "value"), [
    ("a fallback reaching the value", lambda x: 3.0 * switch.domain_guard(x > 0, x, 1.0)),
    ("a fallback reaching the derivative (_unit's old selection)",
     lambda x: x / switch.domain_guard(jnp.abs(x) > 0, jnp.abs(x), 1.0)),
    ("a zero fallback, which scaling cannot change", lambda x: jnp.sqrt(switch.domain_guard(x > 0, x, 0.0)) * 0.0),
])
def test_control_a_planted_guard_whose_fallback_reaches_an_output_fails(name, value):
    point = jnp.asarray(0.0)
    found = fallback_problems(lambda: {"value": value(point)}, lambda: {"value": jax.grad(value)(point)}, name)
    assert found, name


def test_the_harness_runs_in_float64():
    assert jnp.zeros(()).dtype == np.float64
