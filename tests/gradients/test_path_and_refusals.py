"""What a study's derivative is taken through, and what it refuses: on the toy probe pulled by a planted spring
(``tests/study/planted.py``), whose node by its kind holds the stiffness twice (a linear map, read through its factor),
derives a static from it (another program a little way off), consumes it at build (a leaf the step never reads) or
hides it (no leaf moves); a variant axis, an axis with no number, an unknown mode; a ``cut`` or ``unverified`` path
refused unless acknowledged, a callback controller's path ``cut`` (its writes held as the forward run made them), and
the statuses composed."""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

STUDY = str(Path(__file__).resolve().parents[1] / "study")
if STUDY not in sys.path:
    sys.path.append(STUDY)

from control_cases import Follower, spun_magnet  # noqa: E402
from planted import planted  # noqa: E402
from planted_gradients import BOUND, planted_gradient  # noqa: E402

from mime.build import ArgumentRefused, BuildRefused  # noqa: E402
from mime.gradients import ACKNOWLEDGED_ONLY, GradientStatus, composed, unacknowledged  # noqa: E402
from mime.study import DT, Study, along  # noqa: E402
from mime.units import N, m, ms, s  # noqa: E402

STEPS = 20


def position(final):
    return jnp.sum(final["probe"]["x"] * jnp.asarray((0.3, -0.5, 0.7)))


def spring_study(kind: str, values=(1.7, 2.1)):
    exp, spring = planted(kind)
    return Study(exp).sweep(along({spring.k: [N(k) / m(1.0) for k in values]}), duration=s(STEPS * 1e-2)), spring


def codes(raised) -> list[str]:
    return [refusal.code for refusal in raised.value.refusals]


def run_at(kind: str, k: float) -> float:
    """The objective of a run of the experiment with its stiffness at ``k``, through a study's forward run."""
    exp, spring = planted(kind)
    result = Study(exp).sweep(along({spring.k: [N(k) / m(1.0)]}), duration=s(STEPS * 1e-2)).run()
    return float(position(result.member(0).final))


def test_a_leaf_twice_the_value_is_read_through_its_factor():
    """The ``double`` spring's node holds 2 k: the derivative in k is twice the one in its leaf, the central
    difference's that rebuilds."""
    study, _ = spring_study("double")
    h = 1e-4
    difference = (run_at("double", 1.7 * (1 + h)) - run_at("double", 1.7 * (1 - h))) / (2 * h * 1.7)
    for mode in ("forward", "reverse"):
        result = study.gradient(position, mode=mode)
        found = float(result.member(0).gradient[result.wrt[0]])
        assert abs(found / difference - 1) < 1e-7, (mode, found, difference)


@pytest.mark.parametrize(("kind", "code"), [("static", "gradient.value-keyed"), ("baked", "gradient.leaf-unread"),
                                            ("hidden", "gradient.no-leaf")])
def test_an_axis_the_program_does_not_read_as_a_leaf_is_refused(kind, code):
    """At one point (a sweep of several is refused by the plan already: ``study.leaf-unread``, ``study.axis-inert``),
    the derivative's own probe finds what the axis writes."""
    study, _ = spring_study(kind, values=(1.7,))
    with pytest.raises(BuildRefused) as raised:
        study.gradient(position)
    assert codes(raised) == [code]


@pytest.mark.parametrize(("kind", "value", "code"), [("squared", 1.3, "gradient.derived-leaf"),
                                                     ("held", 1.3, "gradient.held-leaf"),
                                                     ("bounded", BOUND, "gradient.probe-refused")])
def test_a_leaf_not_affine_in_its_axis_one_held_out_and_a_probe_past_a_refusal_are_refused(kind, value, code):
    """The planted springs (``planted_gradients.py``): the node holding k squared, a copy of k the derivative would
    drop, and k at the entry's own bound, past which the probe point is refused."""
    exp, spring = planted_gradient(kind)
    study = Study(exp).sweep(along({spring.k: [N(value) / m(1.0)]}), duration=s(STEPS * 1e-2))
    with pytest.raises(BuildRefused) as raised:
        study.gradient(position)
    assert codes(raised) == [code]


def test_the_same_springs_inside_their_lines_are_differentiated():
    """The bounded spring below its bound is the plain spring, and its derivative the double's halved: a control
    that the refusals above are the planted maps', not the rig's."""
    exp, spring = planted_gradient("bounded")
    found = Study(exp).sweep(along({spring.k: [N(1.3) / m(1.0)]}), duration=s(STEPS * 1e-2)).gradient(position)
    assert found.member(0).finite and found.member(0).gradient[found.wrt[0]] != 0.0


def test_no_axis_is_refused():
    exp, spring = planted("double")
    with pytest.raises(ArgumentRefused) as raised:
        Study(exp).sweep(along({spring.k: [N(1.7) / m(1.0)]}), duration=s(0.2)).gradient(position, wrt=[])
    assert codes(raised) == ["gradient.no-axis"]


def test_a_variant_an_axis_with_no_number_and_an_unknown_mode_are_refused():
    exp, spring = planted("double")
    study = Study(exp).sweep(along({DT: [ms(10.0)]}), duration=s(0.2))
    with pytest.raises(BuildRefused) as raised:
        study.gradient(position)
    assert codes(raised) == ["gradient.variant"]
    with pytest.raises(BuildRefused) as raised:
        Study(exp).sweep(along({spring.kind: ["double"]}), duration=s(0.2)).gradient(position)
    assert "gradient.not-a-number" in codes(raised)
    with pytest.raises(ArgumentRefused) as raised:
        Study(exp).sweep(along({spring.k: [N(1.7) / m(1.0)]}), duration=s(0.2)).gradient(position, mode="sideways")
    assert codes(raised) == ["gradient.mode"]


def test_statuses_compose_to_the_least_and_a_cut_or_unverified_entry_needs_its_acknowledgement():
    assert composed([]) is GradientStatus.EXACT
    assert composed([GradientStatus.EXACT, GradientStatus.PIECEWISE_KINK]) is GradientStatus.PIECEWISE_KINK
    assert composed([GradientStatus.FIXED_KEY, GradientStatus.EXACT]) is GradientStatus.FIXED_KEY
    assert composed([GradientStatus.UNVERIFIED, GradientStatus.CUT]) is GradientStatus.CUT
    path = {"a": GradientStatus.CUT, "b": GradientStatus.UNVERIFIED, "c": GradientStatus.PIECEWISE_JUMP}
    assert unacknowledged(path, {}) == ("a", "b")
    assert unacknowledged(path, {"a": "why", "b": "  "}) == ("b",)
    assert ACKNOWLEDGED_ONLY == {GradientStatus.CUT, GradientStatus.UNVERIFIED}


def magnet_loss(final):
    return final["umr"]["pose"][4]


@pytest.fixture(scope="module")
def magnet():
    """The spun magnet closed by a smooth follower, as a callback and inside the compiled step, differentiated in the
    rotor magnet's moment, which turns the body the follower reads, and through its writes the field again."""
    found = {}
    for mode in ("callback", "graph"):
        exp = spun_magnet(f"followed_{mode}", controller=Follower(), mode=mode)
        found[mode] = Study(exp).sweep(along({"spin.magnetic_moment": [_moment(exp)]}), duration=ms(30))
    return found


def _moment(exp):
    from mime.study._axes import base_value, resolve

    settings, declarations, _ = exp.declared_as()
    return base_value(resolve("spin.magnetic_moment", declarations), settings, declarations)


def test_a_callback_controllers_path_is_cut_refused_unless_acknowledged_and_never_silently_zero(magnet):
    with pytest.raises(BuildRefused) as raised:
        magnet["callback"].gradient(magnet_loss)
    assert codes(raised) == ["gradient.path-status"] and "cut" in raised.value.refusals[0].message
    owner = raised.value.refusals[0].message.split("passes through ")[1].split(",")[0]
    acknowledged = magnet["callback"].gradient(magnet_loss, acknowledged={owner: "its writes held as the run made them"})
    assert acknowledged.status is GradientStatus.CUT and acknowledged.path[owner] is GradientStatus.CUT
    assert acknowledged.acknowledged == {owner: "its writes held as the run made them"}
    forward = magnet["callback"].run().member(0).final
    member = acknowledged.member(0)
    assert member.value == pytest.approx(float(magnet_loss(forward)), rel=1e-12, abs=1e-15)
    graph = magnet["graph"].gradient(magnet_loss).member(0)
    assert graph.finite and member.finite
    assert np.all(member.gradient["spin.magnetic_moment"] != graph.gradient["spin.magnetic_moment"]), \
        "the cut path holds the controller's writes, so its derivative is not the closed loop's"
