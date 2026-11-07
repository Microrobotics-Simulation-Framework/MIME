"""Gates 2 and 3. A group compiles once, whatever its size; the structure key is found from what each point assembles,
so a planted node whose static array is derived from the swept leaf splits the study into one group per value, and one
that bakes the leaf into its trace without declaring it is refused as changing nothing in the compiled step. Every
member's parameter tree is the tree its own full build gives, leaf by leaf, for every kind of axis, and a planted entry
that derives its leaf at build (twice the value) shows the derived value, which a sweep writing the raw value would
not."""

from __future__ import annotations

import numpy as np
import pytest
from planted import planted
from script_a import reproduction_arm

from mime import Study
from mime.build import BuildRefused
from mime.study import grid
from mime.study._study import Compiles
from mime.testing._saved import _compilations  # noqa: PLC2701 - the build's own counter of XLA compiles
from mime.units import Hz, N, kg, m, mm, s

K = [N(k) / m(1.0) for k in (1.1, 1.5, 1.9, 2.3, 2.7, 3.1, 3.5, 3.9)]


def _spring_study(kind: str, values):
    exp, spring = planted(kind)
    return Study(exp).sweep(grid({spring.k: values}), duration=s(0.4))


def test_a_group_of_eight_points_compiles_once_and_a_group_of_three_as_often():
    eight = _spring_study("double", K)
    plan = eight.plan()
    assert len(plan.groups) == 1 and plan.compiles == 1
    with _compilations() as seen:
        result = eight.run()
    three = _spring_study("double", K[:3])
    three.plan()
    with _compilations() as seen_three:
        three.run()
    assert result.compiles == (1,)
    assert len(seen) == len(seen_three), (len(seen), len(seen_three))


def test_a_static_derived_from_the_swept_leaf_splits_the_study_into_one_group_per_value():
    plan = _spring_study("static", K[:3]).plan()
    assert len(plan.groups) == 3, plan.text()
    assert all(group.compiles == 1 for group in plan.groups)
    assert any("static" in reason for reason in plan.groups[1].reasons), plan.groups[1].reasons
    result = _spring_study("static", K[:3]).run()
    assert result.groups == (0, 1, 2) and result.compiles == (1, 1, 1)


def test_a_leaf_baked_into_the_trace_without_its_declaration_is_refused():
    with pytest.raises(BuildRefused) as refused:
        _spring_study("baked", K[:3]).plan()
    assert "study.leaf-unread" in refused.value.codes()
    assert "changes nothing in the compiled step" in refused.value.refusals[0].message


def _trees_equal(point, built) -> list[str]:
    own = built.graph.params["nodes"]
    found = []
    for node, leaves in point.tree.items():
        for leaf, value in leaves.items():
            held = np.asarray(own[node][leaf])
            if value.shape != held.shape or value.dtype != held.dtype or not np.array_equal(value, held):
                found.append(f"{node}.{leaf}")
    return found


AXES = {
    "a command": ("field.commands.frequency", [Hz(12.0), Hz(25.0)]),
    "a thing's value": ("umr.mass", [kg(8.2e-5), kg(8.8e-5)]),
    "a law's setting": ("drag:umr.lubrication.epsilon", [mm(0.12), mm(0.2)]),
    "an initial state": ("umr.initial.position", [mm((0.0, 0.0, -1.0)), mm((0.0, 0.3, -0.9))]),
}


@pytest.mark.parametrize("kind", sorted(AXES))
def test_each_members_tree_and_state_are_its_own_full_builds(kind):
    from mime.experiment import Experiment

    path, values = AXES[kind]
    plan = Study(reproduction_arm()).sweep(grid({path: values}), duration=s(0.01)).plan()
    assert not plan.refused, plan.text()
    for point in plan.points:
        built = Experiment.declared_from(point.declared.settings, point.declared.declarations).build()
        assert not _trees_equal(point, built), (kind, _trees_equal(point, built))
        state = {name: built.graph.get_node(name).initial_state() for name in built.graph.node_names}
        assert all(np.array_equal(point.state[n][f], np.asarray(v)) for n in state for f, v in state[n].items())
    first, second = (plan.points[0].tree, plan.points[1].tree)
    moved = [(n, k) for n in first for k in first[n] if not np.array_equal(first[n][k], second[n][k])]
    assert moved or kind == "an initial state", kind


def test_a_leaf_the_build_derives_holds_the_derived_value_in_every_member():
    """The planted entry stores twice the stiffness: each member's leaf is twice its point's value, as its own build's
    is, and the raw value a sweep writing straight into the leaf would give differs from both."""
    from mime.experiment import Experiment

    plan = _spring_study("double", K[:3]).plan()
    for point, value in zip(plan.points, K[:3], strict=True):
        leaf = point.tree["planted.spring:probe"]["k"]
        built = Experiment.declared_from(point.declared.settings, point.declared.declarations).build()
        assert np.array_equal(leaf, np.asarray(built.graph.params["nodes"]["planted.spring:probe"]["k"]))
        assert np.isclose(float(leaf), 2 * float(value.value)) and not np.isclose(float(leaf), float(value.value))


def test_the_compile_counter_hears_a_compile():
    """The control for the counts above: a new program, compiled, is heard."""
    import jax
    import jax.numpy as jnp

    with Compiles() as compiled:
        jax.jit(lambda x: x * 3.0 + 1.0)(jnp.arange(7.0)).block_until_ready()
    assert compiled.count >= 1


def test_a_point_its_full_build_refuses_is_refused_with_every_point_that_shares_its_verdict(monkeypatch):
    """What only the full build judges (a node's own check at the start, the post-build checks) refuses the group's
    first point, and every point whose checked nodes it shared: here the first build is made to refuse, and the toy's
    points have no node with checks of its own, so all three share its verdict and are listed refused, none raising."""
    from mime.build import Refusal, RefusalClass
    from mime.experiment import Experiment

    original = Experiment.build
    calls = []

    def build(self):
        calls.append(self)
        if len(calls) == 1:
            raise BuildRefused([Refusal("drag.mrs-resistance", RefusalClass.USER, "planted", "planted")])
        return original(self)

    monkeypatch.setattr(Experiment, "build", build)
    plan = _spring_study("double", K[:3]).plan()
    assert [point.refused for point in plan.points] == [True, True, True] and not plan.groups, plan.text()
    assert len(calls) == 1
