"""Gate 1's case on script A, shared by the float32 and float64 modules: eight locked points over the field's frequency
and the body's mass, each member against ``run(override=...)`` of its point on a fresh build, with the controls."""

from __future__ import annotations

import dataclasses

from compare import TOLERANCE_EPS, miss_eps, misses, worst
from script_a import reproduction_arm

from mime import Study
from mime.study import grid
from mime.study._study import run_members
from mime.units import Hz, kg, s

SPEED = "swim_speed:umr.speed"
FREQUENCIES = (10.0, 20.0, 30.0, 40.0)
MASSES = (8.0e-5, 9.0e-5)
DURATION = s(1.0)


def gate_one(dtype: str) -> dict[str, object]:
    """The study, each member's misses against its own run, and the two controls' misses: what the module asserts."""
    exp = reproduction_arm()
    field = exp.handle("field")
    axis = field.commands.frequency  # type: ignore[union-attr]
    points = grid({axis: [Hz(f) for f in FREQUENCIES], "umr.mass": [kg(m) for m in MASSES]})
    study = Study(exp).sweep(points, duration=DURATION, record=[SPEED])
    plan = study.plan()
    result = study.run()
    references = []
    for index in range(len(result)):
        values = dict(zip(plan.axes, (plan.points[index].values[a] for a in plan.axes), strict=True))
        override = {(axis if a.path == str(axis) else a.path): v for a, v in values.items()}
        references.append(reproduction_arm().build().run(DURATION, history=True, override=override))
    found = [misses(references[i], result.member(i), [SPEED]) for i in range(len(result))]
    # The controls. A planted move of one member's frequency leaf by 1e-6 relative, run in the batch as the study runs
    # it; and every member read against its neighbour's run.
    group = plan.groups[0]
    assert group.built is not None
    moved = [dataclasses.replace(p) for p in plan.points]
    node = "drive:field.frequency"
    target = 1
    tree = {n: dict(leaves) for n, leaves in moved[target].tree.items()}
    tree[node]["value"] = (tree[node]["value"] * (1 + 1e-6)).astype(tree[node]["value"].dtype)
    moved[target].tree = tree
    final, _, _ = run_members(group.built, moved, group.members, (*group.varying, (node, "value")) if
                              (node, "value") not in group.varying else group.varying, group.chunks,
                              group.built.flag_tests(), ())
    history = references[target].history
    assert history is not None
    planted = {f"final {n}.{f}": miss_eps(references[target].final[n][f], final[n][f][target], history[n][f])
               for n in final for f in final[n]}
    shifted = [worst(misses(references[(i + 1) % len(result)], result.member(i)))[1] for i in range(len(result))]
    return {"tolerance": TOLERANCE_EPS[dtype], "found": found, "plan": plan, "result": result, "planted": planted,
            "references": references, "shifted": shifted, "moved_member": target}
