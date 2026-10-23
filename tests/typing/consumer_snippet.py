# pyright: strict, reportUnnecessaryTypeIgnoreComment=error
"""A consumer's code, type-checked by CI against the installed wheel rather than the source tree; never run.

Strict pyright refuses the import of a package without ``py.typed``, and ``assert_type`` fails when a public type
changes. A write into the read-only lock must be refused: were it ever accepted, the ignore on that line would be
unnecessary, which the line above makes an error. The same holds for each ``pyright: ignore[rule]`` below, which must
meet that rule on its line: the typed node state convention, whose closed TypedDicts refuse a missing or extra field
and a view of a TypedDict that is not closed, and the override rule that makes the convention necessary.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, assert_type, override

from jax import Array
from jaxtyping import Float
from maddening.core.node import SimulationNode
from typing_extensions import ReadOnly, TypedDict

import mime
from mime import Experiment
from mime.build import BuildRefused, BuildReport
from mime.core import NodeFields, node_fields, typed_view
from mime.experiment import BuiltExperiment
from mime.saved import ExperimentDocument, LoadReport, Lock, SceneDoc, VerifyReport
from mime.scene import Scene, Z
from mime.units import Dimension, Force, Length, Quantity, mm, s

experiment = Experiment("snippet", scene=Scene.new(up=Z, units="m"), dt=s(0.01))
assert_type(experiment, Experiment)
assert_type(mm(3.0), Quantity)
assert_type(mime.__version__, str)

document = experiment.document()
assert_type(document, ExperimentDocument)
assert_type(document["experiment"], str)
assert_type(document["scene"], SceneDoc)

loaded = Experiment.load(Path("a-saved-experiment"))
assert_type(loaded.load_report, LoadReport | None)
built = loaded.build()
assert_type(built, BuiltExperiment)
assert_type(built.report, BuildReport)
assert_type(built.report.run.text(), str)
assert_type(built.verify(), VerifyReport)
assert_type(built.verify().reproduced, bool)

lock = built.save("elsewhere")
assert_type(lock, Lock)
assert_type(lock["hashes"]["derived"], str)
assert_type(lock["environment"]["x64"], bool)
lock["hashes"]["derived"] = "sha256:0"  # type: ignore

try:
    experiment.build()
except BuildRefused as refused:
    assert_type(refused.codes(), list[str])

assert_type(Force / Length, Dimension)


class PointState(TypedDict, closed=True):
    x: ReadOnly[Float[Array, "3"]]
    v: ReadOnly[Float[Array, "3"]]


class OpenState(TypedDict):
    x: Array


class Point(SimulationNode):
    @override
    def initial_state(self) -> NodeFields:
        x = typed_view(PointState, {})["x"]
        return node_fields(PointState(x=x, v=x))

    @override
    def update(self, state: NodeFields, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeFields:
        held = typed_view(PointState, state)
        assert_type(held["v"], Array)
        held["x"] = held["v"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
        node_fields(PointState(x=held["x"]))  # pyright: ignore[reportCallIssue]
        node_fields(PointState(x=held["x"], v=held["v"], w=held["v"]))  # pyright: ignore[reportCallIssue]
        typed_view(OpenState, state)  # pyright: ignore[reportArgumentType]
        return node_fields(PointState(x=held["x"] + dt * held["v"], v=held["v"]))


class Narrowed(SimulationNode):
    @override
    def initial_state(self) -> NodeFields:
        return {}

    @override
    def update(  # pyright: ignore[reportIncompatibleMethodOverride]
            self, state: PointState, boundary_inputs: Mapping[str, Array], dt: float) -> NodeFields:
        return node_fields(state)
