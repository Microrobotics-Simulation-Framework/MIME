"""The experiments the views of an experiment (``built.to_mermaid()``, ``exp.tree()``) are tested and documented on: tutorial
A0 as ``examples/tutorial_a0.py`` declares it, and a toy with two bodies."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path

from mime.dynamics import Motor
from mime.experiment import Experiment
from mime.scene import Box, Z
from mime.testing import Declare, declare_in_order
from mime.testing.toys import (
    PROBE_START,
    ToyBody,
    ToyCouple,
    ToyDrag,
    ToyField,
    ToyHold,
    ToyRecorder,
    ToyRelaxation,
    ToySpring,
    toy_experiment,
)
from mime.things import Apparatus, Body, Medium, Part, PermanentMagnet
from mime.units import mm, rad, rad_per_s

ROOT = Path(__file__).resolve().parents[1]


def tutorial_a0() -> Experiment:
    """Tutorial A0's experiment as ``examples/tutorial_a0.py`` declares it, without its run: the file's source up to its
    ``run`` region, executed, with ``script_a`` the examples' (the tests have a script A of their own under that name)."""
    spec = importlib.util.spec_from_file_location("example_script_a", ROOT / "examples" / "script_a.py")
    assert spec is not None and spec.loader is not None
    examples_script_a = importlib.util.module_from_spec(spec)
    sys.modules["example_script_a"] = examples_script_a
    spec.loader.exec_module(examples_script_a)
    source = (ROOT / "examples" / "tutorial_a0.py").read_text(encoding="utf-8").split("# region: run")[0]
    held = sys.modules.get("script_a")
    sys.modules["script_a"] = examples_script_a
    try:
        names = {"__name__": "tutorial_a0_declarations", "__file__": str(ROOT / "examples" / "tutorial_a0.py")}
        exec(compile(source, "examples/tutorial_a0.py", "exec"), names)  # noqa: S102 - the repository's own example
    finally:
        sys.modules.pop("example_script_a", None)
        if held is None:
            sys.modules.pop("script_a", None)
        else:
            sys.modules["script_a"] = held
    return names["close_the_loop"](names["a0"]())


def two_body_declarations() -> tuple[Declare, ...]:
    """An experiment with two of everything the tree and the graph bind a law to, so that a law bound to the wrong one
    changes what they show: two bodies, two media, two fields of one apparatus, and a chain, parts and fields declared in
    an order that is not alphabetical (so that an order the views sort, or keep, shows). The body that sorts last is the
    one the first declaration names, and each law is bound across (``zed`` to ``bath``, ``abe`` to ``oil``)."""
    box = lambda: Box(size=mm((1.0, 1.0, 1.0)))  # noqa: E731 - three uses, one line each
    rig = Apparatus(
        "rig", fields={"push": ToyField(), "pull": ToyField()},
        chain={"zeta": Motor.Kinematic(axis=Z, prescribed="velocity", initial_angle=rad(0.0)),
               "alpha": Motor.Kinematic(axis=Z, prescribed="velocity", initial_angle=rad(0.0), mount="zeta.rotor")},
        parts={"yoke": Part(box(), mount="zeta.rotor"), "cap": Part(box(), mount="alpha.rotor")},
        magnets={"yoke": PermanentMagnet(direction=(1.0, 0.0, 0.0))},
        commands={"zeta.velocity": rad_per_s(1.0), "alpha.velocity": rad_per_s(2.0)})
    return (
        Declare("add_apparatus(rig)", lambda e: e.add_apparatus(rig)),
        Declare("add_medium(oil)", lambda e: e.add_medium(Medium("oil", flow=ToyRelaxation()))),
        Declare("add_medium(bath)", lambda e: e.add_medium(Medium("bath", flow=ToyRelaxation()))),
        Declare("add_body(zed)", lambda e: e.add_body(
            Body("zed", dynamics=ToyBody(), initial=PROBE_START, parts={"shell": Part(box()), "cap": Part(box())}))),
        Declare("add_body(abe)", lambda e: e.add_body(Body("abe", dynamics=ToyBody(), initial=PROBE_START))),
        Declare("attach(couple, zed)", lambda e: e.attach(ToyCouple(), body="zed", medium="bath")),
        Declare("attach(couple, abe)", lambda e: e.attach(ToyCouple(), body="abe", medium="oil")),
        Declare("attach(hold, zed)", lambda e: e.attach(ToyHold(), body="zed", field="rig.push")),
        Declare("attach(hold, abe)", lambda e: e.attach(ToyHold(), body="abe", field="rig.pull")),
        Declare("attach(drag, zed)", lambda e: e.attach(ToyDrag.Quadratic(), body="zed")),
        Declare("attach(spring, abe)", lambda e: e.attach(ToySpring(), body="abe")),
        Declare("observe(recorder, abe)", lambda e: e.observe(ToyRecorder(body="abe"))),
        Declare("observe(recorder, zed)", lambda e: e.observe(ToyRecorder(body="zed"))),
    )


def two_bodies(order: Sequence[int] | None = None) -> Experiment:
    return declare_in_order(lambda: toy_experiment("two_bodies"), two_body_declarations(), order)
