"""A controller's run flags (``mime.nodes.control._flags``) through the real path: declared in its class's ``raises``,
raised by ``mime.control.raise_flag`` in its step, read at one path, ``controller.<name>``, inside the compiled step and
as a callback (whose holder node is the build's own), counted and warned of by the run, carried by a study's members
and by ``Study.gradient``'s; and refused, with the fix named, where they would be lost: raised inside a ``jax.lax``
body, under a name not declared, or by a step compiled with no function underneath it.

The rig is the coil sphere of ``believed_cases`` with :class:`iterative_cases.FlaggingPD`, which raises
``far_from_goal`` while the sphere is more than 3 mm from its goal (its first ticks: it starts 3.35 mm away).

**Its plant** (``FLAG_PLANT=unleveled``): ``raise_flag`` without its check of the level it is called at, which the
test of a raise inside a ``jax.lax`` body must catch on every run (the value is then lost, or leaks as JAX's own error,
never the rule's message)."""

from __future__ import annotations

import os
import warnings
from typing import Any

import jax
import numpy as np
import pytest
from believed_cases import believed_sphere
from iterative_cases import FAR, FlaggingPD

from mime.build import BuildRefused, MimeError
from mime.units import ms

PLANT = os.environ.get("FLAG_PLANT", "")
PATH = f"controller.{FAR.field}"
#: The rule's words, which a raise below the step's level must name.
RULE = "raise at the step's own level; return the value out of the branch or loop and raise it there"


@pytest.fixture(autouse=True)
def float64() -> Any:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


@pytest.fixture
def plant(monkeypatch: pytest.MonkeyPatch) -> None:
    if PLANT == "unleveled":
        from mime.nodes.control import _flags

        def unleveled(name: str, value: Any) -> None:
            gathering = _flags._raised.get()  # pyright: ignore[reportPrivateUsage]
            if gathering is not None:
                gathering.held[name].append(value)

        monkeypatch.setattr(_flags, "raise_flag", unleveled)
        import iterative_cases

        monkeypatch.setattr(iterative_cases, "raise_flag", unleveled)


def run(mode: str, duration: float = 30.0, **settings: Any) -> tuple[Any, Any, list[str]]:
    built = believed_sphere(f"flagging_{mode}", mode=mode, controller=FlaggingPD(**settings)).build()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = built.run(ms(duration), history=True)
    return built, result, [str(w.message) for w in caught]


def test_a_declared_flag_reads_one_path_in_both_modes_counted_and_warned():
    """Inside the compiled step, as a callback, and as a callback whose step the law compiled with ``jax.jit``: the
    layout, the report and the run read ``controller.far_from_goal``, raised at the same ticks, and the run warns of
    it; its value is the distance."""
    seen = {}
    for mode, compiled in (("graph", ""), ("callback", ""), ("callback", "jit")):
        built, result, said = run(mode, compiled=compiled)
        assert [flag.path for flag in built.layout.flags if flag.owner == "controller"] == [PATH], mode
        assert f"    {PATH} (warns): " in "\n".join(built.report.controller)
        (read,) = [read for read in result.flags if read.path == PATH]
        assert read.raised is not None and 0 < read.raised < read.steps, (mode, compiled, read)
        assert any(PATH in text for text in said), said
        seen[(mode, compiled)] = (read.raised, read.first, read.last)
    assert len(set(seen.values())) == 1, seen


def test_a_study_and_its_derivative_carry_the_flag_at_its_path():
    """A sweep's members and ``Study.gradient``'s each read the flag at ``controller.far_from_goal``: raised where the
    level is below the sphere's start (3.35 mm from its goal), never where it is above."""
    from mime.study import Study, grid

    levels = [3e-3, 4e-3]
    done = Study(believed_sphere("flagging_study", controller=FlaggingPD())).sweep(
        grid({"controller.level": levels}), duration=ms(30)).run()
    raised = [{read.path for read in done.member(i).flags if read.was_raised} for i in range(2)]
    assert raised == [{PATH}, set()], raised
    gradient = Study(believed_sphere("flagging_gradient", controller=FlaggingPD())).sweep(
        grid({"controller.kp": [2500.0], "controller.level": levels}), duration=ms(30)).gradient(
        lambda final: final["sphere"]["pose"][0], wrt=["controller.kp"])
    carried = [{flag.path for flag in gradient.member(i).flags} for i in range(2)]
    assert carried == [{PATH}, set()], carried
    assert gradient.member(0).flags[0].meaning == FAR.meaning


@pytest.mark.parametrize("inside", ["cond", "while_loop", "scan", "fori_loop"])
@pytest.mark.parametrize("mode", ["graph", "callback"])
def test_a_flag_raised_inside_a_loop_or_branch_body_is_refused_naming_the_rule(mode, inside, plant):
    """Its value would not leave the body: the build refuses it inside the compiled step (``controller.untraceable``)
    and a callback's first tick does (``control.law-raised``), each naming the rule."""
    experiment = believed_sphere(f"inside_{mode}", mode=mode, controller=FlaggingPD(inside=inside))
    if mode == "graph":
        with pytest.raises(BuildRefused) as refused:
            experiment.build()
        assert refused.value.codes() == ["controller.untraceable"], refused.value.codes()
        assert RULE in str(refused.value), str(refused.value)
        return
    built = experiment.build()
    with pytest.raises(MimeError) as raised:
        built.run(ms(10))
    assert raised.value.codes() == ["control.law-raised"] and RULE in str(raised.value), str(raised.value)


@pytest.mark.parametrize("mode", ["graph", "callback"])
def test_a_flag_not_declared_is_refused_naming_its_declaration(mode):
    experiment = believed_sphere(f"undeclared_{mode}", mode=mode, controller=FlaggingPD(raised="not_declared"))
    fix = "declare `not_declared` in the controller's `raises` (a `Flag(field=..., meaning=..., warns=...)`)"
    if mode == "graph":
        with pytest.raises(BuildRefused) as refused:
            experiment.build()
        assert refused.value.codes() == ["controller.untraceable"] and fix in str(refused.value), str(refused.value)
        return
    built = experiment.build()
    with pytest.raises(MimeError) as raised:
        built.run(ms(10))
    assert raised.value.codes() == ["control.law-raised"] and fix in str(raised.value), str(raised.value)


@pytest.mark.parametrize("mode", ["graph", "callback"])
def test_a_step_compiled_with_nothing_underneath_is_refused_where_it_declares_flags(mode):
    """A step compiled ahead of time has no function to trace with the flags gathered: refused at the build, never a
    flag lost. The same law's step compiled by ``jax.jit`` builds (the first test runs it)."""
    with pytest.raises(BuildRefused) as refused:
        believed_sphere(f"aot_{mode}", mode=mode, controller=FlaggingPD(compiled="aot")).build()
    assert "no __wrapped__" in str(refused.value) and "far_from_goal" in str(refused.value), str(refused.value)
    believed_sphere(f"jit_{mode}", mode=mode, controller=FlaggingPD(compiled="jit")).build()


def test_raise_flag_outside_a_runs_step_does_nothing():
    from mime.control import raise_flag

    assert raise_flag("anything", np.float64(1.0)) is None


__all__: list[str] = []
