"""What a script's own ``try/except`` sees of an error that crosses a public call: a bare number is the user's
refusal and still a ``ValueError``; an error MIME raised with no code keeps its builtin class and is not called a
defect; and an error from below (the script's own exception from a hook or a controller's step, an ``OSError``, a
``MemoryError``, a warning raised as an error) is caught by the handler written for it, as before the boundary.

Each catch has its control: the same check with the translation's class left out (``standing_for`` a no-op, MIME's
classes alone, as phase 7l-b first shipped them) must fail, so that the check is one that can."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from control_cases import spun_magnet

from mime import Body, Experiment, State
from mime.build import ArgumentRefused, InternalError, MimeError, RefusalClass, UnrecognisedError
from mime.dynamics import RigidInertial
from mime.errors import TRACEBACK_VARIABLE, public, short_text, translated
from mime.errors import _boundary
from mime.errors._translate import UNCODED
from mime.gradients import GradientStatus
from mime.observations import SwimSpeed
from mime.registry import ControlLaw, Controller, GradientRecord, Signal, register_controller
from mime.scene import Scene, Z
from mime.units import UnitError, kg, kg_m2, m_per_s, m_per_s2, mm, ms, rad_per_s


class Done(Exception):
    """A script's own exception, ending a run early, with an attribute of its own."""

    def __init__(self, tick: int) -> None:
        super().__init__(f"done at tick {tick}")
        self.tick = tick


@register_controller
@dataclass(frozen=True, kw_only=True)
class Ender(Controller):
    """A callback controller whose step raises the script's own :class:`Done` at its tick 1."""

    name = "oracle.ender"
    version = "0.1.0"
    differentiability = GradientStatus.CUT
    gradient = GradientRecord(where="its step", reason="a test's callback that raises", route="none: it raises")
    builds = {"": ()}
    reads = {"pose": Signal("m; 1", (7,))}
    writes = {"angle": Signal("1", ())}

    def law(self, ctx: Any) -> ControlLaw:
        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            if int(state["ticks"]) == 1:
                raise Done(1)
            return {"ticks": int(state["ticks"]) + 1}, {"angle": np.asarray(0.0)}

        return ControlLaw(state={"ticks": 0}, held={"angle": np.asarray(0.0)}, step=step)


def rod() -> tuple[Experiment, Any]:
    exp = Experiment("drift", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
    exp.add_body(Body("rod", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    exp.declare("rod.mass", kg(1e-6), reason="a millimetre rod")
    exp.declare("rod.inertia", kg_m2((1e-13, 1e-13, 1e-13)), reason="a millimetre rod")
    return exp, exp.observe(SwimSpeed("rod", along=Z))


@pytest.fixture(scope="module")
def built() -> tuple[Any, Any]:
    exp, drift = rod()
    return exp.build(), drift


def _raised(action: Callable[[], object]) -> BaseException:
    try:
        action()
    except Exception as error:  # noqa: BLE001 - the test's own catch of what it planted
        return error
    raise AssertionError("the action did not raise")


def caught(action: Callable[[], object], kind: type[BaseException]) -> bool:
    """Whether a handler written for ``kind`` catches what ``action`` raises."""
    try:
        action()
    except kind:
        return True
    except Exception:  # noqa: BLE001 - anything else is what the check reports as not caught
        return False
    return False


@pytest.fixture
def as_first_shipped(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The control: the boundary's errors of MIME's classes alone, as phase 7l-b first shipped them."""
    from mime.build import _refusals
    from mime.control import _run

    def alone(coded: MimeError, original: BaseException) -> MimeError:
        return coded

    for module in (_boundary, _run, _refusals):
        monkeypatch.setattr(module, "standing_for", alone)
    yield


# -- a bare number --------------------------------------------------------------------------------------------------


def bare_numbers(built: tuple[Any, Any]) -> dict[str, Callable[[], object]]:
    held, drift = built
    return {"a run's duration": lambda: held.run(0.002),
            "every=": lambda: held.run(ms(2), record=[drift.speed], every=0.001),
            "a window's start": lambda: held.run(ms(2), record=[drift.speed])[drift].speed.between(0.0005)}


def is_the_users_value_error(error: BaseException) -> bool:
    return (isinstance(error, ValueError) and isinstance(error, UnitError) and isinstance(error, MimeError)
            and error.codes() == ["call.units"] and error.refusals[0].category is RefusalClass.USER
            and "from mime.units import s" in error.refusals[0].action)


def test_a_bare_number_where_a_time_is_wanted_is_the_users_refusal_and_still_a_value_error(built):
    for what, action in bare_numbers(built).items():
        error = _raised(action)
        assert is_the_users_value_error(error), (what, error)
        assert isinstance(error, ArgumentRefused) and "the bare value" in error.refusals[0].message, what
    assert "s(0.002)" in _raised(bare_numbers(built)["a run's duration"]).refusals[0].action  # type: ignore[union-attr]


def test_the_control_without_its_translation_a_bare_number_is_not_the_users(built, monkeypatch):
    from mime.errors import _translate

    monkeypatch.setattr(_translate, "TRANSLATIONS", tuple(t for t in _translate.TRANSLATIONS
                                                          if t is not _translate.unit_error))
    for what, action in bare_numbers(built).items():
        assert not is_the_users_value_error(_raised(action)), what


def _planted_in_mime(error: BaseException) -> Callable[[], object]:
    """A function whose module is MIME's, raising ``error`` with no code: a gap in MIME's codes, planted."""
    namespace: dict[str, Any] = {"__name__": "mime._planted_gap", "error": error}
    exec(compile("def gap():\n    raise error\n", "/planted/mime/_planted_gap.py", "exec"), namespace)  # noqa: S102
    return namespace["gap"]


def test_an_error_mime_raised_with_no_code_keeps_its_class_and_is_not_called_a_defect(tmp_path, monkeypatch):
    monkeypatch.setenv("MIME_TRACEBACK_DIR", str(tmp_path))
    gap = public(_planted_in_mime(KeyError("speed")))
    error = _raised(gap)
    assert isinstance(error, KeyError) and isinstance(error, InternalError) and error.codes() == ["error.uncoded"]
    shown = short_text(error, None, frames=False)
    assert UNCODED in shown and "This is a defect in MIME" not in shown
    assert "not a fault of yours" in (error.refusals[0].why or "")


def test_the_control_an_uncoded_error_without_its_class_is_not_caught(monkeypatch, as_first_shipped):
    assert not caught(public(_planted_in_mime(KeyError("speed"))), KeyError)


def test_an_error_of_mimes_raised_inside_a_library_mime_called_is_mimes():
    """``did_you_mean`` handing ``difflib`` a number: the frame that raised is the standard library's, the frame that
    called it MIME's, so the error is MIME's gap and not one from below."""
    from mime.things import did_you_mean

    error = _raised(lambda: did_you_mean(3, ["speed"]))  # type: ignore[arg-type]
    assert translated(error).codes() == ["error.uncoded"]
    assert translated(_raised(lambda: np.linalg.inv(np.zeros((2, 2))))).codes() == ["error.unrecognised"]


# -- errors from below, caught as they were --------------------------------------------------------------------------


def from_below(built: tuple[Any, Any], tmp_path: Path) -> dict[str, tuple[Callable[[], object], type[BaseException]]]:
    """The review's five, each a script's handler around a public call, with the class it is written for."""
    held, _ = built

    def done(chunk: Any) -> None:
        raise Done(chunk.index)

    def memory(chunk: Any) -> None:
        raise MemoryError("planted: the host's own")

    def warned(chunk: Any) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warnings.warn("planted: a warning the script's filter raises", UserWarning, stacklevel=1)

    blocked = tmp_path / "a file"
    blocked.write_text("not a folder", encoding="utf-8")
    callback = spun_magnet("ended", controller=Ender(), rate=ms(1)).build()
    return {"a hook's own exception": (lambda: held.run(ms(4), hook=done, chunk=ms(2)), Done),
            "a callback law's own exception": (lambda: callback.run(ms(4)), Done),
            "save where a folder cannot be made": (lambda: held.save(blocked / "saved", steps=2), OSError),
            "the host's MemoryError": (lambda: held.run(ms(4), hook=memory, chunk=ms(2)), MemoryError),
            "a warning raised as an error": (lambda: held.run(ms(4), hook=warned, chunk=ms(2)), UserWarning)}


def test_an_error_from_below_is_caught_by_the_handler_written_for_it(built, tmp_path):
    for what, (action, kind) in from_below(built, tmp_path).items():
        assert caught(action, kind), what
        error = _raised(action)
        assert isinstance(error, MimeError) and error.original is not None and isinstance(error.original, kind), what
    ended = _raised(from_below(built, tmp_path)["a hook's own exception"][0])
    assert isinstance(ended, Done) and ended.tick == 0 and ended.codes() == ["error.unrecognised"]
    law = _raised(from_below(built, tmp_path)["a callback law's own exception"][0])
    assert isinstance(law, Done) and law.codes() == ["control.law-raised"] and law.tick == 1
    memory = _raised(from_below(built, tmp_path)["the host's MemoryError"][0])
    assert memory.codes() == ["error.unrecognised"], "the host's own memory is not the device's"  # type: ignore[union-attr]
    saved = _raised(from_below(built, tmp_path)["save where a folder cannot be made"][0])
    assert saved.codes() == ["save.not-writable"] and saved.refusals[0].category is RefusalClass.ENVIRONMENT  # type: ignore[union-attr]


def test_the_control_without_the_originals_class_none_of_them_is_caught(built, tmp_path, as_first_shipped):
    for what, (action, kind) in from_below(built, tmp_path).items():
        assert not caught(action, kind), what


def test_a_users_pytest_raises_around_a_run_whose_hook_raises_still_passes(built):
    held, _ = built

    def hook(chunk: Any) -> None:
        raise ValueError("the script's own")

    with pytest.raises(ValueError, match="the script's own"):
        held.run(ms(4), hook=hook, chunk=ms(2))


def test_the_control_a_users_pytest_raises_fails_on_mimes_class_alone(built, as_first_shipped):
    held, _ = built

    def hook(chunk: Any) -> None:
        raise ValueError("the script's own")

    assert not caught(lambda: held.run(ms(4), hook=hook, chunk=ms(2)), ValueError)


def test_a_class_that_cannot_be_a_base_leaves_the_original_unchanged(monkeypatch):
    monkeypatch.setenv(TRACEBACK_VARIABLE, "short")

    class Fixed(Exception):
        def __init_subclass__(cls, **kwargs: Any) -> None:
            raise TypeError("Fixed is final")

    planted = Fixed("planted")

    @public
    def call() -> None:
        raise planted

    assert _raised(call) is planted


def test_an_error_standing_for_one_from_below_pickles_and_keeps_both_classes():
    error = _raised(public(lambda: (_ for _ in ()).throw(Done(4))))
    assert isinstance(error, UnrecognisedError) and isinstance(error, Done)
    back = pickle.loads(pickle.dumps(error))  # noqa: S301 - the test's own object
    assert isinstance(back, UnrecognisedError) and isinstance(back, Done) and back.tick == 4
    assert back.codes() == ["error.unrecognised"] and str(back) == str(error)
