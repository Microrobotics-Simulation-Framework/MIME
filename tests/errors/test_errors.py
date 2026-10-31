"""MIME's errors at the boundary of its public calls: the one shape, the classes as they are caught, the boundary's
translation and its traceback, the display in a script, IPython and the CLI, the full chain's file, the errors of a
run and of a result, and the errors from below translated.

The suite runs in full mode (``tests/conftest.py``); a test of the short form sets ``MIME_TRACEBACK`` itself."""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mime import Body, Experiment, State
from mime.build import (
    ArgumentRefused,
    ArgumentTypeRefused,
    BuildRefused,
    InternalError,
    MimeError,
    Refusal,
    RefusalClass,
    RunFailed,
    UnrecognisedError,
)
from mime.build._refusals import Explained, page_of, plain_possessive
from mime.dynamics import RigidInertial
from mime.errors import (
    DIRECTORY_VARIABLE,
    KEPT,
    TRACEBACK_VARIABLE,
    crossed,
    is_library,
    public,
    script_traceback,
    short_text,
    translated,
    write_traceback,
)
from mime.errors import _display
from mime.errors._translate import first_lines, out_of_memory, traced_value_error
from mime.observations import SwimSpeed
from mime.results import NoOutput, NotARunFolder, NotInResult, NotRecorded, RunResult
from mime.scene import Scene, Z
from mime.units import kg, kg_m2, m3, m_per_s, m_per_s2, mm, ms, rad_per_s, s

HERE = Path(__file__).resolve()


def rod(dt: float = 1.0, spin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[Experiment, Any]:
    """A millimetre rod alone, its dynamics its own: the cheapest experiment that builds and runs; with a spin and a
    base step far above its period, its explicit gyroscopic term grows without bound."""
    exp = Experiment("drift", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(dt))
    exp.add_body(Body("rod", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s(spin))))
    exp.declare("rod.mass", kg(1e-6), reason="a millimetre rod")
    exp.declare("rod.inertia", kg_m2((1e-13, 2e-13, 3e-13)), reason="a millimetre rod, three different axes")
    return exp, exp.observe(SwimSpeed("rod", along=Z))


@pytest.fixture(scope="module")
def built():
    exp, drift = rod()
    return exp.build(), drift


def refusal(code: str = "frame.gravity-missing", **extra: Any) -> Refusal:
    return Refusal(code, RefusalClass.USER, "the experiment declares no gravity", "pass gravity=standard_gravity()",
                   **extra)


# -- the shape -------------------------------------------------------------------------------------------------------


def test_a_refusal_says_what_why_what_to_do_where_and_its_page():
    lines = refusal(why="gravity has no default", where="settle.py, line 3").lines()
    assert lines == ["[user] frame.gravity-missing: the experiment declares no gravity",
                     "    why: gravity has no default", "    do: pass gravity=standard_gravity()",
                     "    at: settle.py, line 3", f"    see: {page_of('frame.gravity-missing')}"]
    assert refusal().lines()[1].startswith("    do:"), "a refusal with no why and no line has neither"


def test_a_quoted_name_with_a_possessive_reads_plain_and_a_quoted_name_alone_keeps_its_quotes():
    assert plain_possessive("body 'umr''s part 'shell''s solid") == "body umr's part shell's solid"
    held = Refusal("parts.inertia-axes", RefusalClass.USER, "body 'umr''s parts", "declare 'umr''s inertia")
    assert held.message == "body umr's parts" and held.action == "declare umr's inertia"
    assert plain_possessive("setting 'radius' needs units") == "setting 'radius' needs units"


def test_a_why_travels_in_a_problem_tuple_to_the_refusal_it_becomes():
    code, message, action = ("record.no-pick", Explained("what happened", "why it did"), "do it: x=1")
    assert isinstance(message, str) and Refusal(code, RefusalClass.USER, message, action).why == "why it did"


def test_one_reason_needs_no_count_and_several_are_counted():
    assert str(BuildRefused([refusal()])).startswith("the build is refused:\n")
    assert str(BuildRefused([refusal(), refusal()])).startswith("the build is refused for 2 reasons:\n")


def test_the_classes_are_caught_as_they_were():
    caught = {ArgumentRefused: ValueError, ArgumentTypeRefused: TypeError, RunFailed: RuntimeError,
              NotRecorded: KeyError, NotInResult: LookupError, NoOutput: AttributeError,
              NotARunFolder: FileNotFoundError}
    for cls, old in caught.items():
        with pytest.raises(old):
            raise cls([refusal()])
        assert issubclass(cls, MimeError)
    from mime.control import ControllerError
    from mime.env import EnvRefused
    from mime.saved._folder import LoadRefused

    assert issubclass(ControllerError, RuntimeError) and issubclass(LoadRefused, BuildRefused)
    assert issubclass(EnvRefused, BuildRefused) and str(EnvRefused([refusal()])).startswith("the environment")


# -- the boundary ----------------------------------------------------------------------------------------------------


@public
def _outer(error: BaseException) -> None:
    _inner(error)


@public
def _inner(error: BaseException) -> None:
    raise error


def test_a_mime_error_crosses_the_boundary_as_it_is_with_its_chain_kept(monkeypatch):
    monkeypatch.setenv(TRACEBACK_VARIABLE, "short")
    error = BuildRefused([refusal()])
    with pytest.raises(BuildRefused) as caught:
        _outer(error)
    assert caught.value is error and error.chain is not None and "_inner" in error.chain
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert all(not is_library(frame) for frame, _ in traceback.walk_tb(caught.value.__traceback__)
               if frame.f_code.co_filename != _display.__file__ and "_boundary" not in frame.f_code.co_filename)


def test_an_error_mime_raised_with_no_code_is_its_defect_and_one_from_below_is_never(monkeypatch):
    monkeypatch.setenv(TRACEBACK_VARIABLE, "short")
    from mime.units import TIME, si

    uncoded = translated(_raised(lambda: si(mm(1.0), TIME, what="a planted time")))
    assert isinstance(uncoded, InternalError) and uncoded.code == "error.uncoded"
    below = translated(_raised(lambda: np.linalg.inv(np.zeros((2, 2)))))
    assert isinstance(below, UnrecognisedError) and below.code == "error.unrecognised"
    assert below.refusals[0].category is RefusalClass.UNRECOGNISED and "LinAlgError" in below.refusals[0].message
    assert below.refusals[0].where is not None and str(HERE) in below.refusals[0].where


def _raised(action: Any) -> BaseException:
    try:
        action()
    except Exception as error:  # noqa: BLE001 - the test's own catch of what it planted
        return error
    raise AssertionError("the action did not raise")


def test_a_translated_error_keeps_its_original_and_full_mode_chains_it(monkeypatch):
    monkeypatch.setenv(TRACEBACK_VARIABLE, "full")
    planted = ZeroDivisionError("planted")
    with pytest.raises(UnrecognisedError) as caught:
        _outer(planted)
    assert caught.value.original is planted and caught.value.__cause__ is planted
    monkeypatch.setenv(TRACEBACK_VARIABLE, "short")
    with pytest.raises(UnrecognisedError) as short:
        _outer(ZeroDivisionError("planted"))
    assert short.value.__cause__ is None and isinstance(short.value.original, ZeroDivisionError)
    assert "ZeroDivisionError: planted" in (short.value.chain or "")


def test_an_interrupt_passes_the_boundary_untouched():
    with pytest.raises(KeyboardInterrupt):
        _outer(KeyboardInterrupt())


def test_a_defect_of_the_translation_never_hides_the_error(monkeypatch):
    from mime.errors import _boundary

    def broken(error: Exception, frame: Any = None) -> MimeError:
        raise RuntimeError("the translation's own defect")

    monkeypatch.setattr(_boundary, "crossed", broken)
    with pytest.raises(ZeroDivisionError):
        _outer(ZeroDivisionError("planted"))


def test_the_script_traceback_keeps_the_scripts_frames_and_the_control_finds_the_librarys():
    error = _raised(lambda: np.linalg.inv(np.zeros((2, 2))))
    frames = [frame for frame, _ in traceback.walk_tb(error.__traceback__)]
    assert any(is_library(frame) for frame in frames), "the control: numpy's frames are the library's"
    kept = [frame for frame, _ in traceback.walk_tb(script_traceback(error.__traceback__))]
    assert kept and all(frame.f_code.co_filename == str(HERE) for frame in kept)


# -- the display -----------------------------------------------------------------------------------------------------


def test_the_full_chain_is_written_to_its_file_and_the_newest_are_kept(tmp_path, monkeypatch):
    monkeypatch.setenv(DIRECTORY_VARIABLE, str(tmp_path))
    paths = []
    for _ in range(KEPT + 3):
        error = crossed(_raised(lambda: _inner(BuildRefused([refusal()]))))
        paths.append(write_traceback(error))
    held = sorted(tmp_path.glob("*.txt"))
    assert len(held) == KEPT and paths[-1] in held and paths[0] not in held
    text = held[-1].read_text(encoding="utf-8")
    assert "frame.gravity-missing" in text and "environment:" in text and "Traceback" in text and "_inner" in text


def test_a_directory_that_cannot_be_written_is_named_and_does_not_stop_the_display(tmp_path, monkeypatch):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    monkeypatch.setenv(DIRECTORY_VARIABLE, str(blocked))
    monkeypatch.setenv(TRACEBACK_VARIABLE, "short")
    text = short_text(BuildRefused([refusal()]), None)
    assert "was not written" in text and "frame.gravity-missing" in text


def test_the_hook_shows_a_mime_error_short_and_hands_any_other_to_the_hook_before_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(DIRECTORY_VARIABLE, str(tmp_path))
    monkeypatch.setenv(TRACEBACK_VARIABLE, "short")
    seen: list[BaseException] = []
    hook = _display._hook(lambda kind, value, tb: seen.append(value))
    error = _raised(lambda: _outer(BuildRefused([refusal()])))
    hook(type(error), error, error.__traceback__)
    shown = capsys.readouterr().err
    assert "BuildRefused: the build is refused:" in shown and f'File "{HERE}"' in shown and not seen
    assert "_boundary.py" not in shown, "the boundary's own frame is left out"
    other = ValueError("not MIME's")
    hook(ValueError, other, None)
    assert seen == [other]
    monkeypatch.setenv(TRACEBACK_VARIABLE, "full")
    hook(type(error), error, error.__traceback__)
    assert seen[-1] is error, "in full mode the hook before it shows a MIME error"


def test_installing_twice_wraps_the_hook_once(monkeypatch):
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    _display.install()
    first = sys.excepthook
    _display.install()
    assert sys.excepthook is first and getattr(first, "_mime_previous") is sys.__excepthook__


class FakeShell:
    """What MIME uses of an IPython shell: its custom exceptions, ``set_custom_exc`` and the traceback formatter."""

    def __init__(self, custom: tuple[type[BaseException], ...] = ()) -> None:
        self.custom_exceptions = custom
        self.handler: Any = None
        self.shown: list[Any] = []

    def set_custom_exc(self, kinds: tuple[type[BaseException], ...], handler: Any) -> None:
        self.custom_exceptions, self.handler = kinds, handler

    def showtraceback(self, exc_tuple: Any, tb_offset: Any = None) -> None:
        self.shown.append("IPython's own")

    def _showtraceback(self, kind: Any, value: Any, lines: list[str]) -> None:
        self.shown.append(lines)


def test_in_ipython_a_mime_error_is_shown_short_through_the_shells_handler(tmp_path, monkeypatch):
    monkeypatch.setenv(DIRECTORY_VARIABLE, str(tmp_path))
    shell = FakeShell()
    monkeypatch.setitem(sys.modules, "IPython", SimpleNamespace(get_ipython=lambda: shell))
    _display._install_in_ipython()
    assert shell.custom_exceptions == (MimeError,)
    error = _raised(lambda: _outer(BuildRefused([refusal()])))
    monkeypatch.setenv(TRACEBACK_VARIABLE, "short")
    lines = shell.handler(shell, type(error), error, error.__traceback__)
    assert any("BuildRefused: the build is refused:" in line for line in lines) and shell.shown == [lines]
    monkeypatch.setenv(TRACEBACK_VARIABLE, "full")
    assert shell.handler(shell, type(error), error, error.__traceback__) == []
    assert shell.shown[-1] == "IPython's own", "in full mode the shell's own traceback is shown"
    taken = FakeShell((ValueError,))
    monkeypatch.setitem(sys.modules, "IPython", SimpleNamespace(get_ipython=lambda: taken))
    _display._install_in_ipython()
    assert taken.custom_exceptions == (ValueError,) and taken.handler is None, "a shell's own handler is kept"


# -- the CLI ---------------------------------------------------------------------------------------------------------


def test_the_command_exits_two_on_a_refusal_three_on_a_defect_and_takes_its_flag_anywhere(tmp_path, monkeypatch,
                                                                                           capsys):
    from mime import cli

    monkeypatch.setenv(DIRECTORY_VARIABLE, str(tmp_path / "tracebacks"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(TRACEBACK_VARIABLE)
    assert cli.main(["run", "missing", "--duration", "1", "--out", "out"]) == cli.REFUSED
    shown = capsys.readouterr().err
    assert "mime run: BuildRefused" in shown and "load.missing-folder" in shown and 'File "' not in shown
    assert cli.main(["run", "missing", "--traceback", "--duration", "1", "--out", "out"]) == cli.REFUSED
    assert 'File "' in capsys.readouterr().err and TRACEBACK_VARIABLE not in os.environ, "the flag is undone"
    assert cli.refused(ZeroDivisionError("below"), "run") == cli.FAILED
    assert "error.unrecognised" in capsys.readouterr().err


def test_a_saved_folder_misspelt_is_named_with_the_saved_ones_beside_it(tmp_path, monkeypatch):
    from mime.env._check import missing_folder

    (tmp_path / "experiment").mkdir()
    (tmp_path / "experiment" / "mime.lock").write_text("{}")
    refused = missing_folder(tmp_path / "experimnt")
    assert refused.code == "load.missing-folder" and "Did you mean" in refused.refusals[0].action


# -- a run's and a result's errors ------------------------------------------------------------------------------------


def test_a_runs_arguments_are_refused_with_their_codes(built):
    held, drift = built
    cases = {"run.duration": lambda: held.run(ms(2.5)), "run.every": lambda: held.run(ms(2), every=ms(1)),
             "run.record": lambda: held.run(ms(2), record=["swim_speed:rod.sped"]),
             "run.chunks": lambda: held.run(ms(2), hook=lambda chunk: None)}
    for code, action in cases.items():
        error = _raised(action)
        assert isinstance(error, ValueError) and isinstance(error, MimeError) and error.codes() == [code], code
    error = _raised(lambda: held.run(ms(2), every=ms(1.5), record=[drift.speed]))
    assert error.codes() == ["run.every"] and "every=s(" in error.refusals[0].action
    typed = _raised(lambda: held.run(ms(2), record=[3]))
    assert isinstance(typed, TypeError) and typed.codes() == ["run.record"]


def test_a_chunked_run_that_is_not_whole_chunks_and_a_result_file_of_another_kind_are_refused(built, tmp_path):
    held, _ = built
    chunks = _raised(lambda: held.run(ms(3), hook=lambda chunk: None, chunk=ms(2)))
    assert isinstance(chunks, ValueError) and chunks.codes() == ["run.chunks"] and "s(...)" in chunks.refusals[0].action
    np.savez(tmp_path / "other.npz", x=np.zeros(1))
    assert _raised(lambda: RunResult.read(tmp_path / "other.npz")).codes() == ["result.not-a-run-folder"]
    other = _raised(lambda: RunResult.from_manifest({"format": "other", "version": 0}, None))  # type: ignore[arg-type]
    assert isinstance(other, ValueError) and other.codes() == ["result.format"]


def test_a_setting_the_saved_form_cannot_hold_is_refused_as_unsavable_and_is_a_value_error():
    from mime.saved import UnsavableError

    error = UnsavableError("the setting is a function")
    assert isinstance(error, ValueError) and error.codes() == ["save.unsavable"] and error.reason in str(error)


def test_a_call_given_the_wrong_type_and_a_verify_with_nothing_saved_are_refused_with_their_codes(built):
    exp, _ = rod()
    wrong = _raised(lambda: exp.add_body("rod"))
    assert isinstance(wrong, TypeError) and wrong.codes() == ["call.wrong-type"]
    nothing = _raised(lambda: built[0].verify())
    assert isinstance(nothing, ValueError) and nothing.codes() == ["verify.nothing-saved"]


def test_a_results_lookups_are_refused_with_their_codes(built, tmp_path):
    held, drift = built
    result = held.run(ms(2), record=[drift.speed])
    assert _raised(lambda: result["swim_speed:bod"]).codes() == ["result.no-observation"]
    assert _raised(lambda: result[drift]["sped"]).codes() == ["result.no-output"]
    missing = _raised(lambda: result[drift].sped)
    assert isinstance(missing, AttributeError) and missing.codes() == ["result.no-output"]
    assert _raised(lambda: result[drift].speed.between(ms(2), ms(1))).codes() == ["result.window"]
    assert _raised(lambda: result[drift].speed.between(ms(0), ms(9))).codes() == ["result.not-recorded"]
    assert _raised(lambda: held.run(ms(1))[drift].speed).codes() == ["result.not-recorded"]
    assert _raised(lambda: RunResult.read(tmp_path)).codes() == ["result.not-a-run-folder"]
    from mime.results._mat import names_of
    from mime.results._series import series_table

    speed = result[drift].speed
    assert _raised(lambda: series_table([speed, speed.between(ms(1), ms(2))])).codes() == ["result.table"]

    assert _raised(lambda: names_of("output", ["x" * 70])).codes() == ["result.matlab-name"]


def test_a_state_that_is_not_finite_names_its_first_step_thing_and_field_where_the_run_kept_them():
    exp, _ = rod(dt=100.0, spin=(30.0, 20.0, 10.0))
    kept = _raised(lambda: exp.build().run(s(5.0), history=True))
    assert isinstance(kept, RunFailed) and kept.codes() == ["run.non-finite"]
    said = kept.refusals[0].message
    assert "at step" in said and "rod's" in said and "among the fields" not in said and kept.result is not None
    first = int(said.split("at step ")[1].split(" ")[0])
    history = kept.result.history  # type: ignore[union-attr]
    bad = {(node, name): int(np.argmax(~np.isfinite(np.asarray(value).reshape(len(value), -1)).all(axis=1)))
           for node, fields in history.items() for name, value in fields.items()
           if np.issubdtype(np.asarray(value).dtype, np.inexact)
           and not np.isfinite(np.asarray(value)).all()}
    assert min(bad.values()) == first - 1, "the step named is the first at which any field is not finite"
    assert all(f"{owner}'s {name}" in said for (owner, name), at in bad.items() if at == first - 1 and owner == "rod")
    unkept = _raised(lambda: exp.build().run(s(5.0)))
    assert "the run kept no step's state" in unkept.refusals[0].message
    assert "history=True" in unkept.refusals[0].action


def test_the_control_a_finite_run_is_not_refused(built):
    assert built[0].run(ms(3)).steps == 3


def test_an_input_nobody_feeds_is_mimes_defect(monkeypatch):
    from mime.build import _maddening

    monkeypatch.setattr(_maddening, "external_inputs", lambda graph: [("field", "frequency")])
    error = _raised(lambda: _maddening.fed_inputs(object(), {}))  # type: ignore[arg-type]
    assert isinstance(error, InternalError) and error.codes() == ["run.unfed-input"]


# -- errors from below -----------------------------------------------------------------------------------------------


def test_running_out_of_memory_says_what_to_reduce_and_the_control_another_error_does_not():
    for error in (MemoryError(), RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 8.00GiB")):
        found = out_of_memory(error)
        assert found is not None and found.codes() == ["run.out-of-memory"]
        assert found.refusals[0].category is RefusalClass.ENVIRONMENT and "record=" in found.refusals[0].action
    assert out_of_memory(RuntimeError("INVALID_ARGUMENT: shapes differ")) is None


def test_a_traced_values_error_is_recognised_and_jaxs_note_is_left_out():
    import jax
    import jax.numpy as jnp

    def branch(x: Any) -> Any:
        return x if x > 0 else -x

    error = _raised(lambda: jax.jit(branch)(jnp.ones(())))
    assert traced_value_error(error) and not traced_value_error(ValueError("x"))
    assert "JAX has removed" not in first_lines(error) and first_lines(error).startswith("TracerBoolConversionError")


# -- the walk's messages ---------------------------------------------------------------------------------------------


def test_an_unchecked_condition_points_to_require_and_an_unstated_one_to_an_acceptance():
    """The first-hour walk's 3b: test_method is a condition the record states and MIME's request gave nothing to
    check it against, so its route is the record's require= (#188), never an acceptance; shear_rate the record does
    not state, which only an acceptance covers."""
    from mime.interactions import Drag
    from mime.materials import Record, packaged_record
    from mime.scene import Sphere
    from mime.things import Medium, Part
    from mime.units import degC, kg_per_m3, um

    exp = Experiment("settle", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.05),
                     conditions={"temperature": degC(20.0, plus_minus=0.5)})
    water = exp.add_medium(Medium("water", material=Record("water", packaged_record("openmaterica_example_water"))))
    ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=um((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"sphere": Part(Sphere(radius=um(50.0)))}))
    exp.declare("ball.parts.sphere.density", kg_per_m3(7850.0), reason="steel")
    exp.attach(Drag.BEM(spacing=um(20.0), blob=0.5, wall_kind="free_space"), body=ball, medium=water)
    from mime.interactions import Gravity

    exp.attach(Gravity(), body=ball, medium=water)
    refused = _raised(exp.build)
    viscosity = next(r for r in refused.refusals if "dynamic_viscosity" in r.message)
    density = next(r for r in refused.refusals if "bulk.density" in r.message)
    assert "require={'test_method': ('reference_formulation',)}" in density.action
    assert "Acceptance" not in density.action, "an unchecked condition is stated, never accepted"
    assert "Acceptance(covers=('condition_unstated:shear_rate',)" in viscosity.action
    assert "condition_unchecked:test_method" not in viscosity.action
    assert viscosity.why is not None and "OpenMaterica's verdict: " in viscosity.message


def test_a_record_with_no_conditions_names_experiment_conditions():
    from mime.materials import Record, packaged_record
    from mime.things import Medium

    exp, _ = rod()
    exp.add_medium(Medium("water", material=Record("water", packaged_record("openmaterica_example_water"))))
    exp.declare("rod.buoyant_volume", m3(1e-9), reason="a millimetre rod")
    from mime.interactions import Gravity

    exp.attach(Gravity(), body="rod", medium="water")
    refused = _raised(exp.build)
    picks = [r for r in refused.refusals if r.code == "record.no-pick"]
    assert picks and all("Experiment(..., conditions={'temperature': degC(20, plus_minus=0.5)})" in r.action
                         for r in picks)


def test_a_value_declared_where_a_thing_has_none_names_the_declaration_with_an_article():
    exp, _ = rod()
    from mime.fields import UniformRotating
    from mime.scene import X
    from mime.things import Apparatus
    from mime.units import Hz, mT

    exp.add_apparatus(Apparatus("field", fields={"magnetic": UniformRotating(rotation_axis=X,
                                                                             initial_direction=(0.0, 1.0, 0.0))},
                                commands={"frequency": Hz(5.0), "amplitude": mT(1.2)}))
    exp.declare("field.frequency", Hz(5.0), reason="the drive's")
    refused = _raised(exp.build)
    held = next(r for r in refused.refusals if r.code == "value.unknown")
    assert "an apparatus has no value 'frequency'" in held.message and "exp.declare(" in held.action


def test_a_script_shows_a_short_error_in_a_fresh_process_and_the_control_shows_every_frame(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("from mime import Experiment\nfrom mime.scene import Scene, Z\nfrom mime.units import ms\n"
                      "Experiment('x', scene=Scene.new(up=Z, units='m'), dt=ms(1)).build()\n", encoding="utf-8")
    environment = {**os.environ, "JAX_PLATFORMS": "cpu", DIRECTORY_VARIABLE: str(tmp_path / "tracebacks")}
    short = subprocess.run([sys.executable, str(script)], env={**environment, TRACEBACK_VARIABLE: "short"},
                           capture_output=True, text=True, check=False)
    full = subprocess.run([sys.executable, str(script)], env={**environment, TRACEBACK_VARIABLE: "full"},
                          capture_output=True, text=True, check=False)
    assert "frame.gravity-missing" in short.stderr and f'File "{script}", line 4' in short.stderr
    assert "src/mime" not in short.stderr.replace(str(tmp_path), "") and "_plan.py" in full.stderr
