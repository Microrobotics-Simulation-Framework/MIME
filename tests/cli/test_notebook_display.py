"""MIME's live display for a run started from Python in a notebook (``built.run``, ``run_to`` and ``built.save``,
``progress=``). On Colab a ``!mime run`` cell stacks a line at every redraw, since its output ignores cursor-up, while
rich drawn through the notebook's own output draws two-line live displays whole; so a run started in the notebook's
Python shows ``mime run``'s display that way.

The notebook here is simulated where rich meets it, and rich's own code runs between: IPython's ``get_ipython`` in
the builtins, returning a shell of the class rich reads (``ZMQInteractiveShell``, a Jupyter kernel's; Colab's by its
module), and stand-ins for ``IPython.display.display`` and ``ipywidgets.Output``, which record each frame rich draws
into the output widget. The display is eager in a notebook (drawn again at each phase and each chunk), so the frames
are the run's chunks, whatever the clock.

Each with its control. The detection: a terminal's IPython shell and plain Python make no display, and progress=False
none in a notebook, while the same counter counts a Jupyter kernel's and Colab's. Bit for bit: a run and a ``run_to``
with the display and without it give the same state and ``outputs.npz`` in every byte and the same chunks, as the
caller's own ``stop`` saw them; a run's state cannot tell chunkings apart (a run in chunks is the one scan bit for bit,
``tests/env/test_run_interrupt.py``), so the chunks are compared themselves, and a display planted to change them is
caught. The caller's ``stop`` is asked and its stop kept."""

import builtins
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from mime.cli import _display
from mime.cli._display import NOTEBOOK_STOP, SAVING, SAVING_RUN, RunDisplay
from mime.control import _run as control_run
from mime.control._run import MIN_CHUNK, automatic_chunks
from mime.experiment import BuiltExperiment
from mime.testing.toys import full_toy
from mime.units import s

#: The run's steps: chunked by the automatic rule into 32 chunks of two.
STEPS = 64


class ZMQInteractiveShell:
    """A Jupyter kernel's shell, by the name rich's detection reads."""


class TerminalInteractiveShell:
    """IPython's shell in a terminal, which rich does not take for a notebook."""


ColabShell = type("Shell", (), {"__module__": "google.colab._shell"})


class Output:
    """``ipywidgets.Output`` as rich's live display uses it: shown once, entered for each frame, closed at the end."""

    def __init__(self, notebook: "Notebook") -> None:
        self.notebook = notebook
        self.frames: list[str] = []
        self.closed = False

    def __enter__(self) -> "Output":
        self.notebook.inside = self
        return self

    def __exit__(self, *_: object) -> None:
        self.notebook.inside = None

    def clear_output(self, wait: bool = False) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class Notebook:
    """What the simulated notebook was asked to show: the widgets, and the frames drawn into them."""

    def __init__(self) -> None:
        self.widgets: list[Output] = []
        self.inside: Output | None = None
        self.outside: list[str] = []

    def display(self, shown: object) -> None:
        if isinstance(shown, Output):
            self.widgets.append(shown)
        elif self.inside is not None:
            self.inside.frames.append(str(getattr(shown, "text", shown)))
        else:
            self.outside.append(str(getattr(shown, "text", shown)))

    @property
    def frames(self) -> list[str]:
        return [frame for widget in self.widgets for frame in widget.frames]


def in_ipython(monkeypatch, shell: type) -> None:
    monkeypatch.setattr(builtins, "get_ipython", lambda: shell(), raising=False)


@pytest.fixture
def notebook(monkeypatch) -> Notebook:
    held = Notebook()
    in_ipython(monkeypatch, ZMQInteractiveShell)
    ipython, shown, widgets = (types.ModuleType(name) for name in ("IPython", "IPython.display", "ipywidgets"))
    shown.display = held.display  # pyright: ignore[reportAttributeAccessIssue]
    ipython.display = shown  # pyright: ignore[reportAttributeAccessIssue]
    widgets.Output = lambda: Output(held)  # pyright: ignore[reportAttributeAccessIssue]
    for module in (ipython, shown, widgets):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return held


def seconds(built: BuiltExperiment, steps: int = STEPS):
    return s(steps * built.plans.dt)


def steps_shown(frames: list[str]) -> list[int]:
    """The step each frame that shows the bar names (``step 12 of 64``), in the order drawn."""
    import re

    return [int(found.group(1).replace(",", "")) for frame in frames
            if (found := re.search(r"step ([\d,]+) of ", frame)) is not None]


class Chunks:
    """A caller's own ``stop``: it notes each chunk's index and end, and stops at the ``at``-th when given."""

    def __init__(self, at: int | None = None) -> None:
        self.seen: list[tuple[int, int]] = []
        self.at = at

    def __call__(self, progress) -> str | None:
        self.seen.append((progress.index, progress.steps))
        return "enough" if self.at is not None and len(self.seen) == self.at else None


def digest(state) -> dict[str, bytes]:
    return {f"{node}.{name}": np.asarray(value).tobytes() for node, fields in state.items()
            for name, value in fields.items()}


def npz_digest(path: Path) -> dict[str, bytes]:
    with np.load(path, allow_pickle=False) as held:
        return {name: held[name].dtype.str.encode() + held[name].tobytes() for name in held.files}


def differences(plain: tuple[dict[str, bytes], list[tuple[int, int]]],
                shown: tuple[dict[str, bytes], list[tuple[int, int]]]) -> list[str]:
    """What tells a run with the display from the same run without it: a field not equal in every byte, or another
    chunking (each chunk's index and end, as the caller's ``stop`` saw them)."""
    found = [f"{name} differs" for name in sorted(set(plain[0]) | set(shown[0]))
             if plain[0].get(name) != shown[0].get(name)]
    if plain[1] != shown[1]:
        found.append(f"the chunks differ: {plain[1][:3]}... against {shown[1][:3]}...")
    return found


# -- in a notebook ----------------------------------------------------------------------------------------------------


def test_in_a_notebook_run_to_shows_the_display_to_the_steps_asked_then_its_saving_phase_before_the_save_and_closes(
        notebook, monkeypatch, tmp_path):
    seen_at_save: list[list[str]] = []
    original = BuiltExperiment._save  # pyright: ignore[reportPrivateUsage]

    def save(self, folder, steps):
        seen_at_save.append(list(notebook.frames))
        return original(self, folder, steps)

    monkeypatch.setattr(BuiltExperiment, "_save", save)
    built = full_toy().build()
    chunk = automatic_chunks(STEPS)[0]
    written = built.run_to(tmp_path / "run", seconds(built))

    assert written.steps == STEPS and written.stopped is None
    assert len(notebook.widgets) == 1 and notebook.widgets[0].closed, "one display, closed at the end"
    # rich's live display ends with an empty draw outside its widget (its buffer, flushed as it stops), whatever it showed
    assert all(not frame.strip() for frame in notebook.outside), "every frame was drawn into the display's widget"
    frames = notebook.frames
    assert "compiling the first chunk" in frames[0] and NOTEBOOK_STOP in frames[0]
    assert steps_shown(frames) == sorted(steps_shown(frames)), "the bar only advances"
    assert set(steps_shown(frames)) == set(range(chunk, STEPS + 1, chunk)), "a frame at each chunk's end"
    assert len(seen_at_save) == 1, "run_to saves the run's own saved form once"
    before = seen_at_save[0]
    assert steps_shown(before)[-1] == STEPS, "the bar reached the steps asked before the save"
    assert SAVING_RUN in before[-1], "the saving phase is on the screen when the save begins"
    assert not any(SAVING_RUN in frame for frame in before[:-1] if steps_shown([frame])), \
        "the saving phase follows the bar"


def test_in_a_notebook_run_shows_the_display_to_the_steps_asked_and_closes_and_progress_false_shows_none(notebook):
    built = full_toy().build()
    result = built.run(seconds(built))
    assert result.steps == STEPS and len(notebook.widgets) == 1 and notebook.widgets[0].closed
    frames = notebook.frames
    assert steps_shown(frames)[-1] == STEPS and not any(SAVING_RUN in frame for frame in frames)
    assert not any("writing the run folder" in frame for frame in frames), "run writes no folder"

    shown = len(notebook.widgets)
    again = full_toy().build().run(seconds(built), progress=False)
    assert again.steps == STEPS and len(notebook.widgets) == shown, "progress=False shows nothing in a notebook"


def test_in_a_notebook_save_shows_its_phase_and_closes(notebook, tmp_path):
    built = full_toy().build()
    built.save(tmp_path / "saved")
    assert len(notebook.widgets) == 1 and notebook.widgets[0].closed
    assert notebook.frames and all(SAVING in frame for frame in notebook.frames)
    built.save(tmp_path / "again", progress=False)
    assert len(notebook.widgets) == 1


def test_a_stop_of_the_callers_own_is_still_asked_beside_the_display_and_its_stop_is_kept(notebook):
    built = full_toy().build()
    chunk = automatic_chunks(STEPS)[0]
    asked = Chunks(at=5)
    result = built.run(seconds(built), stop=asked)
    assert result.stopped is not None and result.stopped.reason == "enough"
    assert result.steps == 5 * chunk and [index for index, _ in asked.seen] == list(range(5))
    assert steps_shown(notebook.frames)[-1] == 5 * chunk and notebook.widgets[0].closed
    # the control: the same stop that never stops lets the run end where it was asked to
    goes_on = Chunks()
    assert full_toy().build().run(seconds(built), stop=goes_on).steps == STEPS
    assert len(goes_on.seen) == STEPS // chunk - 1, "asked after every chunk but the last"


# -- where it is not shown --------------------------------------------------------------------------------------------


def counted(monkeypatch) -> list[RunDisplay]:
    made: list[RunDisplay] = []
    original = RunDisplay.__init__

    def init(self, *args, **kwargs) -> None:
        made.append(self)
        original(self, *args, **kwargs)

    monkeypatch.setattr(RunDisplay, "__init__", init)
    return made


def test_outside_a_notebook_the_default_asks_nothing_and_shows_nothing(monkeypatch, capsys):
    assert not hasattr(builtins, "get_ipython"), "the test process is not IPython"

    def asked(progress) -> None:
        raise AssertionError("plain Python asked for a display")

    monkeypatch.setattr(_display, "python_display", asked)
    built = full_toy().build()
    assert built.run(seconds(built)).steps == STEPS
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


@pytest.mark.parametrize("shell, shown", [(TerminalInteractiveShell, 0), (ZMQInteractiveShell, 1), (ColabShell, 1)],
                         ids=["ipython-in-a-terminal", "control-jupyter", "control-colab"])
def test_a_display_is_made_in_a_notebook_alone(monkeypatch, shell, shown):
    in_ipython(monkeypatch, shell)
    made = counted(monkeypatch)
    assert len([display for display in (_display.python_display(None),) if display is not None]) == shown
    assert len(made) == shown
    assert _display.python_display(False) is None and len(made) == shown, "progress=False makes none"


def test_progress_true_outside_a_notebook_follows_mime_runs_rules(monkeypatch):
    made = counted(monkeypatch)
    assert not sys.stderr.isatty() and _display.python_display(True) is None, "not a terminal: none, as mime run"
    monkeypatch.setattr(_display, "styled", lambda stream: stream is sys.stderr)
    display = _display.python_display(True)
    assert display is not None and len(made) == 1, "the control: a styled standard error is drawn on"
    display.close()


# -- bit for bit ------------------------------------------------------------------------------------------------------


def test_a_run_and_a_run_to_with_the_display_are_the_run_without_it_bit_for_bit_and_in_the_same_chunks(
        notebook, monkeypatch, tmp_path):
    built = full_toy().build()
    duration = seconds(built)
    plain_chunks, shown_chunks = Chunks(), Chunks()
    plain = full_toy().build().run(duration, stop=plain_chunks, progress=False)
    shown = full_toy().build().run(duration, stop=shown_chunks)
    assert notebook.widgets, "the display was on in the arm that has it"
    assert differences((digest(plain.final), plain_chunks.seen), (digest(shown.final), shown_chunks.seen)) == []
    assert len(plain_chunks.seen) == STEPS // automatic_chunks(STEPS)[0] - 1

    plain_to, shown_to = Chunks(), Chunks()
    plain_written = full_toy().build().run_to(tmp_path / "plain", duration, history=True, stop=plain_to,
                                              progress=False)
    widgets = len(notebook.widgets)
    shown_written = full_toy().build().run_to(tmp_path / "shown", duration, history=True, stop=shown_to)
    assert len(notebook.widgets) == widgets + 1
    assert differences((npz_digest(plain_written.outputs), plain_to.seen),
                       (npz_digest(shown_written.outputs), shown_to.seen)) == []

    # the control: a display that changed the chunking is caught, by the chunks if not by the state
    monkeypatch.setattr(control_run, "automatic_chunks", lambda steps: (2 * MIN_CHUNK,) * (steps // (2 * MIN_CHUNK)))
    planted = Chunks()
    replanted = full_toy().build().run(duration, stop=planted)
    found = differences((digest(plain.final), plain_chunks.seen), (digest(replanted.final), planted.seen))
    assert any("the chunks differ" in line for line in found), found

