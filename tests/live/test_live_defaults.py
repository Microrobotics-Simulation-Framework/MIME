"""A live input's declared default changes nothing: a session that sets no live input is the batch run, bit for bit.
A default is a declaration, and registration can check its shape and that it is finite but not what the step does with
it, so this walk runs every registered controller with live inputs, MIME's own and the test rigs', and compares. A
registered controller with live inputs (MIME's, or a namespace's, the lab fixture's included) and no rig here fails the
walk by name. The control: a controller whose default (0.5) moves the angle is refused, naming the controller and the
live input."""

import warnings

import numpy as np
import pytest
from live_cases import jit_steered, offset_default, steered, tracked

from mime.registry import EntryKind, registered
from mime.units import ms

#: The rig each controller with live inputs is run in, by its registered name.
RIGS = {"arm_tracking": tracked, "oracle.steered": steered, "oracle.jit_steered": jit_steered}


def default_problems(name: str, make) -> list[str]:
    """What shows that the declared defaults of controller ``name`` change the run: a session of ``make``'s rig that
    sets nothing, against the same rig in batch, over three chunks."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = make().build().live(pace=None)
        frames = session.step(3)
        steps = frames[-1].step
        history = make().build().run(ms(steps), history=True).history
    off = [f"{path} at step {held.step}" for held in frames[1:] for path, value in held.values.items()
           if np.asarray(value).tobytes() != np.asarray(
               history[path.rpartition(".")[0]][path.rpartition(".")[2]][held.step - 1]).tobytes()]
    if not off:
        return []
    live = next(r for r in registered(EntryKind.CONTROLLER) if r.name == name).cls.live
    declared = ", ".join(f"{label!r} (default {value.default!r})" for label, value in live.items())
    return [f"{name}: a session that sets nothing differs from the batch run, {len(off)} fields (first: {off[0]}); "
            f"a live input's default must change nothing, and this controller declares {declared}"]


def unrigged(rigs) -> list[str]:
    return sorted(r.name for r in registered(EntryKind.CONTROLLER)
                  if r.cls.live and not r.name.startswith("oracle.") and r.name not in rigs)


def test_every_controllers_declared_defaults_reproduce_the_batch_run():
    assert unrigged(RIGS) == []
    # The control of the coverage: with no rigs, the controller with live inputs is found.
    assert "arm_tracking" in unrigged(())
    assert [problem for name, make in RIGS.items() for problem in default_problems(name, make)] == []


def test_a_default_that_changes_the_run_is_refused_by_name():
    problems = default_problems("oracle.offset_default", offset_default)
    assert len(problems) == 1
    assert "oracle.offset_default" in problems[0] and "'offset' (default 0.5)" in problems[0]
