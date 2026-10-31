"""The swap test (dev1 phase 7c2, design #75 and #174): A0's controller on its dynamic rungs placed inside the compiled
step (mode 1) and called back between compiled chunks (mode 2), the same ``law(ctx)`` in both, compared field by field.

Its tolerance was stated before it ran, in ``a0_dynamic_expected_differences.json`` (``swap_test``), from what mode 2
changes: not the schedule (7b2 measured the same writes acting at the same steps, and mode 1 adds no step of lag, the
order that would add one being refused), nor the hold (the command is held per control period in both), but where
the step is compiled, so that XLA may round its arithmetic otherwise: 8 single-precision units in the last place per
step run, of each field's largest magnitude. Its controls plant the one thing the tolerance says it carries none of,
a step of lag: the latency removed (1 to 0 steps) or doubled (1 to 2) in mode 1 against mode 2 at one step."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest
from script_a0_dynamic import a0_dynamic

from mime.units import Hz, ms

DECLARED = json.loads((Path(__file__).parent / "a0_dynamic_expected_differences.json").read_text())["swap_test"]
STEPS = 1000
ULPS_PER_STEP = 8
STEP = 0.5


@pytest.fixture(autouse=True, scope="module")
def x64():
    """This module runs in float32, as A0's comparison does."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", held)


_RUNS: dict[tuple[str, int], dict] = {}


def history(mode: str, latency: int) -> dict:
    """A0 on its dynamic rungs at 3 Hz, its controller every step ``latency`` steps late, run ``STEPS`` steps: each
    placement and latency run once in the module, the cases sharing it (a run of the callback at every step is most
    of a case's time)."""
    if (mode, latency) not in _RUNS:
        built = a0_dynamic(frequency=Hz(3.0), mode=mode, latency=ms(STEP * latency) if latency else None).build()
        found = built.run(ms(STEP * STEPS), history=True).history
        assert found is not None
        _RUNS[(mode, latency)] = found
    return _RUNS[(mode, latency)]


def moves(inside: dict, called: dict) -> dict[str, tuple[float, float]]:
    """For every field of every node the callback's graph holds, what it moved between the modes and the declared
    bound, ``steps x 8 x eps x max |field|``."""
    eps = float(np.finfo(np.float32).eps)
    found = {}
    for node, fields in called.items():
        for name, oracle in fields.items():
            held = np.asarray(oracle, dtype=np.float64)
            moved = float(np.max(np.abs(np.asarray(inside[node][name], dtype=np.float64) - held)))
            found[f"{node}.{name}"] = (moved, STEPS * ULPS_PER_STEP * eps * float(np.max(np.abs(held))))
    return found


def failures(inside: dict, called: dict) -> list[str]:
    """Every field that moved by more than its bound, with what it moved and the bound."""
    found = moves(inside, called)
    for field, (moved, bound) in found.items():
        print(f"{field}: {moved:.3g} (bound {bound:.3g})")
    return [f"{field} moved {moved:.3g}, over {bound:.3g}" for field, (moved, bound) in found.items()
            if not moved <= bound]


def test_the_declared_tolerance_is_the_one_this_test_holds():
    """The bound and the configurations are the declaration's, stated before the comparison ran."""
    assert "8 single-precision units in the last place per step run" in DECLARED["bound"]
    assert [c["latency_steps"] for c in DECLARED["configurations"]] == [0, 1]
    assert "1000 steps" in DECLARED["run"]


@pytest.mark.parametrize("latency", [0, 1], ids=["no-latency", "one-step"])
def test_a0s_controller_runs_alike_inside_the_compiled_step_and_as_a_callback(latency):
    """The same law in both modes, at A0's rate (every step) with no latency (A0's own) and one step late: every field
    of every node agrees within the declared bound, the controller's writes (the arm's torques and the rotor's) among
    them, through the rotor's spin-up and the arm's kick."""
    assert failures(history("graph", latency), history("callback", latency)) == []


@pytest.mark.parametrize("planted", [0, 2], ids=["latency-removed", "latency-doubled"])
def test_a_step_of_lag_removed_or_added_inside_the_compiled_step_fails_the_swap(planted):
    """The control: mode 1 at a latency of ``planted`` steps against mode 2 at one step. A step of lag is the least
    the schedule can be off by, and the tolerance carries none: it fails."""
    assert failures(history("graph", planted), history("callback", 1)) != []
