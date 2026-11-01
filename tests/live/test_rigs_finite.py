"""Every rig the live tests and a client's development run stays finite for ten seconds of simulated time, in every
field: twice the longest a rig was seen to last before it went non-finite (4.4 s, ``tracked``: its body, with nothing
to damp it, was driven by a rotating field and its gyroscopic term, taken explicitly, gained energy at every step;
``MIME-ANO-129``). A run that stops being finite refuses with ``run.non-finite``, which fails the test, and the final
state is checked here too.

The control: the same rig with the gyroscopic term taken explicitly again (the step's matrix planted) is refused, at
the same duration, every run: the scheme has no randomness.
"""

import warnings

import numpy as np
import pytest
from live_cases import in_graph, jit_steered, offset_default, steered, tracked
from script_a0 import a0

from mime.build import RunFailed
from mime.nodes.body import RigidInertialMidpointNode
from mime.units import ms, s

#: Each rig and the duration it runs: ten seconds, or the whole number of its controller's periods nearest below it
#: (a run with a controller called between chunks is whole periods: 3 ms for the steered rigs).
RIGS = {
    "tracked": (tracked, s(10.0)), "steered": (steered, s(9.999)), "jit_steered": (jit_steered, s(9.999)),
    "offset_default": (offset_default, s(9.999)), "steered_in_graph": (in_graph, s(10.0)), "a0": (a0, s(10.0)),
    "a0_callback": (lambda: a0(mode="callback", rate=ms(10)), s(10.0)),
}


def final_state(make, duration=s(10.0)):
    """The rig's state after ``duration``, by path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = make().build()
        built.run(duration)
    return {f"{node}.{name}": value for node, fields in built.current_state().items() for name, value in fields.items()}


@pytest.mark.parametrize("name", RIGS)
def test_each_rig_runs_ten_seconds_with_every_field_finite(name):
    state = final_state(*RIGS[name])
    bad = [path for path, value in state.items() if np.issubdtype(value.dtype, np.inexact) and not np.isfinite(value).all()]
    assert state and bad == []


def test_control_the_gyroscopic_term_taken_explicitly_stops_the_tracked_rig_being_finite(monkeypatch):
    monkeypatch.setattr(RigidInertialMidpointNode, "stepping_mass",
                        lambda self, p, mass_matrix, rotation, omega, dt: mass_matrix)
    with pytest.raises(RunFailed, match="run.non-finite"):
        final_state(tracked)
