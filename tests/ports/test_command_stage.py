"""The command stage (``mime.nodes.control.CommandStageNode``) against a NumPy oracle of the design's order of rungs:
the writer sampled at the port's period and held, delayed by its latency (its rest before), bounded (clipped, or
tripped to its rest for the rest of the run), slew-limited from the value the amplifier drove last, then shaped by its
response (a first-order lag). Drawn over the port's numbers and a written sequence that leaves the bounds and comes back,
at the scales a coil's amplifier has: tens of amperes, kiloamperes a second, milliseconds.

The tolerance is on the inputs' scale, the largest of the written values, the bounds and the rest: a float32 run's
rounding over its steps, 64 units in the last place of that scale.

The bound's flag is compared too, step by step: whether the clip acted, or whether the trip has latched.

**The controls** (``PORT_PLANT``, rated by ``.github/probe/detection_rate.py``): the stage with two rungs swapped, each
a plausible ordering of an amplifier chain (``slew_before_bound``: the slew acts on what arrives, then the bound;
``bound_before_delay``: the bound acts on what was written, before the link samples and delays it, which differs only
where a value beyond the bounds falls between samples; ``respond_before_slew``: the lag before the slew, as though the
amplifier's output stage slewed what the coil let through, which the drawn examples missed on 10 runs of 200 and the
explicit case below catches on every run), and a trip that does not latch (``trip_unlatched``):
each must fail this test on every run, its rate measured, not assumed: a swap the drawn examples could not tell apart
would be no evidence."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from port_cases import Link, command_flags, command_oracle, planted, stage_of, stepped

from mime.core import run_dtype
from mime.nodes.control import CommandStageNode

DT = 1e-3
STEPS = 24
SHAPE = (3,)
#: The orders the plants put the stage's rungs in.
PLANTED_RUNGS = {"slew_before_bound": ("delay", "slew", "bound", "respond"),
                 "bound_before_delay": ("bound", "delay", "slew", "respond"),
                 "respond_before_slew": ("delay", "bound", "respond", "slew")}


@st.composite
def links(draw: st.DrawFn) -> tuple[Link, np.ndarray]:
    """A port's numbers and a written sequence: a bound of 1 to 30 A about zero, a slew moving 0.2 to 4 bounds' worth a
    step or none, a lag of 1 to 20 ms or none, a period of 1 to 4 steps and a latency of 0 to 5, and a sequence of
    square waves reaching past the
    bound and back, with a one-step spike past it at a drawn step (which falls between samples wherever the period is
    above one and the step is not a multiple of it)."""
    bound = draw(st.floats(1.0, 30.0))
    bounded = draw(st.sampled_from(["", "clip", "trip"]))
    slewed = draw(st.booleans())
    lagged = draw(st.booleans())
    link = Link(period=draw(st.integers(1, 4)), latency=draw(st.integers(0, 5)),
                low=-bound if bounded else None, high=bound if bounded else None, trip=bounded == "trip",
                rest=draw(st.floats(-0.5, 0.5)) * bound,
                slew=draw(st.floats(0.2, 4.0)) * bound / DT if slewed else None,
                tau=draw(st.floats(1e-3, 2e-2)) if lagged else None)
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    written = np.empty((STEPS, *SHAPE))
    for j in range(SHAPE[0]):
        half = int(rng.integers(2, 6))
        levels = (bound * rng.uniform(1.1, 2.0), bound * rng.uniform(-0.9, 0.9))
        written[:, j] = [levels[(n // half) % 2] for n in range(STEPS)]
    spike = int(rng.integers(1, STEPS))
    written[spike] = bound * rng.uniform(2.5, 3.0) * rng.choice([-1.0, 1.0], size=SHAPE)
    return link, written


def scale_of(link: Link, written: np.ndarray) -> float:
    return max(float(np.max(np.abs(written))), abs(link.rest), abs(link.high or 0.0))


def tolerance(scale: float) -> float:
    return 64.0 * float(np.finfo(np.dtype(run_dtype())).eps) * scale * STEPS


@contextmanager
def planted_stage() -> Iterator[None]:
    """The stage as the plant leaves it: its rungs reordered, or its trip unlatched; the design's with nothing planted."""
    held = (CommandStageNode.RUNGS, CommandStageNode._bound)  # noqa: SLF001 - a plant replaces a rung
    try:
        for name, order in PLANTED_RUNGS.items():
            if planted(name):
                CommandStageNode.RUNGS = order
        if planted("trip_unlatched"):
            original = held[1]

            def unlatched(self, key, value, p, state, out, count, dt):
                cleared = {**state, f"tripped_{key}": jnp.zeros_like(state[f"tripped_{key}"])}
                return original(self, key, value, p, cleared, out, count, dt)

            CommandStageNode._bound = unlatched  # noqa: SLF001
        yield
    finally:
        CommandStageNode.RUNGS, CommandStageNode._bound = held  # noqa: SLF001


@settings(deadline=None)
@given(links())
def test_the_command_stage_delivers_the_design_order_of_its_rungs(drawn):
    link, written = drawn
    flag = None if link.low is None else ("tripped_currents" if link.trip else "clipped_currents")
    with planted_stage():
        held = stepped(stage_of(link, SHAPE, DT), written, DT, flag=flag)
    delivered, flags = held if isinstance(held, tuple) else (held, np.zeros(STEPS))
    expected = command_oracle(written, link, DT)
    worst = float(np.max(np.abs(delivered - expected)))
    assert worst <= tolerance(scale_of(link, written)), (link, worst)
    np.testing.assert_array_equal(flags, command_flags(written, link))


def test_a_trip_acts_on_what_the_link_delivers_not_on_what_was_written():
    """The case that tells the bound before the delay from after it: a one-step spike past a trip's bounds falling
    between the samples of a port of period 4. The link never delivers it, so the port never trips: it delivers its
    first sample throughout, and its flag stays down. Bounding first would trip at the spike's step."""
    link = Link(period=4, low=-1.0, high=1.0, trip=True, rest=0.25)
    written = np.full((8, 1), 0.5)
    written[1] = 3.0
    with planted_stage():
        delivered, flags = stepped(stage_of(link, (1,), DT), written, DT, flag="tripped_currents")
    np.testing.assert_array_equal(delivered[:, 0], np.full(8, 0.5))
    np.testing.assert_array_equal(flags, np.zeros(8))


def test_a_trip_latches_the_rest_and_flags_it():
    link = Link(low=-1.0, high=1.0, trip=True, rest=0.25)
    written = np.array([[0.5], [2.0], [0.5], [0.5]])
    node = stage_of(link, (1,), DT)
    state = node.initial_state()
    seen = []
    for value in written:
        state = node.fresh_update(state, {"currents": jnp.asarray(value)}, DT)
        seen.append((float(state["currents"][0]), float(state["tripped_currents"])))
    assert seen == [(0.5, 0.0), (0.25, 1.0), (0.25, 1.0), (0.25, 1.0)]


def test_a_port_buffers_samples_not_steps():
    """A latency's buffer holds ``ceil(L / k) + 1`` samples: a 17-step period with 23 steps of latency holds 3."""
    node = stage_of(Link(period=17, latency=23), (2,), DT)
    assert node.initial_state()["buffer_currents"].shape == (3, 2)


def test_the_lag_follows_the_slew():
    """The case that tells the response before the slew from after it: a step of 10 A, slewed at 2 A a step, then
    lagged at 5 ms. In the design's order the lag follows the slewed ramp; lagged first, the step's lag rises slower
    than the slew and passes through it unslewed, another value at every step of the rise."""
    link = Link(slew=2.0 / DT, tau=5e-3)
    written = np.concatenate([np.full((8, 1), 10.0), np.zeros((8, 1))])
    with planted_stage():
        delivered = stepped(stage_of(link, (1,), DT), written, DT)
    expected = command_oracle(written, link, DT)
    assert float(np.max(np.abs(delivered - expected))) <= tolerance(10.0), (delivered[:, 0], expected[:, 0])
