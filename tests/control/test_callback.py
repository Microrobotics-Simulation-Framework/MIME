"""Mode 2, the callback, and the chunked run it needs: the controller called once every control period on the host,
with the state as the period starts, between compiled chunks; its writes acting from the step its latency puts them
at; the per-chunk hook seeing every chunk; and A0's controller closing its loop on the kinematic arm.

Each property is a check that returns what is wrong, so that its control, a planted fault in the run, can be shown
to fail it: (b) a latency off by one step, (c) a hook skipped on the last chunk, (e) a callback called twice in one
period. Controls (a) and (d) are in ``test_declaration.py``."""

import numpy as np
import pytest
from chain_cases import tracked_rpm, tracking
from control_cases import Misspelt, Ticker, expected_angles, spun_magnet

import mime.control._run as run_module
from mime.control import Chunk, ControllerError
from mime.things import State
from mime.units import mm, ms

PERIOD = 3


# -- latency --------------------------------------------------------------------------------------------------------


def latency_failures(latency: int, period: int = PERIOD, steps: int | None = None) -> list[str]:
    """Every step of a run whose angle is not the rule's. The run covers the latency and three more periods by
    default, so that at least three writes act."""
    steps = steps if steps is not None else period * (latency // period + 4)
    built = spun_magnet(rate=ms(period), latency=ms(latency) if latency else None).build()
    history = built.run(ms(steps), history=True).history
    assert history is not None
    found = [float(a) for a in history["spin.motor"]["angle"]]
    expected = expected_angles(steps, latency, period)
    return [] if found == expected else [f"period {period}, latency {latency}: angles {found}, expected {expected}"]


@pytest.mark.parametrize("latency", [0, 1, 2, 4, 5, 9, 11])
@pytest.mark.parametrize("period", [1, 2, 3, 5])
def test_a_declared_latency_delays_the_writes_by_exactly_its_steps(period, latency):
    """The rule holds across periods and latencies: at a period of one step (A0's), a latency longer than the period
    (several writes in flight at once), one equal to it and one that is not a multiple of it."""
    assert latency_failures(latency, period) == []


def test_control_a_latency_off_by_one_step_is_caught(monkeypatch):
    """Control (b): the run puts each write one step later than declared, and the angles show it."""
    assert latency_off_by_one(monkeypatch) != []


def latency_off_by_one(monkeypatch) -> list[str]:
    held = run_module.Callback.tick

    def late(self, graph, step):
        held(self, graph, step)
        start, fed = self.waiting.pop()
        self.waiting.append((start + 1, fed))

    monkeypatch.setattr(run_module.Callback, "tick", late)
    return latency_failures(2)


def test_the_report_states_the_latency_in_steps_and_the_rule():
    line = spun_magnet(latency=ms(4)).build().report.controller[0]
    assert "every 3 steps (0.003 s), latency 4 steps (0.004 s)" in line
    assert "its writes act on every step from the one that starts at t + latency" in line


# -- the hook --------------------------------------------------------------------------------------------------------


def hook_failures(chunks: list[Chunk], history: dict, steps: int, period: int, dt: float) -> list[str]:
    """Every way the chunks a hook saw differ from the run's: one per period, in order, each at its end's step and
    time, with the state the run's history holds there."""
    failures: list[str] = []
    ends = [period * (k + 1) for k in range(steps // period)]
    if [c.index for c in chunks] != list(range(len(ends))) or [c.steps for c in chunks] != ends:
        return [f"chunks {[(c.index, c.steps) for c in chunks]}, expected ends {ends}"]
    for chunk in chunks:
        if not np.isclose(chunk.time, chunk.steps * dt, rtol=0, atol=1e-15):
            failures.append(f"chunk {chunk.index} at {chunk.time} s, its end is {chunk.steps * dt} s")
        for node, fields in history.items():
            for name, values in fields.items():
                if not np.array_equal(chunk.state[node][name], values[chunk.steps - 1]):
                    failures.append(f"chunk {chunk.index}: {node}.{name} is not the state after step {chunk.steps}")
    return failures


def hooked(steps: int = 12):
    built = spun_magnet(rate=ms(PERIOD)).build()
    seen: list[Chunk] = []
    history = built.run(ms(steps), history=True, hook=seen.append).history
    return seen, history, built.report.dt


def test_the_hook_sees_every_chunk_with_its_time_and_state():
    seen, history, dt = hooked()
    assert len(seen) == 4
    assert hook_failures(seen, history, 12, PERIOD, dt) == []


def test_control_a_hook_skipped_on_the_last_chunk_is_caught(monkeypatch):
    """Control (c): the run does not call the hook after its last chunk."""
    assert hook_skipped_on_the_last_chunk(monkeypatch) != []


def hook_skipped_on_the_last_chunk(monkeypatch) -> list[str]:
    held = run_module.ChunkedRun.run

    def skipping(self, steps, *, history, hook=None, chunk=None):
        end = self.steps + steps
        given = None if hook is None else (lambda c: None if c.steps == end else hook(c))
        return held(self, steps, history=history, hook=given, chunk=chunk)

    monkeypatch.setattr(run_module.ChunkedRun, "run", skipping)
    seen, history, dt = hooked()
    return hook_failures(seen, history, 12, PERIOD, dt)


def test_a_hook_chunks_a_run_with_no_controller_and_changes_nothing_it_computes():
    """Script A (its 0.5 ms step), with no controller: run in chunks of 4 steps for a hook, its 16 steps are bit for
    bit those of one compiled scan, and the hook sees each chunk."""
    from script_a import script_a

    from mime.units import m_per_s

    def a():
        return script_a(velocity=m_per_s((1e-3, 0.0, 0.0)))

    whole = a().build().run(ms(8), history=True).history
    built = a().build()
    assert built.report.run.chunk is None
    seen: list[Chunk] = []
    chunked = built.run(ms(8), history=True, hook=seen.append, chunk=ms(2)).history
    assert whole is not None and chunked is not None
    assert all(np.array_equal(whole[n][f], chunked[n][f]) for n in whole for f in whole[n])
    assert hook_failures(seen, chunked, 16, 4, built.report.dt) == []


def test_a_run_is_a_whole_number_of_chunks_and_a_hook_needs_them():
    built = spun_magnet(rate=ms(PERIOD)).build()
    with pytest.raises(ValueError, match=r"not a whole number of chunks of 3 steps \(0.003 s\); the nearest runs are "
                                         r"3 steps \(0.003 s\) and 6 steps \(0.006 s\)"):
        built.run(ms(4))
    with pytest.raises(ValueError, match=r"the nearest runs are 3 steps \(0.003 s\)$"):
        built.run(ms(2))
    with pytest.raises(ValueError, match="a controller's chunks are its control period"):
        built.run(ms(3), hook=print, chunk=ms(1))
    from chain_cases import kinematic_rpm

    with pytest.raises(ValueError, match="a hook needs chunks"):
        kinematic_rpm().build().run(ms(2), hook=print)


# -- the callback, once per period ---------------------------------------------------------------------------------


def callback_failures(seen: list[dict], history: dict, initial: np.ndarray, steps: int) -> list[str]:
    """The controller is called once a period, and each call reads the body's pose as the period starts: the initial
    state for the first, the state after the step before for each later one."""
    ticks = steps // PERIOD
    if len(seen) != ticks:
        return [f"{len(seen)} calls in {ticks} periods"]
    starts = [initial] + [history["umr"]["pose"][k * PERIOD - 1] for k in range(1, ticks)]
    return [f"call {k} read a pose that is not the state as its period starts" for k, (read, start) in
            enumerate(zip(seen, starts, strict=True)) if not np.array_equal(read["pose"], start)]


def called(steps: int = 12) -> list[str]:
    seen: list[dict] = []
    built = spun_magnet(controller=Ticker(seen=seen), rate=ms(PERIOD)).build()
    initial = np.asarray(built.graph.get_node_state("umr")["pose"])
    history = built.run(ms(steps), history=True).history
    return callback_failures(seen, history, initial, steps)


def test_the_callback_is_called_once_a_period_with_the_state_as_the_period_starts():
    assert called() == []


def test_control_a_callback_called_twice_in_one_period_is_caught(monkeypatch):
    """Control (e): the run calls the controller's step twice at its first tick."""
    assert called_twice(monkeypatch) != []


def called_twice(monkeypatch) -> list[str]:
    held = run_module.Callback.tick

    def twice(self, graph, step):
        if self.ticks == 0:
            held(self, graph, step)
            self.waiting.pop()
        held(self, graph, step)

    monkeypatch.setattr(run_module.Callback, "tick", twice)
    return called()


def test_runs_continue_where_the_last_left_its_controller():
    """Two runs of 9 ms with a latency of 4 steps give the 18 steps of one: the controller's state and its writes
    waiting to act carry over, and the second run's first steps act on writes the first run's ticks made."""
    one = spun_magnet(latency=ms(4)).build().run(ms(18), history=True).history
    built = spun_magnet(latency=ms(4)).build()
    first, second = (built.run(ms(9), history=True).history for _ in range(2))
    assert one is not None and first is not None and second is not None
    joined = np.concatenate([first["spin.motor"]["angle"], second["spin.motor"]["angle"]])
    assert np.array_equal(joined, one["spin.motor"]["angle"])
    assert [float(a) for a in joined] == expected_angles(18, 4, PERIOD)


def test_writes_that_are_not_the_controllers_labels_stop_the_run():
    with pytest.raises(ControllerError, match=r"the writes of tick 0 hold \['angel', 'angle'\]"):
        spun_magnet(controller=Misspelt()).build().run(ms(3))


# -- A0's controller on the kinematic arm ---------------------------------------------------------------------------


def tracked(x: float, smoothing: float):
    """A0's rig closed by its controller every step with no latency, the body started ``x`` along the tube: the
    flange's position at the start and after each of 10 steps, and the body's after each."""
    exp = tracked_rpm(controller=tracking(smoothing=smoothing), rate=ms(1), latency=None)
    for declaration in exp._declarations:
        thing = getattr(declaration, "thing", None)
        if getattr(thing, "name", None) == "umr":
            object.__setattr__(thing, "initial", State(**{**dict(thing.initial), "position": mm((x * 1e3, 0, -1))}))
    built = exp.build()
    flange = next(p for p in built.plans.entries if p.owner == "rpm.arm").segment.frames.index("link_6")
    start = np.asarray(built.graph.get_node_state("rpm.arm")["frame_poses"])[flange, :3]
    history = built.run(ms(10), history=True).history
    assert history is not None
    return start, history["rpm.arm"]["frame_poses"][:, flange, :3], history["umr"]["pose"][:, :3]


@pytest.mark.parametrize("x", [0.0, 0.02, 0.08])
def test_a0s_controller_holds_the_flange_at_its_standoff_above_the_body_within_its_reach(x):
    """Unfiltered (smoothing 1), each step's flange is at the target the controller computed from the state that
    step started from: 15 cm above the body's distance along x, held within 5 cm of the origin (the 8 cm body is
    followed to the envelope's edge), to the inverse kinematics' five iterations."""
    _, flange, body = tracked(x, 1.0)
    along = np.clip(np.concatenate([[x], body[:-1, 0]]), -0.05, 0.05)
    target = np.stack([along, np.zeros_like(along), np.full_like(along, 0.15)], axis=1)
    assert np.max(np.abs(flange - target)) < 3e-5


def test_a0s_controller_moves_its_target_by_its_smoothing_each_tick():
    """Filtered as the 0.2 rig was (0.005 a tick), the flange starts from the home pose and has covered the share
    ``1 - 0.995^n`` of the way to the body's target after ``n`` ticks."""
    start, flange, _ = tracked(0.02, 0.005)
    covered = (flange[:, 0] - start[0]) / (0.02 - start[0])
    expected = 1.0 - 0.995 ** np.arange(1, len(covered) + 1)
    assert np.max(np.abs(covered - expected)) < 0.02 * expected[-1]
