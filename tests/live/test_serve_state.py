"""What ``mime serve``'s live view keeps, as plain data (``mime.live.ServeState``), tested without a terminal or a server:
the default rows of the table of values and the rows ``--watch`` names, a name the frames do not carry refused by its code
with the nearest name, the minimum and maximum over every frame taken and not over the frames that happen to be drawn, a flag
seen raised, the speed and the frame rate from a status, a refused connection said once with the others, and a command said
as its name and arguments only.

The controls: a state that updates its extremes only when it is sampled for a draw (the fault this file's extremes test is
there to find) fails the same assertion on every run, because the peak is planted at a frame that is never sampled; and a
selection that would add to the frames is not possible, since ``watch_rows`` takes the schema and cannot write to it
(checked by the schema's id, which is a hash of its fields)."""

import json
import socket
import time
from pathlib import Path

import numpy as np
import pytest

from mime.build import ArgumentRefused
from mime.live import FlagWatch, Frame, FrameSchema, ServeState, default_rows, watch_rows
from mime.live import SCALAR, VECTOR, ARRAY, FLAG
from mime.live._state import REFUSED_EVERY
from mime.live._zmtp import PeerEvent, Server
from mime.results._layout import FlagLayout

SPEED = "swim_speed:umr.speed"


def schema() -> FrameSchema:
    """The frames of a small rig: a body's pose, a chain's 27 frame poses, a 3-vector field, an observation's scalar output
    and a flag."""
    return FrameSchema.of([
        ("umr", "pose", np.zeros(7, np.float32), "m; 1"),
        ("rpm.arm", "frame_poses", np.zeros((27, 7), np.float32), "m; 1"),
        ("rpm.magnetic", "flux", np.zeros(3, np.float32), "T"),
        ("swim_speed:umr", "speed", np.zeros((), np.float32), "m/s"),
        ("guard", "tripped", np.zeros((), np.float32), "1"),
        ("bare", "level", np.zeros((), np.float32), None),
    ])


BOUND = [{"node": "umr", "field": "pose", "index": None, "kind": "pose"},
         {"node": "rpm.arm", "field": "frame_poses", "index": 0, "kind": "pose"},
         {"node": "rpm.arm", "field": "frame_poses", "index": 1, "kind": "pose"}]
FLAGS = [FlagWatch("guard.tripped", "guard.tripped")]


def frame(number: int, speed: float = 0.0, tripped: float = 0.0, position=(0.0, 0.0, 0.0)) -> Frame:
    values = {spec.path: np.zeros(spec.shape, np.float32) for spec in schema().fields}
    values[SPEED] = np.asarray(speed, np.float32)
    values["guard.tripped"] = np.asarray(tripped, np.float32)
    values["umr.pose"][:3] = position
    return Frame(number, number * 16, number * 0.016, values)


def refusal_of(specs):
    with pytest.raises(ArgumentRefused) as caught:
        watch_rows(schema(), specs, BOUND, FLAGS)
    (refusal,) = caught.value.refusals
    return refusal


def test_by_default_the_table_holds_each_scalar_output_each_flag_and_each_bodys_position_and_not_the_chains_frames():
    rows = default_rows(schema(), BOUND, FLAGS)
    assert [(row.label, row.kind) for row in rows] == [
        (SPEED, SCALAR), ("bare.level", SCALAR), ("guard.tripped", FLAG), ("umr.pose[0:3]", VECTOR)]
    by_label = {row.label: row for row in rows}
    assert by_label["umr.pose[0:3]"].unit == "m", "the position is the metres of a pose's 'm; 1', not its quaternion's"
    assert by_label[SPEED].unit == "m/s" and by_label["bare.level"].unit is None
    assert not any("frame_poses" in row.label for row in rows), "27 frame poses are not a reading"
    assert "guard.tripped" not in [row.label for row in rows if row.kind == SCALAR], "a flag is a flag, not a scalar"


def test_watch_names_the_rows_and_all_adds_every_scalar_and_three_vector_and_none_hides_the_table():
    rows, shown = watch_rows(schema(), ["umr.pose[0:3]", "rpm.arm.frame_poses[26,0:3]", "umr.pose[3]", SPEED], BOUND, FLAGS)
    assert shown and [(row.label, row.kind, row.shape, row.unit) for row in rows] == [
        ("umr.pose[0:3]", VECTOR, (3,), "m"), ("rpm.arm.frame_poses[26,0:3]", VECTOR, (3,), "m"),
        ("umr.pose[3]", SCALAR, (), "1"), (SPEED, SCALAR, (), "m/s")]
    assert rows[2].flat == 3 and watch_rows(schema(), ["rpm.arm.frame_poses[26,6]"], BOUND, FLAGS)[0][0].flat == 26 * 7 + 6
    everything, _ = watch_rows(schema(), ["all"], BOUND, FLAGS)
    assert [row.label for row in everything] == [SPEED, "bare.level", "guard.tripped", "umr.pose[0:3]",
                                                   "rpm.magnetic.flux"]
    big, _ = watch_rows(schema(), ["rpm.arm.frame_poses"], BOUND, FLAGS)
    assert big[0].kind == ARRAY and big[0].shape == (27, 7)
    assert watch_rows(schema(), ["none"], BOUND, FLAGS) == ([], False)


def test_a_name_the_frames_do_not_carry_is_refused_by_its_code_with_the_nearest_name():
    refusal = refusal_of([SPEED[:-1]])
    assert refusal.code == "live.watch" and "which the frames do not carry" in refusal.message
    assert f"Did you mean '{SPEED}'" in refusal.action and "--select" in refusal.action
    assert "adds nothing to the frames" in refusal.why
    for spec, part in [("umr.pose[0:3", "is not <node>.<field>"), ("umr.pose[x]", "not an integer or a low:high slice"),
                       ("umr.pose[9]", "index 9 is outside dimension 0"), ("umr.pose[5:3]", "selects nothing"),
                       ("umr.pose[0,0]", "gives 2 indices to umr.pose"), (f"{SPEED}[0]", "gives 1 indices"),
                       ("umr.pose[3:]x", "is not <node>.<field>")]:
        assert part in refusal_of([spec]).message, spec
    assert refusal_of(["none", SPEED]).code == "live.watch"


def test_watching_reads_the_schema_and_cannot_change_what_the_frames_carry():
    held = schema()
    before = (held.id, held.frame_bytes, [spec.record() for spec in held.fields])
    for specs in (None, ["all"], ["none"], ["umr.pose[0:3]"], ["rpm.arm.frame_poses"]):
        watch_rows(held, specs, BOUND, FLAGS)
    assert (held.id, held.frame_bytes, [spec.record() for spec in held.fields]) == before


class SampledState(ServeState):
    """The control: extremes kept only at the draws (``sample``), as a view that sampled its frames would keep them."""

    def take(self, frame_: Frame) -> None:
        with self.lock:
            self.latest = frame_
            self.frames_taken += 1

    def sample(self) -> None:
        with self.lock:
            if self.latest is None:
                return
            for row in self._scalars:
                number = float(self.latest.values[row.path].item(row.flat))
                row.low, row.high = min(row.low, number), max(row.high, number)
                row.samples.append(number)


def drive(state: ServeState) -> tuple[float, float]:
    """1,000 frames of a swim speed that is 1 almost always and -3 at frame 501 and 9 at frame 733, drawn (sampled) at
    every 50th frame from frame 0: neither extreme is at a frame that is drawn. The extremes the state holds for the row."""
    rows, _ = watch_rows(schema(), [SPEED], BOUND, FLAGS)
    state.watch(rows, FLAGS, 1)
    for number in range(1000):
        state.take(frame(number, speed={501: -3.0, 733: 9.0}.get(number, 1.0)))
        if number % 50 == 0:
            state.sample()
    return rows[0].low, rows[0].high


def test_the_minimum_and_maximum_are_over_every_frame_not_over_the_frames_that_are_drawn():
    assert drive(ServeState("t")) == (-3.0, 9.0)


def test_the_control_extremes_kept_at_the_draws_miss_both_peaks_every_run():
    low, high = drive(SampledState("t"))
    assert (low, high) == (1.0, 1.0) and (low, high) != (-3.0, 9.0), "the check above would fail it: it finds a missed peak"


def test_a_flag_is_seen_raised_at_the_first_frame_that_shows_it_and_reads_as_the_results_do():
    state = ServeState("t")
    state.watch(default_rows(schema(), BOUND, FLAGS), FLAGS, 1)
    for number in range(10):
        state.take(frame(number, tripped=1.0 if number in (4, 5, 8) else 0.0))
    assert state.raised == {"guard.tripped": 4 * 16}, "first seen, never moved by a later frame"
    # the flag's reading is a result's: nonzero without limits, a magnitude above a limit with them
    layout = FlagLayout("o", "e", "n", "f", "m", limits=(1.0, float("inf")))
    watch = FlagWatch("n.f", "n.f", (1.0, float("inf")))
    plain = FlagLayout("o", "e", "n", "f", "m")
    for value in ([0.5, 1e9], [-1.5, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, -2.0]):
        assert watch.raised(value) == bool(layout.raised(np.asarray(value)[None])[0]), value
        assert FlagWatch("n.f", "n.f").raised(value) == bool(plain.raised(np.asarray(value)[None])[0]), value


def test_the_speed_and_the_frame_rate_come_from_the_status_over_a_few_seconds():
    state = ServeState("t")
    base = {"state": "running", "pace": None, "recording": False, "activity": "stepping", "pending": [],
            "non_finite": None}
    assert state.real_time_factor() is None and state.frames_per_second() is None
    state.note_status({**base, "step": 0, "time": 0.0, "frames": 0, "wall_time": 1.0}, 100.0)
    assert state.real_time_factor() is None, "no span to say it over yet"
    for second in range(1, 4):
        state.note_status({**base, "step": 1000 * second, "time": 2.0 * second, "frames": 300 * second,
                           "wall_time": 1.0 + second}, 100.0 + second)
    assert state.real_time_factor() == pytest.approx(2.0) and state.frames_per_second() == pytest.approx(300.0)
    assert state.wall(103.5) == pytest.approx(4.5), "the wall time carried on from when it was read"
    # the window forgets: after a pause (no time passing) the speed is 0 over the new window
    for second in range(4, 9):
        state.note_status({**base, "step": 3000, "time": 6.0, "frames": 900, "wall_time": 1.0 + second}, 100.0 + second)
    assert state.real_time_factor() == 0.0


def test_connections_are_events_and_refused_ones_are_counted_and_said_once_together():
    state = ServeState("t")
    state.peer("connected", "frames", "127.0.0.1:5000")
    state.peer("connected", "commands", "127.0.0.1:5001")
    state.peer("left", "frames", "127.0.0.1:5000", dropped=7)
    for _ in range(37):
        state.peer("refused", "commands", "127.0.0.1:6000")
    first = [event.text for event in state.drain(10.0)]
    assert first == ["viewer connected (127.0.0.1:5000)", "commander connected (127.0.0.1:5001)",
                     "viewer left; 7 frames dropped for it", "37 connections refused (not ZMTP, or no handshake)"]
    state.peer("refused", "commands", "")
    assert state.drain(10.0 + REFUSED_EVERY / 2) == [], "not again until a few seconds have passed"
    assert [event.text for event in state.drain(10.0 + REFUSED_EVERY)] == [
        "1 connection refused (not ZMTP, or no handshake)"]


def test_a_command_is_said_as_its_name_and_arguments_and_what_came_of_it_and_a_poll_is_not_said():
    state = ServeState("t", step=3200)
    ok = {"ok": True, "step": 3200, "pending": []}
    state.command({"command": "status"}, ok)
    state.command({"command": "hello"}, ok)
    state.command({"command": "set", "values": {"offset": 0.001}}, ok)
    state.command({"command": "pause"}, {**ok, "pending": ["pause"]})
    state.command({"command": "record", "on": False}, ok)
    state.command({"command": "pace", "rtf": None}, ok)
    state.command({"command": "step", "chunks": 3}, ok)
    state.command({"command": "x" * 500}, {"ok": False, "errors": [{"code": "live.command"}]})
    texts = [event.text for event in state.drain(0.0)]
    assert texts[:6] == ["command: set offset=0.001 (from step 3,200)",
                         "command: pause (taken; it acts at the chunk's end)", "command: record off (from step 3,200)",
                         "command: pace unpaced (from step 3,200)", "command: step 3 chunks (stepped to step 3,200)",
                         f"refused: {'x' * 47}… (live.command)"] and len(texts) == 6


def test_what_a_status_states_of_recent_commands_carries_names_arguments_and_steps_and_nothing_of_who():
    state = ServeState("t")
    status = {"state": "paused", "step": 5, "time": 0.1, "frames": 2, "pace": 1.0, "recording": True, "activity": "idle",
              "pending": ["stop"], "non_finite": None, "wall_time": 12.5, "clients": {"commands": 2, "frames": 1},
              "recent": [{"command": "set", "arguments": {"label": "offset", "value": [0.5]}, "step": 4}]}
    state.note_status(status, 1.0)
    assert state.clients == (2, 1) and state.recording and state.pending == ("stop",)
    assert [(each.command, dict(each.arguments), each.step) for each in state.recent] == [
        ("set", {"label": "offset", "value": [0.5]}, 4)]
    assert state.wall(2.0) == pytest.approx(13.5)
    state.note_status({k: v for k, v in status.items() if k not in ("wall_time", "clients", "recent")}, 3.0)
    assert state.wall(3.0) is None and state.recent == (), "a server that does not say them: the view copes"


CAPTURED = json.loads((Path(__file__).parent / "libzmq_handshakes.json").read_text())


def polled(server: Server, until, what: str) -> list[PeerEvent]:
    """The events the server has reported once ``until`` holds of them: a guard against waiting for ever."""
    found: list[PeerEvent] = []
    deadline = time.monotonic() + 30
    while not until(found):
        assert time.monotonic() < deadline, f"never saw: {what}; saw {found}"
        server.poll(0.02)
        found += server.take_events()
    return found


def test_the_server_reports_who_connected_who_left_and_who_never_finished_a_handshake():
    server = Server("tcp://127.0.0.1:0", "tcp://127.0.0.1:0")
    try:
        def connect(kind: str) -> socket.socket:
            host, _, port = server.endpoints[kind].removeprefix("tcp://").rpartition(":")
            return socket.create_connection((host, int(port)), timeout=5)

        viewer, commander, scanner = connect("PUB"), connect("REP"), connect("REP")
        viewer.sendall(bytes.fromhex(CAPTURED["SUB"]))
        commander.sendall(bytes.fromhex(CAPTURED["REQ"]))
        scanner.sendall(b"GET / HTTP/1.1\r\n\r\n")
        found = polled(server, lambda events: {e.what for e in events} >= {"connected", "refused"} and
                       sum(e.what == "connected" for e in events) == 2, "two connected, one refused")
        by = {(e.what, e.role) for e in found}
        assert {("connected", "frames"), ("connected", "commands"), ("refused", "commands")} <= by
        assert all(e.address.startswith("127.0.0.1:") for e in found), "where the connection came from, as the transport gave it"
        assert server.counts() == (1, 1), "a refused peer is not a client"
        viewer.close()
        found = polled(server, lambda events: any(e.what == "left" for e in events), "the viewer left")
        left = [e for e in found if e.what == "left"]
        assert [(e.role, e.dropped) for e in left] == [("frames", 0)] and server.counts() == (1, 0)
        commander.close()
        scanner.close()
        polled(server, lambda events: any(e.what == "left" and e.role == "commands" for e in events), "the commander left")
        assert server.counts() == (0, 0) and server.take_events() == []
    finally:
        server.close()


NAMED = [FlagWatch("guard.tripped_limit", "guard.tripped", (1.0,))]


def test_a_flag_that_has_a_name_of_its_own_is_watched_by_the_name_its_row_is_labelled_with_and_by_its_field():
    """The module's own rule is that a row is one name a person can pass to ``--watch``. The default table labels a flag by
    its name, which is not the frame field it is read from when the flag has a name of its own."""
    (default,) = [row for row in default_rows(schema(), BOUND, NAMED) if row.kind == FLAG]
    assert default.label == "guard.tripped_limit" and default.path == "guard.tripped"
    for spec in (default.label, default.path):
        (row,), shown = watch_rows(schema(), [spec], BOUND, NAMED)
        assert shown and (row.label, row.path, row.kind, row.flag) == (default.label, default.path, FLAG, NAMED[0]), spec
    both, _ = watch_rows(schema(), ["guard.tripped_limit", "guard.tripped"], BOUND, NAMED)
    assert [row.label for row in both] == ["guard.tripped_limit"], "one flag, one row, whichever name reached it"
    with pytest.raises(ArgumentRefused) as caught:  # the control: a name that is neither is still refused
        watch_rows(schema(), ["guard.tripped_limi"], BOUND, NAMED)
    assert caught.value.refusals[0].code == "live.watch"
