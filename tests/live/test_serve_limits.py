"""``mime serve``'s optional status keys, its ready file, ``--idle-timeout``, ``--max-wall`` and ``--watch``, each against a
server in a process of its own (A0's apparatus closed by ``ArmTracking``, ``live_cases.tracked``).

``wall_time``, ``clients`` and ``recent`` are in every status: the seconds since the session opened, the commanders and
the viewers connected, and the last commands taken, each as its name, its arguments and the step it took effect at. They
carry no address of a peer and nothing a peer wrote but what the session accepted, so a request refused is not in them
(control: the check that finds an address in a status is run on a status with one planted). A client that does not read
them is unaffected: ``LiveClient`` reads none of them.

``--idle-timeout`` stops a session with no client connected for that long, writing its run folder and saying why on
standard output; a client connected resets it (control: with one connected for longer than the timeout the server serves
on, and without the flag a server with no client serves on). ``--max-wall`` stops it at the wall time, a client
connected or not. A time that is not above zero is refused by its code before anything is built.

``--watch`` chooses the rows of a view's table and nothing the frames carry: the frames a client receives with
``--watch all`` are byte for byte the frames it receives without (control: ``--select`` of another field is not)."""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
from live_cases import saved

from mime.live import PROTOCOL, LiveClient

MIME = [sys.executable, "-m", "mime.cli"]
#: A guard against waiting for ever, never a measure of speed.
HANG = 600


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> Path:
    return saved(tmp_path_factory.mktemp("limits"))


def start(launch, folder: Path, tmp_path: Path, *extra: str, name: str = "run") -> tuple[subprocess.Popen, Path, Path]:
    """``mime serve`` of ``folder`` once its ready file is written: the process, the ready file and the run folder."""
    ready, out = tmp_path / f"{name}.ready.json", tmp_path / f"{name}.out"
    process = launch([*MIME, "serve", str(folder), "--commands", "tcp://127.0.0.1:0", "--frames", "tcp://127.0.0.1:0",
                      "--ready-file", str(ready), "--out", str(out), *extra],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + HANG
    while not ready.exists() or not ready.read_text():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, "mime serve never wrote its ready file"
        time.sleep(0.05)
    return process, ready, out


def connect(ready: Path) -> LiveClient:
    endpoints = json.loads(ready.read_text())
    return LiveClient(endpoints["commands"], endpoints["frames"], timeout=60.0)


#: The keys of the ready file, under mime.live/1's rule: fields may be added and a reader ignores what it does not know.
#: ``mechanism`` and ``server_key`` are reserved for a later CURVE session and are not written now.
READY_KEYS = {"protocol", "commands", "frames", "out"}


def holds_an_address(status: dict, ports: set[int]) -> bool:
    """Whether a status names a peer's address: an IPv4 address in its text, or a port of a connection it was reached by."""
    text = json.dumps(status)
    return re.search(r"\d+\.\d+\.\d+\.\d+", text) is not None or any(re.search(rf"\b{port}\b", text) for port in ports)


def test_the_ready_file_holds_the_four_keys_and_none_of_the_reserved_ones(launch, folder, tmp_path):
    process, ready, out = start(launch, folder, tmp_path)
    held = json.loads(ready.read_text())
    assert set(held) == READY_KEYS and held["protocol"] == PROTOCOL and held["out"] == str(out)
    assert "mechanism" not in held and "server_key" not in held, "reserved for a later CURVE session, not written yet"
    for endpoint in (held["commands"], held["frames"]):
        assert endpoint.startswith("tcp://127.0.0.1:")
    # the control: a ready file that gained a key fails the pin
    assert set({**held, "mechanism": "curve"}) != READY_KEYS
    process.terminate()
    process.communicate(timeout=HANG)


def test_every_status_carries_the_wall_time_the_clients_and_the_recent_commands_and_no_peer_address(launch, folder, tmp_path):
    process, ready, out = start(launch, folder, tmp_path)
    live = connect(ready)
    peers = {live.commands.sock.getsockname()[1], live.frames.sock.getsockname()[1]}  # this client's ends
    first = live.request("status")
    assert first["wall_time"] > 0.0 and first["clients"] == {"commands": 1, "frames": 1} and first["recent"] == []
    other = LiveClient(json.loads(ready.read_text())["commands"], json.loads(ready.read_text())["frames"], timeout=60.0)
    assert live.request("status")["clients"] == {"commands": 2, "frames": 2}
    other.close()
    deadline = time.monotonic() + HANG
    while live.request("status")["clients"] != {"commands": 1, "frames": 1}:
        assert time.monotonic() < deadline, "a client that left was never counted out"
        time.sleep(0.05)
    assert live.request("set", values={"offset": 0.002})["ok"]
    refused = live.request("bogus\x1b]0;pwned\x07")
    assert not refused["ok"]
    status = live.request("status")
    (taken,) = status["recent"]  # the command taken, with its arguments and its step, and not the one refused
    assert (taken["command"], taken["step"], taken["arguments"]["label"]) == ("set", 0, "offset")
    assert taken["arguments"]["value"] == pytest.approx(0.002)
    assert not holds_an_address(status, peers), "no peer address, and no peer's port"
    assert set(status["recent"][0]) == {"command", "arguments", "step"}
    # the control: the check finds an address where one is planted
    assert holds_an_address({**status, "clients": {"127.0.0.1:51822": 1}}, peers)
    assert live.hello["ok"] and live.request("hello")["recent"] == status["recent"]
    live.request("stop")
    assert process.communicate(timeout=HANG)[0].endswith(f"wrote {out}\n")


def test_idle_timeout_stops_a_session_nobody_is_connected_to_writes_its_folder_and_says_why(launch, folder, tmp_path):
    process, ready, out = start(launch, folder, tmp_path, "--idle-timeout", "2s")
    stdout, _ = process.communicate(timeout=HANG)
    assert process.returncode == 0 and (out / "outputs.json").is_file() and (out / "run.json").is_file()
    assert "mime serve: no client has been connected for 2 s (--idle-timeout); stopping\n" in stdout
    assert stdout.endswith(f"mime serve: stopped at step 0; wrote {out}\n")


def test_a_client_connected_keeps_the_session_and_one_that_left_starts_the_clock_again(launch, folder, tmp_path):
    process, ready, out = start(launch, folder, tmp_path, "--idle-timeout", "3s")
    live = connect(ready)
    time.sleep(7.0)  # more than twice the timeout, a client connected all the while
    assert process.poll() is None and live.request("status")["ok"], "stopped with a client connected"
    live.close()
    stdout, _ = process.communicate(timeout=HANG)
    assert process.returncode == 0 and "no client has been connected for 3 s" in stdout and (out / "run.json").is_file()


def test_the_control_without_the_flag_a_session_nobody_is_connected_to_serves_on(launch, folder, tmp_path):
    process, ready, out = start(launch, folder, tmp_path)
    time.sleep(5.0)
    assert process.poll() is None, "it must serve on: the two tests above are only evidence if this holds"
    process.terminate()
    assert "idle-timeout" not in process.communicate(timeout=HANG)[0]


def test_max_wall_stops_a_session_at_its_wall_time_a_client_connected_or_not(launch, folder, tmp_path):
    process, ready, out = start(launch, folder, tmp_path, "--max-wall", "3s")
    live = connect(ready)
    stdout, _ = process.communicate(timeout=HANG)
    assert process.returncode == 0 and (out / "run.json").is_file() and (out / "outputs.json").is_file()
    assert "mime serve: its wall time reached 3 s (--max-wall); stopping\n" in stdout
    assert stdout.endswith(f"mime serve: stopped at step 0; wrote {out}\n")
    assert json.loads((out / "run.json").read_text())["live"]["wall_seconds"] >= 3.0
    live.close()


@pytest.mark.parametrize(("option", "value", "part"), [
    ("--idle-timeout", "0", "--idle-timeout is 0.0, not a time above zero"), ("--max-wall", "-5s", "--max-wall is -5.0"),
    ("--max-wall", "soon", "--max-wall soon is not a duration"), ("--idle-timeout", "nan", "is nan, not a time above zero"),
])
def test_a_time_that_is_not_above_zero_is_refused_by_its_code_before_anything_is_built(folder, tmp_path, option, value, part):
    done = subprocess.run([*MIME, "--plain", "serve", str(folder), f"{option}={value}", "--out",
                           str(tmp_path / "out")],
                          capture_output=True, text=True, check=False, timeout=HANG)
    assert done.returncode == 2 and "live.timeout" in done.stderr and part in done.stderr, done.stderr
    assert "15min" in done.stderr and not (tmp_path / "out").exists()


def frames_of(launch, folder: Path, tmp_path: Path, name: str, *extra: str, chunks: int = 6) -> list[bytes]:
    """The raw bytes of the frames a client receives from a server that steps ``chunks`` chunks, the first (the state it
    starts from) included."""
    process, ready, out = start(launch, folder, tmp_path, "--unpaced", *extra, name=name)
    live = connect(ready)
    assert live.request("step", chunks=chunks)["ok"]
    received = [live.frames.receive() for _ in range(chunks + 1)]
    live.request("stop")
    live.close()
    process.communicate(timeout=HANG)
    return received


@pytest.fixture(scope="module")
def served_frames(folder, tmp_path_factory):
    """The frames of two servers, started the same way but for what ``--watch`` is given."""
    from process_groups import stop_group

    started: list[subprocess.Popen] = []

    def launch(args, **options):
        process = subprocess.Popen(list(args), start_new_session=True, **options)
        started.append(process)
        return process

    tmp = tmp_path_factory.mktemp("frames")
    try:
        yield {name: frames_of(launch, folder, tmp, name, *extra) for name, extra in
               (("plain", ()), ("watch_all", ("--watch", "all", "--watch", "umr.pose[0:3]")))}
    finally:
        for process in started:
            stop_group(process)


def test_watch_adds_nothing_to_and_changes_nothing_of_the_frames_a_client_receives(served_frames):
    plain = served_frames["plain"]
    assert len(plain) == 7 and all(frame[:4] == b"MIMF" for frame in plain)
    assert served_frames["watch_all"] == plain, "byte for byte"


def test_the_control_select_of_another_field_does_change_the_frames_so_the_comparison_can_fail(launch, folder, tmp_path,
                                                                                                served_frames):
    selected = frames_of(launch, folder, tmp_path, "select", "--select", "umr.pose")
    assert selected != served_frames["plain"] and len(selected[0]) < len(served_frames["plain"][0])


def test_a_name_the_frames_do_not_carry_is_refused_with_the_nearest_name_before_a_run_folder_is_opened(folder, tmp_path):
    done = subprocess.run([*MIME, "--plain", "serve", str(folder), "--watch", "umr.pos", "--out", str(tmp_path / "out"),
                           "--commands", "tcp://127.0.0.1:0", "--frames", "tcp://127.0.0.1:0"],
                          capture_output=True, text=True, check=False, timeout=HANG)
    assert done.returncode == 2 and "live.watch" in done.stderr, done.stderr
    assert "--watch names 'umr.pos', which the frames do not carry" in done.stderr and "Did you mean 'umr.pose'" in done.stderr
    assert not (tmp_path / "out").exists(), "no run folder was opened"


def words(text: str) -> str:
    return " ".join(text.split())


def test_the_help_and_the_docs_say_that_a_limit_stops_the_session_at_the_end_of_the_chunk_in_hand():
    """F4 of the review of PR #122: a 4 s ``--max-wall`` stopped at 6.7 s, the first chunk, which compiles, being in hand.
    The stop cannot cut a chunk short, so what is promised is where it comes. The control is an option whose help does not
    say it."""
    helped = subprocess.run([*MIME, "serve", "--help"], capture_output=True, text=True, check=True).stdout
    sections = {each.split()[0]: words(each) for each in re.split(r"\n  (?=--)", helped) if each.startswith("--")}
    for option in ("--max-wall", "--idle-timeout"):
        assert "end of the chunk in hand" in sections[option], option
    assert "end of the chunk in hand" not in sections["--watch"], "the control"
    live = words((Path(__file__).resolve().parents[2] / "docs" / "live.md").read_text(encoding="utf-8"))
    assert "Neither limit stops a session at the second" in live and "end of the chunk in hand" in live


def test_the_docs_state_that_the_server_key_is_forty_characters_of_z85_text():
    """F5: the other side of the protocol (a viewer) reads ``server_key`` as 40 characters of Z85 text, as these docs
    reserve it, so the docs state the length."""
    live = words((Path(__file__).resolve().parents[2] / "docs" / "live.md").read_text(encoding="utf-8"))
    assert "`server_key`, the server's long-term public key as Z85 text of 40 characters" in live
    import zmq

    public, _ = zmq.curve_keypair()
    assert len(public) == 40, "and 40 is what libzmq's keypair writes"
