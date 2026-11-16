"""Where ``mime serve`` binds. Nothing authenticates a client, so an address that is not loopback is refused before
anything binds, the command socket as ``live.commands-not-loopback`` and the frame socket as
``live.frames-not-loopback``, in every form it can be written (the wildcards ``0.0.0.0``, ``*``, ``::`` and an empty host,
an address of another machine, a host name, and ``localhost`` where it resolves elsewhere); ``--allow-remote-commands``
binds it all the same and warns, naming it. A second ``mime serve`` on the ports of one running is refused, naming the
port it could not take (``live.address-in-use``), and leaves the first serving and nothing of its own listening.

The transport's server checks for itself as well (``mime.live._zmtp.Server``, defence in depth): it refuses an address
off loopback unless allowed, resolves ``localhost`` once and binds the address it resolved (so a name that resolves
elsewhere between the check and the bind is refused, not bound), refuses a port outside 0 to 65535 as a malformed
endpoint with the socket it opened first closed, and sets on a listener only the options that keep a second server off
its port on its platform: ``SO_REUSEADDR`` on Linux alone, since on macOS and the BSDs it lets a second server bind
``127.0.0.1:P`` while a first holds ``0.0.0.0:P`` (the platforms workflow runs these on macOS).

No test here opens a socket off loopback: every bind of the test's process goes through :class:`Binds`, which records
it and makes one off loopback on loopback instead. A session that gets past its sockets stops at its load
(:class:`Loaded`), so no build runs but the running server's. The control: the same recorder sees the binds of a session
that is allowed (each loopback one, and the one it moved), so its empty record after a refusal is evidence that nothing
bound."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from live_cases import saved
from live_servers import serve_command, watched
from live_servers import client as watched_client

from mime.build import ArgumentRefused, BuildRefused
from mime.live import serve
from mime.live import _zmtp
from mime.live._zmtp import ListenError, Server, is_loopback, listener_options, parse_endpoint

MIME = [sys.executable, "-m", "mime.cli"]
#: A guard against waiting for ever on a server's ready file, never a measure of speed.
HANG = 600
#: The system's own bind, for the probes of a port, which :class:`Binds` does not record.
_BIND = socket.socket.bind
#: The address forms that are not loopback, as an endpoint's host with ``{port}`` to fill.
NOT_LOOPBACK = {"every IPv4 interface": "tcp://0.0.0.0:{port}", "the ZMQ wildcard": "tcp://*:{port}",
                "every IPv6 interface": "tcp://[::]:{port}", "an empty host": "tcp://:{port}",
                "another machine's address": "tcp://192.0.2.1:{port}", "a host name": "tcp://lab-host:{port}",
                "an IPv4-mapped wildcard": "tcp://[::ffff:0.0.0.0]:{port}"}


class Binds:
    """Every ``bind`` of the test's process, as asked: one on loopback is made as asked, and one off loopback is made on
    loopback in its place (``127.0.0.1``, or ``::1`` for IPv6), the same port, so that no test opens a socket on another
    interface. :attr:`sockets` holds each socket bound, to check it is closed."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.asked: list[str] = []
        self.sockets: list[socket.socket] = []
        real = socket.socket.bind

        def bind(sock: socket.socket, address) -> None:
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                host, port = address[0], address[1]
                self.asked.append(host)
                self.sockets.append(sock)
                if not is_loopback(host):
                    address = ("127.0.0.1" if sock.family == socket.AF_INET else "::1", port, *address[2:])
            real(sock, address)

        monkeypatch.setattr(socket.socket, "bind", bind)


@pytest.fixture
def binds(monkeypatch) -> Iterator[Binds]:
    with monkeypatch.context() as patch:
        yield Binds(patch)


class Loaded(Exception):
    """Raised where the session would load its folder: it got past its sockets."""


@pytest.fixture
def unloaded(tmp_path, monkeypatch) -> Path:
    """A folder whose session stops where it would load it, raising :class:`Loaded`."""

    def load(_) -> None:
        raise Loaded

    monkeypatch.setattr("mime.env._check.apply_lock_precision", load)
    return tmp_path


def free_port() -> int:
    """A loopback port nothing listens on now (the system's choice, released)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        _BIND(probe, ("127.0.0.1", 0))
        return probe.getsockname()[1]


def nothing_listens_on(port: int) -> bool:
    """Whether ``port`` is free on loopback: a socket without ``SO_REUSEADDR`` binds it and listens."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            _BIND(probe, ("127.0.0.1", port))
            probe.listen(1)
        except OSError:
            return False
        return True


def served(folder: Path, **endpoints) -> list[str]:
    """What the session's start said on ``warn``, then the codes it was refused with, or ``loaded``."""
    warned: list[str] = []
    try:
        serve(folder, outlive_launcher=True, say=lambda line: None, warn=warned.append, **endpoints)
    except (ArgumentRefused, BuildRefused) as refused:
        return warned + [refusal.code for refusal in refused.refusals]
    except Loaded:
        return [*warned, "loaded"]
    raise AssertionError("the session was served past its load")


@pytest.mark.parametrize("option", ["commands", "frames"])
@pytest.mark.parametrize("form", sorted(NOT_LOOPBACK))
def test_an_address_that_is_not_loopback_is_refused_before_anything_binds(binds, unloaded, option, form):
    port = free_port()
    other = "frames" if option == "commands" else "commands"
    outcome = served(unloaded, **{option: NOT_LOOPBACK[form].format(port=port), other: "tcp://127.0.0.1:0"})
    assert outcome == [f"live.{option}-not-loopback"]
    assert binds.asked == []
    assert nothing_listens_on(port)


def test_localhost_is_loopback_only_where_it_resolves_to_loopback_only(binds, unloaded, monkeypatch):
    resolved = {"localhost": ["127.0.0.1"], "lab-host": ["127.0.0.1"]}

    def getaddrinfo(host, *_):  # a resolver's, which takes a name in any case
        if host.lower() not in resolved:
            raise socket.gaierror(socket.EAI_NONAME, "unknown")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0)) for address in resolved[host.lower()]]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    assert is_loopback("localhost") and is_loopback("LOCALHOST")
    assert not is_loopback("lab-host")  # a name other than localhost is never resolved: it may resolve elsewhere later
    resolved["localhost"] = ["127.0.0.1", "192.0.2.7"]
    assert not is_loopback("localhost")
    resolved["localhost"] = ["192.0.2.7"]
    assert served(unloaded, commands="tcp://localhost:0", frames="tcp://127.0.0.1:0") == ["live.commands-not-loopback"]
    assert binds.asked == []
    del resolved["localhost"]
    assert not is_loopback("localhost")


def test_an_endpoint_that_is_not_tcp_is_refused_before_anything_binds(binds, unloaded):
    assert served(unloaded, commands="127.0.0.1:5555", frames="tcp://127.0.0.1:0") == ["live.endpoint"]
    assert binds.asked == []


@pytest.mark.parametrize("host", ["127.0.0.1", "127.255.255.254", "::1", "localhost"])
def test_loopback_addresses_are_loopback(host):
    assert is_loopback(host)


def test_loopback_binds_with_no_warning_and_the_recorder_sees_it(binds, unloaded):
    assert served(unloaded, commands="tcp://127.0.0.1:0", frames="tcp://localhost:0") == ["loaded"]
    assert binds.asked == ["127.0.0.1", _zmtp.loopback_address("localhost")]  # localhost bound as it was resolved
    assert all(sock.fileno() == -1 for sock in binds.sockets)  # closed when the load raised
    assert served(unloaded, commands="tcp://127.0.0.1:0", frames="tcp://127.0.0.1:0",
                  allow_remote_commands=True) == ["loaded"]  # allowed, and nothing off loopback: no warning


def test_the_explicit_allow_binds_off_loopback_and_warns_naming_each_such_socket(binds, unloaded):
    port = free_port()
    outcome = served(unloaded, commands=f"tcp://0.0.0.0:{port}", frames="tcp://127.0.0.1:0",
                     allow_remote_commands=True)
    assert outcome[1:] == ["loaded"]
    warning = outcome[0]
    assert warning.startswith("mime serve: warning: --allow-remote-commands: commands on tcp://")
    assert f":{port}: the command socket takes commands from whoever reaches it" in warning
    assert "Nothing authenticates a client" in warning and "frames on" not in warning
    assert binds.asked == ["0.0.0.0", "127.0.0.1"]
    both = served(unloaded, commands="tcp://*:0", frames="tcp://0.0.0.0:0", allow_remote_commands=True)
    assert both[1:] == ["loaded"]
    assert "commands on" in both[0] and "frames on" in both[0] and "every frame" in both[0]


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> Path:
    return saved(tmp_path_factory.mktemp("served"))


def test_a_second_serve_on_the_ports_of_one_running_is_refused_naming_the_port(launch, folder, tmp_path, binds):
    ready = tmp_path / "ready.json"
    first = watched(launch, serve_command(folder, ready, tmp_path / "out"), ready)
    deadline = time.monotonic() + HANG
    while not ready.exists() or not ready.read_text():
        assert first.poll() is None, first.communicate()
        assert time.monotonic() < deadline, "mime serve never wrote its ready file"
        time.sleep(0.05)
    endpoints = json.loads(ready.read_text())
    second = subprocess.run([*MIME, "serve", str(folder), "--commands", endpoints["commands"], "--frames",
                             endpoints["frames"], "--out", str(tmp_path / "second")], capture_output=True, text=True,
                            timeout=120)
    assert second.returncode == 2, second.stdout + second.stderr
    assert f"live.address-in-use: cannot listen on --commands {endpoints['commands']}" in second.stderr
    with pytest.raises(BuildRefused, match=f"cannot listen on --frames {endpoints['frames']}") as refused:
        serve(folder, commands="tcp://127.0.0.1:0", frames=endpoints["frames"], outlive_launcher=True,
              say=lambda line: None)
    assert [refusal.code for refusal in refused.value.refusals] == ["live.address-in-use"]
    assert len(binds.sockets) == 2 and all(sock.fileno() == -1 for sock in binds.sockets)
    live = watched_client(ready)
    assert live.request("status")["ok"] and live.request("stop")["ok"] and first.wait(timeout=60) == 0
    live.close()
    assert not (tmp_path / "second").exists()


def test_force_never_writes_through_a_link_among_the_names_a_run_writes(binds, unloaded, tmp_path):
    """(The security review's finding 1.) ``--force`` followed ``out/run.json -> ../victim.txt`` and ``out/experiment ->
    ../victimdir``: the session overwrote the victim and saved the experiment into it. Now a link among the names a run
    writes, or one inside such a folder, is refused (``run.out-not-empty``, naming the entry) with ``--force`` too,
    before anything binds, and what it points at is as it was. Control: the same folder without its links is written
    into (the session gets as far as its load)."""
    victim, victim_dir = tmp_path / "victim.txt", tmp_path / "victimdir"
    victim.write_text("precious\n", encoding="utf-8")
    victim_dir.mkdir()
    ports = {"commands": "tcp://127.0.0.1:0", "frames": "tcp://127.0.0.1:0"}
    saved_at = unloaded / "saved"
    cases = {"file": ("run.json", victim), "folder": ("experiment", victim_dir), "log": ("commands.jsonl", victim)}
    for name, (entry, target) in cases.items():
        out = tmp_path / f"out_{name}"
        out.mkdir()
        (out / "mine.txt").write_text("mine\n", encoding="utf-8")
        (out / entry).symlink_to(target)
        assert served(saved_at, out=out, force=True, **ports) == ["run.out-not-empty"], name
    inside = tmp_path / "out_inside"  # a real experiment/ with a link in it
    (inside / "experiment").mkdir(parents=True)
    (inside / "experiment" / "mime.lock").symlink_to(victim)
    assert served(saved_at, out=inside, force=True, **ports) == ["run.out-not-empty"]
    assert binds.asked == [] and victim.read_text(encoding="utf-8") == "precious\n" and list(victim_dir.iterdir()) == []
    plain = tmp_path / "out_plain"
    plain.mkdir()
    (plain / "mine.txt").write_text("mine\n", encoding="utf-8")
    assert served(saved_at, out=plain, force=True, **ports)[-1] == "loaded"


@pytest.mark.parametrize("port", ["99999", "65536", "-1", "5x", "\u0663", "\uff15\uff15"],
                         ids=["99999", "65536", "-1", "5x", "arabic-indic-3", "fullwidth-55"])
def test_a_port_outside_0_to_65535_is_refused_as_a_malformed_endpoint_and_nothing_stays_open(binds, unloaded, port):
    """(The review's minor (a).) Before, ``int`` took the port, ``bind`` raised an ``OverflowError`` that no handler
    coded (exit 3), and the command socket bound first stayed open until collected."""
    assert served(unloaded, commands="tcp://127.0.0.1:0", frames=f"tcp://127.0.0.1:{port}") == ["live.endpoint"]
    assert binds.asked == []
    with pytest.raises(ListenError) as refused:
        Server("tcp://127.0.0.1:0", f"tcp://127.0.0.1:{port}")
    assert refused.value.malformed and refused.value.kind == "PUB" and "0 to 65535" in str(refused.value)
    assert len(binds.sockets) == 1 and binds.sockets[0].fileno() == -1  # the command socket, closed


def test_a_port_in_non_ascii_digits_is_no_port_though_int_would_read_it():
    """``str.isdecimal`` is true of every script's digits and ``int`` reads them (``int("\u0663") == 3``): the endpoint
    ``tcp://127.0.0.1:\u0663`` would have bound port 3. Control: the same digits in ASCII are a port."""
    assert int("\u0663") == 3 and parse_endpoint("tcp://127.0.0.1:3") == ("127.0.0.1", 3)
    for port in ("\u0663", "\uff15\uff15", "\u0661\u0662\u0663"):
        with pytest.raises(ValueError, match="not a whole number from 0 to 65535"):
            parse_endpoint(f"tcp://127.0.0.1:{port}")


def test_the_server_refuses_an_address_off_loopback_of_its_own_unless_allowed(binds):
    """(Minor (b).) The transport's ``Server`` called directly, as no code of MIME's but ``mime serve`` does: off
    loopback it binds nothing; allowed, it binds (made loopback by the recorder)."""
    for commands in ("tcp://0.0.0.0:0", "tcp://lab-host:0", "tcp://[::]:0"):
        with pytest.raises(ListenError) as refused:
            Server(commands, "tcp://127.0.0.1:0")
        assert refused.value.not_loopback and binds.asked == []
    allowed = Server("tcp://0.0.0.0:0", "tcp://127.0.0.1:0", allow_remote=True)
    allowed.close()
    assert binds.asked == ["0.0.0.0", "127.0.0.1"]


def test_localhost_is_resolved_once_and_the_address_it_resolved_to_is_bound(binds, unloaded, monkeypatch):
    """(Minor (c).) A resolver that answers loopback for the check and elsewhere after: the server binds the address it
    checked, and ``mime serve``, which checks before the server does, refuses where the server's own look finds it
    elsewhere. Before, the bind resolved ``localhost`` a second time itself."""
    answers: list[list[str]] = []

    def getaddrinfo(host, *_):
        address = answers.pop(0) if answers else ["192.0.2.7"]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (each, 0)) for each in address]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    answers[:] = [["127.0.0.1"]]
    Server("tcp://localhost:0", "tcp://127.0.0.1:0").close()
    assert binds.asked == ["127.0.0.1", "127.0.0.1"] and answers == []  # one resolution, and its address bound
    binds.asked.clear()
    answers[:] = [["127.0.0.1"]]  # the session's own check, then the server's look, which finds it elsewhere
    assert served(unloaded, commands="tcp://localhost:0", frames="tcp://127.0.0.1:0") == ["live.commands-not-loopback"]
    assert binds.asked == []


@pytest.mark.parametrize("platform, options", [("linux", ("SO_REUSEADDR",)), ("darwin", ()), ("freebsd14", ()),
                                               ("openbsd7", ()), ("netbsd10", ()), ("win32", ())])
def test_only_linux_listeners_set_so_reuseaddr(platform, options):
    """(Item 5.) On macOS and the BSDs ``SO_REUSEADDR`` lets a second server bind a specific address on a port a
    wildcard listener holds; on Windows it lets a second take the very address. Before, every platform but Windows set it."""
    assert listener_options(platform) == options


def reuse_address(sock: socket.socket) -> bool:
    return sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) != 0


def test_a_listener_on_this_platform_sets_what_its_policy_says_and_a_planted_reuse_is_seen(monkeypatch):
    """The listeners of a real server, on the platform the test runs on (the platforms workflow runs it on macOS):
    ``SO_REUSEADDR`` set exactly where the policy names it. The control: the option as it was before, planted, is seen
    on every platform where it differs from the policy (macOS and the BSDs), and the policy's own reading is the same
    socket call."""
    server = Server("tcp://127.0.0.1:0", "tcp://127.0.0.1:0")
    try:
        held = [listener for listener, _ in server.listeners.values()]
        assert [reuse_address(listener) for listener in held] == \
            ["SO_REUSEADDR" in listener_options(sys.platform)] * 2
    finally:
        server.close()
    monkeypatch.setattr(_zmtp, "listener_options", lambda platform: () if platform == "win32" else ("SO_REUSEADDR",))
    planted = Server("tcp://127.0.0.1:0", "tcp://127.0.0.1:0")
    try:
        seen = [reuse_address(listener) for listener, _ in planted.listeners.values()]
    finally:
        planted.close()
    assert seen == [sys.platform != "win32"] * 2


def own_run_folder(at: Path) -> Path:
    """A folder that reads as MIME's own run folder: a run's record and nothing else."""
    at.mkdir()
    (at / "run.json").write_text(json.dumps({"steps": 0, "duration": 0.0, "environment": {}}), encoding="utf-8")
    return at


def test_serve_takes_mime_runs_rule_for_its_run_folder_before_anything_binds(binds, unloaded, tmp_path):
    """(Item 1.) ``--out`` was written over whatever it held: the session deleted its ``experiment/`` and truncated its
    ``commands.jsonl``. Now a folder of the user's, an earlier run's folder and a file are refused
    (``run.out-not-empty``) before anything binds; ``--force`` writes into the first two with a warning naming what they hold (the session then
    gets as far as its load here); and a folder that holds a command log is refused with ``--force`` too
    (``live.log-exists``), the log untouched."""
    mine = tmp_path / "mine"
    (mine / "experiment").mkdir(parents=True)
    (mine / "experiment" / "notes.txt").write_text("mine\n", encoding="utf-8")
    earlier = own_run_folder(tmp_path / "earlier")
    logged = own_run_folder(tmp_path / "logged")
    (logged / "commands.jsonl").write_text('{"format": "mime.commands/1"}\n', encoding="utf-8")
    afile = tmp_path / "afile"
    afile.write_text("x", encoding="utf-8")
    ports = {"commands": "tcp://127.0.0.1:0", "frames": "tcp://127.0.0.1:0"}
    saved_at = unloaded / "saved"  # beside the run folders, not holding them
    for out in (mine, earlier, logged, afile):
        assert served(saved_at, out=out, **ports) == ["run.out-not-empty"], out.name
    assert binds.asked == []
    for out in (mine, earlier):
        warned, *rest = served(saved_at, out=out, force=True, **ports)
        assert rest == ["loaded"] and warned.startswith(f"mime serve: warning: --force: {out} holds "), warned
    assert served(saved_at, out=logged, force=True, **ports) == ["live.log-exists"]
    assert served(saved_at, out=afile, force=True, **ports) == ["run.out-not-empty"]
    assert (mine / "experiment" / "notes.txt").read_text(encoding="utf-8") == "mine\n"
    assert (logged / "commands.jsonl").read_text(encoding="utf-8") == '{"format": "mime.commands/1"}\n'


def test_the_cli_passes_force_and_the_record_budget_and_refuses_them_by_their_codes(unloaded, tmp_path, capsys):
    """``mime serve --force`` into a folder holding a command log, and ``--record-budget 0``: each refused by its code
    before anything binds or loads (exit 2)."""
    from mime.cli import main as mime_main

    logged = own_run_folder(tmp_path / "logged")
    (logged / "commands.jsonl").write_text("{}\n", encoding="utf-8")
    ports = ["--commands", "tcp://127.0.0.1:0", "--frames", "tcp://127.0.0.1:0"]
    assert mime_main(["serve", str(unloaded / "saved"), "--out", str(logged), "--force", *ports]) == 2
    assert "live.log-exists" in capsys.readouterr().err
    assert mime_main(["serve", str(unloaded / "saved"), "--record-budget", "0", *ports]) == 2
    assert "live.record-budget" in capsys.readouterr().err
