"""Where ``mime serve`` binds. Nothing authenticates a client, so an address that is not loopback is refused before
anything binds, the command socket as ``live.commands-not-loopback`` and the frame socket as
``live.frames-not-loopback``, in every form it can be written (the wildcards ``0.0.0.0``, ``*``, ``::`` and an empty host,
an address of another machine, a host name, and ``localhost`` where it resolves elsewhere); ``--allow-remote-commands``
binds it all the same and warns, naming it. A second ``mime serve`` on the ports of one running is refused, naming the
port it could not take (``live.address-in-use``), and leaves the first serving and nothing of its own listening.

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
from mime.live._zmtp import is_loopback

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
    assert binds.asked == ["127.0.0.1", "localhost"]
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
