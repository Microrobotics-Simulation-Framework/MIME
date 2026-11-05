"""The live tests' servers and clients, made diagnosable: a client that cannot finish its handshake fails with what the
server was doing, not with a bare ``ConnectionResetError``.

``mime serve`` is started through :data:`SERVE`, ``mime serve`` under a few lines that log the server's accepts, its
closes (who closed, of what age, with how many bytes the kernel still held unread for the peer) and the stalls of its
loop, to ``<ready file>.serverlog``, and a thread's raise (passed on to the interpreter's own hook, so its traceback still
reaches stderr), and nothing else: the server's behaviour is its own. :func:`client` opens a ``LiveClient`` of a server
started so; when the connection is refused or reset it raises an ``AssertionError`` holding the server process's
``poll()``, what is readable of its standard output and error, and the tail of that log, which is what a one-in-forty
reset of a new client's handshake (CI, 2026-11-05, never reproduced locally) needed in order to name its mechanism.

The control (``tests/live/test_serve_diagnostics.py``) stalls the server's loop past a lowered handshake limit with the
test-only hook :data:`STALL_ENVIRONMENT`, which makes the server close the new peer unread, and checks the message."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mime.live import LiveClient

#: Names ``<limit>:<stall>`` in seconds: the server lowers its handshake limit to ``limit`` and sleeps ``stall`` after
#: each accept, so that it closes the peer it just accepted before reading a byte from it (a loop stalled past the
#: limit, as a starved runner stalls it). A test-only hook of the wrapper, read by no production code.
STALL_ENVIRONMENT = "MIME_TEST_HANDSHAKE_STALL"

#: The wrapper's code: it reads ``--ready-file`` from its own arguments, logs beside it, and leaves the server as it is.
WATCH = '''
import atexit, os, sys, threading, time, traceback
import mime.live._zmtp as zmtp

_ready = sys.argv[sys.argv.index("--ready-file") + 1]
_log = open(_ready + ".serverlog", "a", buffering=1)
_t0 = time.monotonic()


def log(*words):
    _log.write(f"{time.monotonic() - _t0:9.3f} " + " ".join(str(word) for word in words) + "\\n")


def _unread(peer):
    """The bytes the kernel holds for the peer's socket, which the server has not read (what a close turns into a
    reset), or "unknown" where the platform cannot say: never 0 for a count it could not read."""
    try:
        import fcntl, struct, termios
        return struct.unpack("i", fcntl.ioctl(peer.sock.fileno(), termios.FIONREAD, b"\\0\\0\\0\\0"))[0]
    except Exception:
        return "unknown"


def _buffered(peer):
    return len(getattr(getattr(peer, "reader", None), "buffer", b""))


_close = zmtp.Server._close


def closing(self, peer):
    where = "<".join(frame.name for frame in traceback.extract_stack(limit=4)[:-1])
    log("close", peer.kind, getattr(peer, "address", "?"), "ready", peer.ready, "age",
        round(time.monotonic() - getattr(peer, "opened", time.monotonic()), 3), "unread", _unread(peer),
        "buffered", _buffered(peer), "from", where)
    _close(self, peer)


_accept = zmtp.Server._accept


def accepting(self, listener, kind):
    before = set(self.peers)
    try:
        _accept(self, listener, kind)
    except BaseException:
        log("accept RAISED", kind, traceback.format_exc().replace("\\n", " | "))
        raise
    log("accept", kind, "peers", len(self.peers), "new", len(set(self.peers) - before))


_poll = zmtp.Server.poll
_left = [None]


def polling(self, timeout):
    entered = time.monotonic()
    if _left[0] is not None and entered - _left[0] > 0.5:
        log("STALL between polls", round(entered - _left[0], 3))
    try:
        found = _poll(self, timeout)
    except BaseException:
        log("poll RAISED", traceback.format_exc().replace("\\n", " | "))
        raise
    if time.monotonic() - entered > timeout + 0.5:
        log("STALL inside poll", round(time.monotonic() - entered - timeout, 3))
    _left[0] = time.monotonic()
    return found


zmtp.Server._close, zmtp.Server._accept, zmtp.Server.poll = closing, accepting, polling

_stall = os.environ.get("%(stall)s")
if _stall:
    _limit, _seconds = (float(each) for each in _stall.split(":"))
    zmtp.HANDSHAKE_SECONDS = _limit
    _accepted = zmtp.Server._accept

    def stalled(self, listener, kind):
        _accepted(self, listener, kind)
        print(f"test hook: the loop sleeps {_seconds} s after an accept, past its {_limit} s handshake limit",
              file=sys.stderr, flush=True)
        time.sleep(_seconds)

    zmtp.Server._accept = stalled

atexit.register(lambda: log("process exit"))
_thread_hook = threading.excepthook


def thread_raised(args):
    log("thread raised", args.exc_type.__name__, args.exc_value, "|",
        "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)).replace("\\n", " | "))
    _thread_hook(args)


threading.excepthook = thread_raised
''' % {"stall": STALL_ENVIRONMENT}

#: ``mime`` with the arguments after ``-c``, under :data:`WATCH`.
SERVE = WATCH + "\nfrom mime.cli import main\nraise SystemExit(main(sys.argv[1:]))\n"

#: The server process of each ready file started through :func:`watched`.
_started: dict[Path, subprocess.Popen[Any]] = {}


def serve_command(folder: Path, ready: Path, out: Path, *extra: str) -> list[str]:
    """``mime serve`` of ``folder`` on two free loopback ports, under :data:`SERVE`."""
    return [sys.executable, "-c", SERVE, "serve", str(folder), "--commands", "tcp://127.0.0.1:0", "--frames",
            "tcp://127.0.0.1:0", "--ready-file", str(ready), "--out", str(out), *extra]


def watched(launch: Callable[..., subprocess.Popen[Any]], command: Sequence[str], ready: Path,
            **options: Any) -> subprocess.Popen[Any]:
    """``launch(command)`` with its standard output and error piped as text, remembered as the server of ``ready``
    (``command`` is a server under :data:`WATCH`, writing ``ready`` as its ready file)."""
    options.setdefault("stdout", subprocess.PIPE)
    options.setdefault("stderr", subprocess.PIPE)
    options.setdefault("text", True)
    process = launch(command, **options)
    register(ready, process)
    return process


def register(ready: Path, process: subprocess.Popen[Any]) -> None:
    """Remember ``process`` as the server of ``ready``, for :func:`client`'s message."""
    _started[ready.resolve()] = process


def client(ready: Path, timeout: float = 30.0) -> LiveClient:
    """A ``LiveClient`` of the server that wrote ``ready``; one that cannot finish its handshake raises an
    ``AssertionError`` holding the server's state (:func:`diagnosis`)."""
    endpoints = json.loads(ready.read_text())
    try:
        return LiveClient(endpoints["commands"], endpoints["frames"], timeout=timeout)
    except ConnectionError as error:
        raise AssertionError(diagnosis(ready, error)) from error


def readable(stream: Any) -> str:
    """What is readable now of a pipe, without waiting for more: a pipe still open and quiet reads as empty."""
    if stream is None or not hasattr(stream, "fileno"):
        return "(not a pipe)"
    try:
        descriptor = stream.fileno()
        os.set_blocking(descriptor, False)
        held = b""
        while True:
            try:
                more = os.read(descriptor, 1 << 16)
            except BlockingIOError:
                break
            if not more:
                break
            held += more
        return held.decode("utf-8", "replace") or "(nothing)"
    except (OSError, ValueError, AttributeError) as error:
        return f"(unreadable: {error!r})"


def diagnosis(ready: Path, error: BaseException, *, lines: int = 80) -> str:
    """What a failed handshake needs to name its mechanism: the error, the server process's ``poll()`` (``None`` while
    it runs), what is readable of its standard output and error, and the last ``lines`` of its log."""
    process = _started.get(ready.resolve())
    parts = [f"a client could not finish its handshake with the server of {ready.name}: {error!r}"]
    if process is None:
        parts.append("the server was not started through tests/live_servers.py: no process to report")
    else:
        parts += [f"server process poll(): {process.poll()!r} (None: still running)",
                  f"server stdout so far:\n{readable(process.stdout)}", f"server stderr so far:\n{readable(process.stderr)}"]
    log = Path(str(ready) + ".serverlog")
    held = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
    parts.append(f"the server's log (accepts, closes, loop stalls; last {lines} of {len(held)} lines):\n"
                 + ("\n".join(held[-lines:]) if held else "(none written)"))
    return "\n".join(parts)


__all__ = ["SERVE", "STALL_ENVIRONMENT", "WATCH", "client", "diagnosis", "register", "serve_command", "watched"]
