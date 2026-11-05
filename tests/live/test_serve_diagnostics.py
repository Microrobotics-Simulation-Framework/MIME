"""A client whose handshake with a live server fails says what the server was doing (``tests/live_servers.py``).

A one-in-forty ``ConnectionResetError`` in a new client's handshake (CI, 2026-11-05, 60 000 handshakes on a laptop and none
reproduced it) named no mechanism: the test's message held the error and nothing of the server. A client opened by the
live tests' ``client`` now fails with the server process's ``poll()``, its output and the log of its accepts, closes and loop
stalls. Machine-free: the server's loop sleeps, test-only, past a handshake limit lowered to 0.1 s after it accepts a
peer, so the next pass of its loop closes the peer unread (the mechanism the failure's report names first, made to
happen), whatever the load.

The control is the planted fault itself and its message: the server's stderr carries a line the hook writes, its log a
close of an unready peer from ``_housekeeping``, and a client's own error carries neither, so a message that lost the
diagnostics fails these assertions on every run. A healthy server is also checked, so the wrapper that logs does not
refuse a handshake."""

import time
from pathlib import Path

import pytest
from live_cases import saved
from live_servers import STALL_ENVIRONMENT, client, serve_command, watched

HANG = 600


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> Path:
    return saved(tmp_path_factory.mktemp("diagnosed"))


def started(launch, folder: Path, tmp_path: Path, name: str, **options):
    ready = tmp_path / f"{name}.ready.json"
    process = watched(launch, serve_command(folder, ready, tmp_path / f"{name}.out"), ready, **options)
    deadline = time.monotonic() + HANG
    while not ready.exists() or not ready.read_text():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, "mime serve never wrote its ready file"
        time.sleep(0.05)
    return process, ready


def test_a_handshake_reset_by_a_stalled_server_is_reported_with_its_stderr_its_log_and_its_state(
        launch, folder, tmp_path, monkeypatch):
    monkeypatch.setenv(STALL_ENVIRONMENT, "0.1:1.0")
    process, ready = started(launch, folder, tmp_path, "stalled")
    with pytest.raises(AssertionError) as raised:
        client(ready)
    message = str(raised.value)
    cause = raised.value.__cause__
    assert isinstance(cause, ConnectionError), "the fault did not reset or close the handshake"
    # the process: alive, said so
    assert "server process poll(): None (None: still running)" in message
    # its stderr: the line the hook wrote
    assert "test hook: the loop sleeps 1.0 s after an accept, past its 0.1 s handshake limit" in message
    # its log: the accept, and the close of the unready peer by the loop's housekeeping
    assert " accept PUB " in message  # a client's frames connection is the first it makes
    close = next(line for line in message.splitlines() if " close PUB " in line)
    assert "ready False" in close and "_housekeeping" in close, close
    # the control: the diagnostics are what carries these, not the error
    for only_in_the_diagnostics in ("test hook", "housekeeping", "ready False", "poll()"):
        assert only_in_the_diagnostics not in repr(cause), only_in_the_diagnostics
    assert process.poll() is None


def test_a_server_that_has_exited_is_reported_with_its_exit_code(launch, folder, tmp_path):
    process, ready = started(launch, folder, tmp_path, "gone")
    process.kill()
    process.wait(timeout=HANG)
    with pytest.raises(AssertionError) as raised:
        client(ready)
    assert isinstance(raised.value.__cause__, ConnectionError)
    assert f"server process poll(): {process.returncode} (None: still running)" in str(raised.value)
    assert process.returncode is not None


def test_a_healthy_server_under_the_logging_wrapper_is_greeted(launch, folder, tmp_path):
    process, ready = started(launch, folder, tmp_path, "healthy")
    live = client(ready)
    assert live.request("status")["ok"]
    live.close()
    log = Path(str(ready) + ".serverlog").read_text()
    assert " accept REP " in log and " accept PUB " in log
    process.terminate()
    process.communicate(timeout=HANG)
