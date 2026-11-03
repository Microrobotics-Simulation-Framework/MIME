"""``BoundedWriter``, the stream ``mime serve`` writes to a terminal through, on a pseudo-terminal nobody reads (PR #122's
review, F1): a write there blocks for as long as nothing reads, and the writer's callers must not.

What is held: against a full terminal a write and a flush return in a bounded time and what the terminal does not take
is counted in ``dropped``; once something reads again what was queued arrives, in order, and the next flush waits
again; a terminal that hung up is dropped to, not raised from; a stream with no file descriptor is written to directly.

The controls: a plain stream over the same full terminal blocks (the check is that its write has not returned after a
second, so the full terminal is one that can hold a caller), and a terminal that is read takes every byte in order with
nothing dropped, so the bounded arms differ from it only in the reading."""

import io
import os
import threading
import time

import pytest

from _platform_support import needs_pty
from _terminal_support import fill_unread, is_full
from mime.cli._bounded import BoundedWriter

if os.name != "nt":
    import pty

pytestmark = needs_pty
#: A guard: a bounded call that has not returned in this long has blocked.
BOUND = 10.0


class Far:
    """A pseudo-terminal: ``stream`` is its near side, ``master`` the end a reader reads."""

    def __init__(self) -> None:
        self.master, slave = pty.openpty()
        self.path = os.ttyname(slave)
        self.stream = os.fdopen(slave, "w", buffering=1, encoding="utf-8", newline="")

    def read_all(self, wait: float = 0.5) -> bytes:
        import select

        held = b""
        while select.select([self.master], [], [], wait)[0]:
            try:
                more = os.read(self.master, 65536)
            except OSError:
                break
            if not more:
                break
            held += more
            wait = 0.1
        return held

    def close(self) -> None:
        for each in (self.stream.close, lambda: os.close(self.master)):
            try:
                each()
            except OSError:
                pass


@pytest.fixture
def far():
    held = Far()
    yield held
    held.close()


def timed(call) -> float:
    started = time.monotonic()
    call()
    return time.monotonic() - started


def test_against_a_full_terminal_a_write_and_a_flush_return_and_what_is_not_taken_is_counted(far):
    fill_unread(far.path)
    assert is_full(far.path)
    writer = BoundedWriter(far.stream, wait=0.2)
    elapsed = timed(lambda: [writer.write(f"line {each}\n" * 40) or writer.flush() for each in range(400)])
    assert elapsed < BOUND, "400 writes and flushes against a terminal nobody reads"
    assert writer.dropped > 0, "the terminal took none of it, and past the queue's limit it was dropped"
    assert not writer.drain(0.0), "and what was queued has not been written"
    assert timed(writer.flush) < 1.0, "a flush once it is behind does not wait again"


def test_the_control_a_plain_stream_over_the_same_full_terminal_blocks_its_writer(far):
    fill_unread(far.path)
    done = threading.Event()

    def write() -> None:
        far.stream.write("a line\n" * 200)
        far.stream.flush()
        done.set()

    threading.Thread(target=write, daemon=True).start()
    assert not done.wait(1.0), "the write is still blocked after a second: the terminal is one that holds a caller"
    deadline = time.monotonic() + BOUND  # read until it finishes, so that the thread does not outlive the test
    while not done.is_set() and time.monotonic() < deadline:
        far.read_all(0.1)
    assert done.is_set()


def test_the_control_a_terminal_that_is_read_takes_every_byte_in_order_and_nothing_is_dropped(far):
    writer = BoundedWriter(far.stream)
    lines = [f"line {each:04d}\n" for each in range(300)]
    taken = bytearray()
    reading = threading.Event()

    def read() -> None:
        reading.set()
        taken.extend(far.read_all(1.0))

    thread = threading.Thread(target=read)
    thread.start()
    reading.wait()
    for line in lines:
        writer.write(line)
        writer.flush()
    thread.join(BOUND)
    assert bytes(taken).decode().replace("\r\n", "\n") == "".join(lines) and writer.dropped == 0


def test_once_something_reads_again_what_was_queued_arrives_in_order_and_the_next_flush_waits_again(far):
    fill_unread(far.path)
    writer = BoundedWriter(far.stream, wait=0.2)
    writer.write("first\n")
    writer.flush()
    writer.write("second\n")
    writer.flush()
    assert not writer.drain(0.0)
    held = far.read_all(0.5)  # the terminal is read: the filler and then what was queued
    assert writer.drain(BOUND)
    held += far.read_all(0.5)
    text = held.decode().replace("\r\n", "\n")
    assert text.endswith("first\nsecond\n") and text.count("x") >= 1024
    assert writer.dropped == 0


def test_a_terminal_that_hung_up_is_dropped_to_and_never_raised_from(far):
    writer = BoundedWriter(far.stream)
    os.close(far.master)
    far.master = os.open(os.devnull, os.O_RDONLY)  # so that the fixture closes something
    writer.write("one\n")
    writer.drain(BOUND)
    writer.write("two\n")
    writer.flush()
    assert writer.dropped >= 1


def test_a_stream_without_a_file_descriptor_is_written_to_directly():
    buffer = io.StringIO()
    writer = BoundedWriter(buffer)
    writer.write("kept\n")
    writer.flush()
    assert buffer.getvalue() == "kept\n" and writer.dropped == 0 and writer.drain(0.0)
