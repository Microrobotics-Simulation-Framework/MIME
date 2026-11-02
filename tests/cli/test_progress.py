"""What a terminal shows of a run in progress, as plain data (``mime.cli._progress``): the arithmetic of the speed and
the time remaining, and the words, tested with no terminal. The honest parts are held here: the estimate leaves the
first chunk out (it paid for compiling) and says nothing until it has a speed; a flag is "by" a step, the end of the
chunk it was first seen at; the last chunk is not announced."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mime.cli._progress import RunState, duration_text, simulated_text


@pytest.mark.parametrize(("seconds", "text"), [
    (0.0, "0 ms"), (0.85, "850 ms"), (4.24, "4.2 s"), (12.4, "12 s"), (59.6, "1 min"), (65.0, "1 min 5 s"),
    (3600.0, "1 h"), (7800.0, "2 h 10 min"),
])
def test_a_span_of_time_is_written_as_a_person_reads_it(seconds, text):
    assert duration_text(seconds) == text


@pytest.mark.parametrize(("seconds", "text"), [(0.0, "0 s"), (2.5, "2.5 s"), (0.041, "41 ms"), (8e-4, "800 us"),
                                              (3000.0, "3,000 s"), (9375.0, "9,375 s")])
def test_a_simulated_time_keeps_its_digits_few(seconds, text):
    assert simulated_text(seconds) == text


def state(chunks, *, asked=1000, dt=0.001, started=0.0, raised=None):
    held = RunState(asked, dt, started)
    for index, (steps, now) in enumerate(chunks):
        held.chunk(steps, now, (raised or {}).get(index))
    return held


def test_there_is_no_speed_after_one_chunk_because_it_compiled():
    held = state([(100, 30.0)])
    assert held.steps_per_second() is None and held.remaining(31.0) is None
    assert "after a chunk that did not compile" in held.estimate_line(31.0)


def test_the_speed_leaves_the_first_chunk_out_and_the_remaining_time_follows_from_it():
    """30 s to compile the first 100 steps, then 100 steps a second: 700 left is 7 s, less the half second since the last chunk ended."""
    held = state([(100, 30.0), (200, 31.0), (300, 32.0)])
    assert held.steps_per_second() == pytest.approx(100.0)
    assert held.remaining(32.5) == pytest.approx(6.5)
    assert "about 6.5 s left (an estimate)" in held.estimate_line(32.5)
    # the control: counting the first chunk's compile in the speed would call it 10 steps a second and 70 s
    assert (1000 - 300) / (300 / 32.0) > 70.0


@given(st.integers(min_value=2, max_value=60), st.integers(min_value=1, max_value=60),
       st.floats(min_value=0.5, max_value=20.0), st.floats(min_value=0.0, max_value=300.0))
def test_a_steady_run_is_estimated_exactly_whatever_its_compile_took(done, left, per_chunk, compile_seconds):
    size = 10
    held = state([(size * (k + 1), compile_seconds + per_chunk * k) for k in range(done)], asked=(done + left) * size)
    assert held.remaining(held.last_chunk[0]) == pytest.approx(left * per_chunk, rel=1e-9)


def test_the_estimate_never_goes_below_zero():
    held = state([(100, 1.0), (200, 2.0)])
    assert held.remaining(500.0) == 0.0


def test_the_last_chunk_is_said_to_be_under_way_and_the_folder_to_be_written_after_it():
    held = state([(100, 1.0), (200, 2.0), (300, 3.0)], asked=400)
    assert held.last_chunk_under_way and "finishing the last chunk, then writing the run folder" in held.estimate_line(3.5)
    assert not state([(100, 1.0), (200, 2.0)], asked=400).last_chunk_under_way


def test_a_flag_is_raised_by_the_end_of_the_chunk_it_was_first_seen_at_and_never_at_a_step():
    held = state([(100, 1.0), (200, 2.0), (300, 3.0)], raised={1: ["swim.slip"], 2: ["swim.slip", "drag.re"]})
    assert held.flags == {"swim.slip": 200, "drag.re": 300}
    text = held.flags_text()
    assert "swim.slip by step 200" in text and "drag.re by step 300" in text and " at step" not in text


def test_no_flag_yet_says_what_the_look_can_and_cannot_see():
    assert "seen at chunk ends" in state([(100, 1.0)]).flags_text()


def test_the_head_gives_steps_and_simulated_time_and_the_line_says_how_to_stop():
    held = state([(250, 1.0)], asked=1000, dt=0.002)
    assert held.head() == " 25%  step 250 of 1,000  ·  500 ms of 2 s simulated"
    assert held.flags_line().endswith("Ctrl+C to stop")


def test_a_run_of_two_chunk_lengths_has_no_speed_until_a_chunk_that_did_not_compile_has_run():
    """1009 steps are 17, then 16 sixty-two times (``automatic_chunks``): the second length is a second compile, 20 s here
    where a chunk takes 0.25 s. The speed is measured from the end of the last chunk that compiled, so that compile is in
    neither the speed nor the estimate; the control is the old rule, from the first chunk's end, which reads 1.6 steps/s."""
    chunks = [(17, 5.0), (33, 25.0), (49, 25.25), (65, 25.5)]
    held = state(chunks[:2], asked=1009, dt=0.001)
    assert held.steps_per_second() is None and "after a chunk that did not compile" in held.estimate_line(25.0)
    held = state(chunks[:3], asked=1009, dt=0.001)
    assert held.steps_per_second() == pytest.approx(64.0)
    assert held.remaining(25.25) == pytest.approx((1009 - 49) / 64.0)
    old = (49 - 17) / (25.25 - 5.0)
    assert old < 2.0 and held.steps_per_second() > 30 * old, "the control: counting the second compile reads far too slow"


@given(st.integers(min_value=17, max_value=40), st.integers(min_value=10, max_value=16), st.integers(min_value=3, max_value=60),
       st.integers(min_value=1, max_value=60), st.floats(min_value=0.5, max_value=20.0),
       st.floats(min_value=0.0, max_value=300.0), st.floats(min_value=0.0, max_value=300.0))
def test_two_lengths_are_estimated_exactly_whatever_both_compiles_took(first, size, done, left, per_chunk, compile_first,
                                                                       compile_second):
    chunks = [(first, compile_first), (first + size, compile_first + compile_second)]
    for k in range(2, done):
        chunks.append((first + size * (k), chunks[1][1] + per_chunk * (k - 1)))
    held = state(chunks, asked=chunks[-1][0] + left * size)
    assert held.remaining(held.last_chunk[0]) == pytest.approx(left * per_chunk, rel=1e-9)
