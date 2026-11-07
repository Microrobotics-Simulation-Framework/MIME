"""The plan's memory estimate: the device bytes of a batch measured at its own size, and a figure a member from the line
through two batch sizes, given only where a third lies within a fifth of a member's bytes of it. The control is the
MRS screw in a tube as XLA assigned its step's temporaries at one to sixteen members (measured once, on one jaxlib, for
the sweep's review): from three members up they grow by about 12.3 MiB a member, and from one to two by 2.7, so the line
through one and two
members, which the first estimate used, misses the third size by far more than the check allows, and the line through
four and eight holds at three and at sixteen."""

from __future__ import annotations

import pytest
from script_a import script_a

from mime import Study
from mime.study import grid
from mime.study._study import LINE_MISS, Memory, line_through, measured_sizes
from mime.units import Hz, s

MiB = 2**20
SCREW = {1: 25.44, 2: 28.12, 3: 39.80, 4: 52.07, 8: 101.15, 16: 199.32}  # MiB, XLA's temporaries of the screw's step


def _at(*sizes: int) -> tuple[tuple[int, int], ...]:
    return tuple((size, round(SCREW[size] * MiB)) for size in sizes)


def test_the_batch_is_always_among_the_sizes_measured_and_the_line_is_drawn_where_xla_grows_by_a_member():
    for batch in range(1, 70):
        line, third = measured_sizes(batch)
        assert batch in (*line, *([third] if third is not None else [])), batch
        assert third is None or third not in line, batch
    assert measured_sizes(1) == ((1,), None) and measured_sizes(8) == ((4, 8), 3) and measured_sizes(64) == ((4, 8), 64)
    assert measured_sizes(4) == ((2, 4), 3) and measured_sizes(2) == ((1, 2), 3)


def test_the_line_through_four_and_eight_holds_on_the_screw_and_gives_its_member():
    for third in (3, 16):
        fixed, each, miss = line_through(_at(4, 8, third))
        assert miss <= LINE_MISS, (third, miss)
        assert each / MiB == pytest.approx(12.27, abs=0.01)
    memory = Memory(16, 16, 0, _at(4, 8, 16), *line_through(_at(4, 8, 16)))
    assert memory.holds and memory.device == round(SCREW[16] * MiB) and "a member on the line" in memory.text()


def test_the_control_the_line_through_one_and_two_members_misses_the_third_and_gives_no_figure_a_member():
    fixed, each, miss = line_through(_at(1, 2, 3))
    assert each / MiB == pytest.approx(2.68, abs=0.01) and miss > 3 * LINE_MISS, miss
    memory = Memory(2, 2, 0, _at(1, 2, 3), fixed, each, miss)
    assert not memory.holds and memory.device == round(SCREW[2] * MiB)
    assert "no figure a member" in memory.text() and "a member on the line" not in memory.text()


def test_a_plan_measures_its_batch_at_its_size():
    exp = script_a()
    field = exp.handle("field")
    plan = Study(exp).sweep(grid({field.commands.frequency: [Hz(f) for f in (10.0, 20.0, 30.0, 40.0)]}),
                            duration=s(0.05)).plan()
    memory = plan.groups[0].memory
    assert memory is not None and [size for size, _ in memory.at] == [2, 4, 3]
    assert memory.device == dict(memory.at)[4] and memory.total == memory.device + 4 * memory.host
    assert "memory about" in plan.text() and "memory" not in plan.text(memory=False).split("\n", 1)[1]
