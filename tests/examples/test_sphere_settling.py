"""``examples/sphere_settling.py``: a steel sphere settling in still water in free space, run as the example runs it.

**What this bounds.** That the boundary-element rung at the example's starting spacing reproduces Stokes's law for a
sphere in free space to the discretisation's known error (the drag at 338 samples is 1.3% above Stokes's, and the
settled speed 1.3% below, measured in ``docs/data/spacing_tables.json``), at the example's Reynolds number, which the
post-run watch reads back as inside the rung's declared range. The agreement is with a closed form that holds at the
example's operating point (Reynolds 0.03), and that is the control for the watch on the larger sphere: at 50 um the
same comparison is made at Reynolds 3.7, where Stokes's law itself does not hold, and the watch says so."""

import importlib.util
from pathlib import Path

import pytest

from mime.results._watch import ABOVE, WITHIN

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("example_sphere_settling", ROOT / "examples" / "sphere_settling.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def small(example):
    return example.run(10.0)


def test_the_starting_size_settles_within_two_per_cent_of_stokess_law_below_it(small):
    result, ratio = small
    assert 0.98 < ratio < 1.0, "the discretisation is the rung's, from below: its drag is above Stokes's"
    assert result.steps == 30 and result.dt == pytest.approx(1e-4)


def test_the_post_run_watch_reads_the_reynolds_number_inside_the_range_and_says_how_much_is_neglected(small):
    (watch,) = small[0].watches
    assert watch.status == WITHIN and 0.02 < watch.reynolds < 0.04 and watch.peak
    assert watch.inertia == pytest.approx(3.0 / 16.0 * watch.translation) and watch.inertia < 0.01
    assert "within" in small[0].text() and "ABOVE" not in small[0].text()


def test_a_larger_sphere_settles_at_a_reynolds_number_the_watch_flags_with_the_size_of_what_the_rung_leaves_out(example):
    result, ratio = example.run(50.0)
    (watch,) = result.watches
    assert watch.status == ABOVE and 3.0 < watch.reynolds < 4.5
    assert watch.inertia == pytest.approx(3.0 / 16.0 * watch.reynolds) and 0.5 < watch.inertia < 0.9
    assert "ABOVE" in result.text() and "3 Re / 16" in result.text()
    assert 0.98 < ratio < 1.0, "the rung still reproduces Stokes's law there: the flag is about the law, not the solve"


def test_the_example_declares_every_value_it_uses_with_its_reason_so_nothing_is_pending(example):
    built = example.sphere_settling(10.0).build()
    assert built.report.profile.pending == (), built.report.pending()
