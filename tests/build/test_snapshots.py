"""The toy experiments' graphs, and script A's body side, as committed golden text. A change to the canonical order,
the builders' flags, the groups' settings, a rate or the step an edge reads shows here as a diff to review.

To accept a reviewed change, run once with ``MIME_REGENERATE_GOLDEN=1``: the files are rewritten and the test fails,
so a regeneration never passes silently.
"""

import os
from pathlib import Path

import pytest
from script_a import script_a

from mime.testing import graph_snapshot
from mime.testing.toys import coupled_toy, full_toy

GOLDEN = Path(__file__).parent / "golden"


@pytest.mark.parametrize("make", [coupled_toy, full_toy, script_a], ids=lambda make: make.__name__)
def test_the_toys_graph_matches_its_golden_text(make):
    snapshot = graph_snapshot(make().build())
    path = GOLDEN / f"{make.__name__}.txt"
    if os.environ.get("MIME_REGENERATE_GOLDEN") == "1":
        path.write_text(snapshot)
        pytest.fail(f"rewrote {path.name}; review the diff and rerun without MIME_REGENERATE_GOLDEN")
    assert snapshot == path.read_text()
