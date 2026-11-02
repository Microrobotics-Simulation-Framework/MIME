"""Each body's drag is computed alone (the drag's validity line, both rungs; design #204 item 4): a second body in the
same medium changes neither's resistance, since no hydrodynamic interaction between bodies is built in MIME 0.3.0, and
the build reports the bodies that share a medium on each one's drag. The control: a body alone in its medium has no
such line."""

import numpy as np
from mrs_cases import BALL, PILL, free_space, resistance_at

from mime.interactions import Drag
from mime.units import mm

AT_REST = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def law():
    return Drag.MRS(spacing=mm(0.4), wall_kind="free_space")


def test_a_second_body_in_the_medium_changes_neither_drag_and_the_build_says_they_share_it():
    together = free_space("together", {"ball": BALL, "pill": PILL}, law()).build()
    alone = free_space("alone", {"ball": BALL}, law()).build()
    lines = together.report.checked
    print("\n" + "\n".join(line for owner in sorted(lines) for line in lines[owner]))
    for owner in ("drag:ball", "drag:pill"):
        assert any("share the medium 'water'" in line and "computed alone" in line for line in lines[owner])
    assert np.array_equal(resistance_at(together, "ball", AT_REST), resistance_at(alone, "ball", AT_REST))


def test_control_a_body_alone_in_its_medium_has_no_shared_medium_line():
    alone = free_space("alone", {"ball": BALL}, law()).build()
    assert not any("share the medium" in line for lines in alone.report.checked.values() for line in lines)
