"""Bodies sharing a medium (design #204 item 4, C1): on drag 0.4.0, kept, each body's drag is computed alone, so a second
body in the same medium changes neither's resistance, and the build says so on each one's drag; on 0.5.0 the mrs bodies
of one medium are solved as one joint system, the medium's near field, so a second body changes the first's coupled
resistance, and the build says that instead. The controls: a body alone in its medium has no such line, on either
version, and keeps its own node (0.4.0's) on 0.5.0.

0.4.0's equality states its validity line and cannot fail by construction (the two bodies' nodes share no state); what
can fail is the report's line, which the control checks. 0.5.0's inequality is the near field's coupling at its
operating point: the pill's 0.33 mm from the ball moves the ball's resistance by more than its rounding by orders."""

import numpy as np
from mrs_cases import BALL, PILL, free_space, resistance_at, uncoupled

from mime.interactions import Drag
from mime.nodes.hydro.mrs import MRSDragNode
from mime.nodes.hydro.near_field import JointDragNode
from mime.units import mm

AT_REST = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def law():
    return Drag.MRS(spacing=mm(0.4), wall_kind="free_space")


def test_on_the_kept_drag_a_second_body_changes_neither_drag_and_the_build_says_they_share_it():
    """The ball's resistance with the pill beside it and alone, bitwise, on drag 0.4.0: a statement of its validity
    line, which cannot fail by construction (the two bodies' nodes share no state); the report's line can, and its
    control is below."""
    together = free_space("together", {"ball": BALL, "pill": PILL}, uncoupled(mm(0.4))).build()
    alone = free_space("alone", {"ball": BALL}, uncoupled(mm(0.4))).build()
    lines = together.report.checked
    print("\n" + "\n".join(line for owner in sorted(lines) for line in lines[owner]))
    for owner in ("drag:ball", "drag:pill"):
        assert any("share the medium 'water'" in line and "computed alone" in line for line in lines[owner])
    assert np.array_equal(resistance_at(together, "ball", AT_REST), resistance_at(alone, "ball", AT_REST))


def test_on_drag_0_5_0_two_mrs_bodies_in_one_medium_are_one_joint_node_and_the_build_says_so():
    together = free_space("near", {"ball": BALL, "pill": PILL}, law()).build()
    names = sorted(together.graph.node_names)
    assert "near_field:water" in names and "drag:ball" not in names and "drag:pill" not in names
    assert isinstance(together.graph.get_node("near_field:water"), JointDragNode)
    lines = together.report.checked
    print("\n" + "\n".join(line for owner in sorted(lines) for line in lines[owner]))
    for owner, other in (("drag:ball", "'pill'"), ("drag:pill", "'ball'")):
        assert any("solved jointly with" in line and other in line for line in lines[owner])


def test_control_a_body_alone_in_its_medium_has_no_shared_medium_line_and_keeps_its_own_node():
    for drag in (uncoupled(mm(0.4)), law()):
        alone = free_space("alone", {"ball": BALL}, drag).build()
        assert not any("share the medium" in line or "solved jointly" in line
                       for lines in alone.report.checked.values() for line in lines)
        assert isinstance(alone.graph.get_node("drag:ball"), MRSDragNode)
