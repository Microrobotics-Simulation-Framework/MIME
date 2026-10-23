"""The node cases the node walk requires: one per registered entry, saying how its nodes are verified (the options
MADDENING's ``verify_node`` draws with, and a refinement ladder for each order a node declares), or why it builds no
node. An entry registered without one fails the walk."""

import os

from mime.testing import NodeCase

NODE_CASES: dict[tuple[str, str, str], NodeCase] = {
    ("dynamics", "toy.body", "0.1.0"): NodeCase(),
    ("dynamics", "toy.anchored_body", "0.1.0"): NodeCase(),
    ("dynamics", "toy.relaxation", "0.1.0"): NodeCase(),
    ("field provider", "toy.field", "0.1.0"): NodeCase(),
    ("interaction", "toy.spring", "0.1.0"): NodeCase(),
    ("interaction", "toy.drag", "0.2.0"): NodeCase(),
    ("interaction", "toy.lift", "0.1.0"): NodeCase(),
    ("interaction", "toy.hold", "0.1.0"): NodeCase(),
    ("interaction", "toy.couple", "0.1.0"): NodeCase(
        no_node="a registration with the medium's dynamics: the medium's node reads the body and loads it"),
    ("interaction", "toy.twist", "0.1.0"): NodeCase(
        no_node="refused at plan time: it delivers into a reserved interface"),
    ("observation", "toy.recorder", "0.1.0"): NodeCase(),
    ("interaction", "labx.nudge", "0.1.0"): NodeCase(),
}

if os.environ.get("MIME_CI_PLANT_FAILURE") == "1":
    # The control that the walk fails CI when an entry's nodes have no case: the planted run drops toy.anchored_body's,
    # as cases.py drops the only round-trip case that uses it.
    del NODE_CASES[("dynamics", "toy.anchored_body", "0.1.0")]
