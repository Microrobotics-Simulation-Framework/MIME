"""The build's held matrix is made by one jitted float64 program, and it is the op-by-op build's, bit for bit.

``MRSDragNode.made_built`` makes the body's discretisation and the free part's matrix (the 82-chunk scan of the tube
demo's ball) from the node's constants. Op by op, every primitive is its own XLA program (255 compiles in the tube demo's
build); jitted, it is one. The held quantity is a constant of every step the graph takes, so a rounding in it would move
every MRS output, and the test is ``np.array_equal`` on everything ``Built`` holds, not a tolerance.

The cases are the bodies the rung serves: a ball, a pill and a screw in free space (the held quantity is the 6 x 6
resistance), the solids that sample their surface (a cylinder, a box, an ellipsoid), a ball in the tube (the held
quantity is the 1014 x 1014 matrix), and a sphere whose written quadrature is under its rule's (the reference
quadrature, whose resistance ``moved`` compares). The build is float64 whatever the run's precision: a float32 process
and a float64 one hold the same bytes (fresh processes, below). That the build's matrix is the jitted program's is one
claim, and the resistance is another: jitted, the sum over the samples (``force_sums``) rounded the tube ball's
resistance differently in every entry (measured), so only the matrix is jitted and the resistance, taken from it, is
made as it was.

**Status of this head:** the equality test fails on every case, by 5e-18 to 5e-16 of the held quantity's largest entry
(the pull request has the table), so the gate stands unmet and nothing is re-pinned or loosened here.

Bits depend on the CPU's thread count; both arms here run in one process, so the comparison is at one setting.

**The control** plants a change that alters the order of summation without changing the maths (the quadrature taken in
chunks of another size) in the jitted arm only. It is deterministic input, so it must move the bits on every run; a case
whose quadrature fits one chunk, where it cannot, is not in the list. **The compile budget** is the point of the change:
the build's compiles in the fresh process (a performance budget, not physics), with the op-by-op arm as the control that
it counts what it should."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
import tube_asset
from mrs_cases import KINDS, SOLIDS, free_space, mrs_free_space, mrs_solids, mrs_tube

from mime.interactions import Drag
from mime.nodes.hydro.mrs import MRSDragNode
from mime.nodes.hydro.mrs._quadrature import Chunks
from mime.scene import Sphere
from mime.units import mm

ROOT = Path(__file__).resolve().parents[2]
#: A sphere whose written quadrature (refinement 2) is under its rule's (4 at the blob 0.5): the node assembles the
#: rule's once more at build, and ``moved`` is how far the resistance moves.
REFERENCE_SPHERE = "reference_sphere"
#: How much more finely the planted chunks cut: the chunk size is the byte budget over this many times the bytes a pair
#: takes, so a quadrature of a few thousand points falls in many chunks.
PLANT_ITEMSIZE = 8 * 1024


def reference_sphere():
    return free_space(REFERENCE_SPHERE, {"ball": Sphere(radius=mm(1.0))},
                      Drag.MRS(spacing=mm(1.5), blob=0.5, quadrature_refinement=2, wall_kind="free_space"))


#: name -> (the experiment, the bodies whose node is checked)
EXPERIMENTS = {
    "free_space": (mrs_free_space, tuple(KINDS)),
    "solids": (mrs_solids, tuple(SOLIDS)),
    "reference": (reference_sphere, ("ball",)),
    "tube": (mrs_tube, ("ball",)),
}
_built: dict[str, object] = {}


def drag_nodes(name: str) -> list[tuple[str, MRSDragNode]]:
    if name == "tube":
        tube_asset.path()  # stored by CI; a missing table fails here by name, with the command that stores it
    if name not in _built:
        _built[name] = EXPERIMENTS[name][0]().build()
    graph = _built[name].graph  # type: ignore[attr-defined]
    return [(f"{name}:{body}", graph.get_node(f"drag:{body}")) for body in EXPERIMENTS[name][1]]


CASES = [pytest.param(name, body, id=f"{name}:{body}") for name, (_, bodies) in EXPERIMENTS.items() for body in bodies]


def node_of(name: str, body: str) -> MRSDragNode:
    node = dict(drag_nodes(name))[f"{name}:{body}"]
    assert isinstance(node, MRSDragNode)
    return node


def equal(a: object, b: object) -> bool:
    return (a is None and b is None) or (a is not None and b is not None
                                         and np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))


@pytest.mark.parametrize(("name", "body"), CASES)
def test_the_jitted_build_is_the_op_by_op_build_bit_for_bit(name, body):
    node = node_of(name, body)
    by_ops, jitted = node.made_built(jitted=False), node.made_built(jitted=True)
    unequal = [field for field in by_ops._fields if not equal(getattr(by_ops, field), getattr(jitted, field))]
    assert not unequal, unequal
    assert jitted.held.dtype == np.float64 and jitted.free.dtype == np.float64
    assert (jitted.moved is not None) == (name == "reference")
    assert jitted.held.shape == ((6, 6) if node._free() else (node.chunks.cells * 3,) * 2)


def test_the_cases_hold_a_free_part_and_a_wall_part():
    """The cases cover both held quantities: the 6 x 6 resistance of free space and the matrix with a wall."""
    kinds = {node._free() for name in ("free_space", "tube") for _, node in drag_nodes(name)}
    assert kinds == {True, False}


@pytest.mark.parametrize(("name", "body"), [pytest.param(n, b, id=f"{n}:{b}") for n, b in
                                            [("free_space", "ball"), ("free_space", "pill"), ("free_space", "screw"),
                                             ("solids", "rod"), ("solids", "brick"), ("solids", "egg"),
                                             ("tube", "ball")]])
def test_control_chunks_of_another_size_move_the_jitted_matrix(name, body):
    """The planted change of the order of summation, in the jitted arm only, must move the held matrix (or the
    resistance) of every case listed: a comparison that cannot see it is no evidence for the cases it passes."""
    node = node_of(name, body)
    by_ops = node.made_built(jitted=False)
    held = node.chunks
    node.__dict__["chunks"] = Chunks.of(node._owners, node.sample_count, itemsize=PLANT_ITEMSIZE)
    try:
        planted = node.made_built(jitted=True)
    finally:
        node.__dict__["chunks"] = held
    assert not equal(by_ops.held, planted.held), "the plant does not move this case: it cannot discriminate"


PRELUDE = """
import hashlib, json, sys
import jax
import jax
import numpy as np
sys.path[:0] = sys.argv[3:]
from mrs_cases import mrs_free_space, mrs_tube
experiment = {"free_space": mrs_free_space, "tube": mrs_tube}[sys.argv[1]]()
"""

#: The held bytes of every drag node, in a fresh process of the precision the environment names.
HELD_HASHES = PRELUDE + """
built = experiment.build()
print(json.dumps({"x64": bool(jax.config.jax_enable_x64), "held": {
    name: [hashlib.sha256(np.ascontiguousarray(node.held_in_body_frame).tobytes()).hexdigest(),
           str(node.held_in_body_frame.dtype)]
    for name, node in sorted((n, built.graph.get_node(n)) for n in built.graph.node_names if n.startswith("drag:"))}}))
"""

#: The backend compiles the build's evaluation of the held quantity takes (the first, cold), in a fresh process.
COMPILES = PRELUDE + """
from jax import monitoring
import mime.nodes.hydro.mrs._node as N
from mime.testing._saved import _compilations
from functools import cached_property
if sys.argv[2] == "op_by_op":
    original = N.MRSDragNode.made_built
    N.MRSDragNode.made_built = lambda self, jitted=True: original(self, jitted=False)
property_ = N.MRSDragNode.__dict__["_built"]
assert isinstance(property_, cached_property)
count = []
function = property_.func
def counted(self):
    with _compilations() as seen:
        out = function(self)
    count.append(len(seen))
    return out
property_.func = counted
experiment.build()
print(json.dumps({"compiles": count}))
"""


def fresh_process(script: str, case: str, how: str, x64: bool = False) -> dict:
    env = {key: value for key, value in os.environ.items() if key not in ("JAX_ENABLE_X64",)}
    env["JAX_PLATFORMS"] = "cpu"
    if x64:
        env["JAX_ENABLE_X64"] = "1"
    done = subprocess.run([sys.executable, "-c", script, case, how, str(ROOT / "tests")], capture_output=True,
                          text=True, check=False, env=env)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


def test_a_float32_process_and_a_float64_one_hold_the_same_bytes():
    """The build is float64 whatever the run's precision: asserted, not assumed."""
    single, double = (fresh_process(HELD_HASHES, "free_space", "-", x64=x64) for x64 in (False, True))
    assert (single["x64"], double["x64"]) == (False, True)
    assert single["held"] == double["held"]
    assert {dtype for _, dtype in single["held"].values()} == {"float64"}
    assert len(single["held"]) == len(KINDS)


#: The compiles the three held quantities of the free-space case take in the first build of a fresh float32 process,
#: matrix and resistance together (measured [59, 15, 15] jitted; the op-by-op form is under "many" below), with room.
COMPILE_BUDGET = 100


def test_the_held_quantity_takes_a_few_compiles():
    jitted = fresh_process(COMPILES, "free_space", "jitted")["compiles"]
    assert len(jitted) == len(KINDS) and sum(jitted) <= COMPILE_BUDGET, jitted


def test_control_the_op_by_op_form_takes_many_compiles():
    """The counter sees what it should: the op-by-op form takes far more than the budget."""
    by_ops = fresh_process(COMPILES, "free_space", "op_by_op")["compiles"]
    assert len(by_ops) == len(KINDS) and sum(by_ops) > 2 * COMPILE_BUDGET, by_ops


def test_the_digest_of_nothing_is_not_the_digest_of_the_held_matrix():
    """A guard on the hashes above: two different bodies hold different bytes (the comparison can fail)."""
    nodes = dict(drag_nodes("free_space"))
    digests = {hashlib.sha256(np.ascontiguousarray(node.held_in_body_frame).tobytes()).hexdigest()
               for node in nodes.values()}
    assert len(digests) == len(KINDS)


def test_the_held_quantity_asked_for_first_inside_a_trace_is_made_there_and_is_the_same():
    """``_built`` is evaluated where it is first asked for, a step's trace included: inside a jit trace the compiled
    program runs on concrete arrays and the node holds NumPy, not a tracer."""
    node = node_of("free_space", "ball")
    before = node.__dict__.pop("_built")
    seen = []

    @jax.jit
    def traced(x):
        seen.append(node._built)
        return x + 1.0

    try:
        traced(1.0)
        after = node.__dict__["_built"]
    finally:
        node.__dict__["_built"] = before
    assert isinstance(seen[0].held, np.ndarray) and np.array_equal(after.held, before.held)
