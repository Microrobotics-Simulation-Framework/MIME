"""The table evaluator against the 0.2 line's wall-table lookup (``cylinder_wall_table.py``, a hand-written clipped
multilinear lookup with a float32 JAX twin), the weakest oracle, on the 0.2 table's own grid. The port's lookup ran
once, in the port's own environment, on a block of a de Jongh wall table (its own knots and values: rho and rho0 near
the wall, the first angles and axial offsets), at the coordinates of every pair of 24 points inside the block, as the
port computed them (``generate_port_table_references.py``; ``references/port_table.npz`` and its provenance:
MIME-port041 at 6485516). ``mime.core.tables`` reads the same block here as a linear table of four axes and a 3 x 3
value. The port's ``mime`` is never imported beside this one.

**Two blocks.** The ``corner`` block is the near-wall end of rho and rho0 (the last eight knots of 40, where the port's
clip began), the first six of 64 angles and the first seven of 128 offsets. The ``axis_far`` block is what that leaves
out: the axis end of rho and rho0 (the first two knots), every angle to pi and every offset to the last, 12.5, two of
its points at exactly the last knot of each (the pair of a point with itself has angle and offset zero, so a block that
holds the far knots holds the first too). Both come from the same generator, which reproduces the corner's file bit
for bit, in the port's environment.

**Inside the range where the port did not clip.** Every pair lies inside its block on every axis (the generator
checks it), where the port's clip of the cell index and of the weight changes nothing; MIME reads every pair in
range.

**Declared before the first run.**

- float64: MIME's read equals the port's NumPy lookup within 64 float64 roundings of the block's largest value. Both
  compute the same multilinear interpolation by different sequences of operations (the port sums sixteen corners,
  each weighted by a product of four factors; MIME interpolates one axis after another), so no difference beyond
  rounding is allowed.
- float32: MIME's read in float32 equals the port's JAX twin within 128 float32 roundings of the block's largest
  value: the twin's own sum (about twenty), its rotation to Cartesian and the generator's inversion of it (about ten),
  and MIME's read (about a dozen).

The controls, each run on the fixed reference so that it fails every run, with the share of pairs that see it
printed: a cell one below the right one, and one interior knot moved by a part in a million."""

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mime.core.tables._evaluate as evaluate_module
from mime.core import switch
from mime.core.tables import Axis, Table, evaluate

REFERENCES = Path(__file__).parent / "references"
FILES = {"corner": "port_table", "axis_far": "port_table_axis_far"}
SHA256 = {"corner": "52e61c76aacdcbe770a5a3f3dbdfdf1a1bf531b4668e192337ee81942e6e7082", "axis_far": "4f1f2cb31510bf53c4d24ff16d3ea10c40d9d79ac6e29f83e1c2ba05da39a54c"}
REFERENCE = {name: np.load(REFERENCES / f"{file}.npz") for name, file in FILES.items()}
PROVENANCES = {name: json.loads((REFERENCES / f"{file}_provenance.json").read_text()) for name, file in FILES.items()}
PORT = REFERENCE["corner"]
OWNER = "toy.wall_table"
BLOCKS = list(FILES)


def block(rho=None, port=PORT, angle=None) -> Table:
    rho = port["rho"] if rho is None else rho
    angle = port["angle"] if angle is None else angle
    axes = (Axis("rho", "1", tuple(rho)), Axis("rho0", "1", tuple(port["rho"])), Axis("angle", "rad", tuple(angle)),
            Axis("offset", "1", tuple(port["offset"])))
    return Table(axes, port["data"])


def misses(table: Table, query: np.ndarray, port: np.ndarray, roundings: int, reference=PORT) -> np.ndarray:
    """Each pair's largest miss over its nine components, in roundings of the block's largest value."""
    found = evaluate(table, jnp.asarray(query), owner=OWNER)
    assert bool(np.asarray(found.in_range).all()) and bool(np.asarray(found.accepted).all())
    epsilon = float(np.finfo(query.dtype).eps)
    scale = float(np.abs(reference["data"]).max())
    return np.abs(np.asarray(found.value, dtype=np.float64) - port).max(axis=(-1, -2)) / (epsilon * scale) > roundings


@pytest.mark.parametrize("name", BLOCKS)
def test_the_reference_is_the_one_its_provenance_names(name):
    port, provenance = REFERENCE[name], PROVENANCES[name]
    assert provenance["pairs"] == len(port["query64"]) == len(port["lookup32"]) == 576
    assert provenance["block"]["name"] == name
    assert port["data"].shape == tuple(len(port[k]) for k in ("rho", "rho", "angle", "offset")) + (3, 3)
    assert provenance["source_table"]["shape"][:4] == [len(port[k]) for k in ("full_rho", "full_rho", "full_angle",
                                                                                "full_offset")]
    assert hashlib.sha256((REFERENCES / f"{FILES[name]}.npz").read_bytes()).hexdigest() == SHA256[name]


def test_the_corner_is_the_near_wall_end_and_the_other_block_the_axis_end_and_the_far_edges():
    corner, far = REFERENCE["corner"], REFERENCE["axis_far"]
    assert corner["rho"][0] > far["rho"][-1] and corner["rho"][-1] == corner["full_rho"][-1]
    assert far["rho"][0] == far["full_rho"][0] and corner["offset"][-1] < far["offset"][len(far["offset"]) // 2]
    assert far["angle"][-1] == far["full_angle"][-1] and far["offset"][-1] == far["full_offset"][-1]
    for query in (far["query64"],):
        assert query[:, 2].max() == far["angle"][-1] and query[:, 3].max() == far["offset"][-1]


@pytest.mark.parametrize("name", BLOCKS)
def test_the_pairs_lie_inside_the_block_on_every_axis_where_the_port_did_not_clip(name):
    port = REFERENCE[name]
    for query in (port["query64"], port["query32"]):
        for axis, knots in enumerate((port["rho"], port["rho"], port["angle"], port["offset"])):
            held = knots.astype(query.dtype)
            assert held[0] <= query[:, axis].min() and query[:, axis].max() <= held[-1]


@pytest.mark.parametrize("name", BLOCKS)
@pytest.mark.parametrize(("precision", "roundings"), [("64", 64), ("32", 128)])
def test_the_evaluator_reads_the_ports_lookup_on_its_own_grid(name, precision, roundings):
    port = REFERENCE[name]
    query, lookup = port[f"query{precision}"], port[f"lookup{precision}"]
    table = block(port=port)
    missed = misses(table, query, lookup, roundings, port)
    found = np.asarray(evaluate(table, jnp.asarray(query), owner=OWNER).value, dtype=np.float64)
    worst = np.abs(found - lookup).max() / (np.finfo(query.dtype).eps * np.abs(port["data"]).max())
    print(f"\n{name}, float{precision}: the largest difference from the port's lookup is {worst:.3g} roundings of the "
          f"block's largest value (declared: {roundings})")
    assert not missed.any(), int(missed.sum())


class _OneBelow:
    def __getattr__(self, name):
        return getattr(switch, name)

    @staticmethod
    def searchsorted(breaks, x, **settings):
        return jnp.maximum(switch.searchsorted(breaks, x, **settings) - 1, 0)


@pytest.mark.parametrize("name", BLOCKS)
def test_control_a_cell_one_below_the_right_one_is_seen(name, monkeypatch):
    port = REFERENCE[name]
    monkeypatch.setattr(evaluate_module, "switch", _OneBelow())
    found = evaluate(block(port=port), jnp.asarray(port["query64"]), owner=OWNER)
    scale = float(np.abs(port["data"]).max())
    missed = ~(np.abs(np.asarray(found.value) - port["lookup64"]).max(axis=(-1, -2)) <= 64 * np.finfo(float).eps
               * scale)
    print(f"\n{name}, a cell one below: {int(missed.sum())} of {missed.size} pairs miss")
    assert missed.any()


@pytest.mark.parametrize("name", BLOCKS)
def test_control_an_interior_knot_moved_by_a_part_in_a_million_is_seen(name):
    """A rho knot of the corner (eight knots), an angle knot of the axis block (whose rho has two): an interior one,
    since the points sit at the first and last knots."""
    port = REFERENCE[name]
    rho, angle = port["rho"].copy(), port["angle"].copy()
    if name == "corner":
        rho[4] *= 1 + 1e-6
    else:
        angle[30] *= 1 + 1e-6
    missed = misses(block(rho, port, angle), port["query64"], port["lookup64"], 64, port)
    print(f"\n{name}, a knot moved by a part in a million: {int(missed.sum())} of {missed.size} pairs miss")
    assert missed.any()
