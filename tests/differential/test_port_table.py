"""The table evaluator against the 0.2 line's wall-table lookup (``cylinder_wall_table.py``, a hand-written clipped
multilinear lookup with a float32 JAX twin), the weakest oracle, on the 0.2 table's own grid. The port's lookup ran
once, in the port's own environment, on a block of a de Jongh wall table (its own knots and values: rho and rho0 near
the wall, the first angles and axial offsets), at the coordinates of every pair of 24 points inside the block, as the
port computed them (``generate_port_table_references.py``; ``references/port_table.npz`` and its provenance:
MIME-port041 at 6485516). ``mime.core.tables`` reads the same block here as a linear table of four axes and a 3 x 3
value. The port's ``mime`` is never imported beside this one.

**Inside the range where the port did not clip.** Every pair lies inside the block on every axis (the generator
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
printed: a cell one below the right one, and one rho knot moved by a part in a million."""

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
PORT = np.load(REFERENCES / "port_table.npz")
PROVENANCE = json.loads((REFERENCES / "port_table_provenance.json").read_text())
OWNER = "toy.wall_table"


def block(rho=None) -> Table:
    rho = PORT["rho"] if rho is None else rho
    axes = (Axis("rho", "1", tuple(rho)), Axis("rho0", "1", tuple(PORT["rho"])), Axis("angle", "rad",
                                                                                      tuple(PORT["angle"])),
            Axis("offset", "1", tuple(PORT["offset"])))
    return Table(axes, PORT["data"])


def misses(table: Table, query: np.ndarray, port: np.ndarray, roundings: int) -> np.ndarray:
    """Each pair's largest miss over its nine components, in roundings of the block's largest value."""
    found = evaluate(table, jnp.asarray(query), owner=OWNER)
    assert bool(np.asarray(found.in_range).all()) and bool(np.asarray(found.accepted).all())
    epsilon = float(np.finfo(query.dtype).eps)
    scale = float(np.abs(PORT["data"]).max())
    return np.abs(np.asarray(found.value, dtype=np.float64) - port).max(axis=(-1, -2)) / (epsilon * scale) > roundings


def test_the_reference_is_the_one_its_provenance_names():
    assert PROVENANCE["pairs"] == len(PORT["query64"]) == len(PORT["lookup32"]) == 576
    assert PORT["data"].shape == (8, 8, 6, 7, 3, 3)
    assert PROVENANCE["source_table"]["shape"][:4] == [len(PORT[k]) for k in ("full_rho", "full_rho", "full_angle",
                                                                               "full_offset")]
    assert hashlib.sha256((REFERENCES / "port_table.npz").read_bytes()).hexdigest() == \
        "52e61c76aacdcbe770a5a3f3dbdfdf1a1bf531b4668e192337ee81942e6e7082"


def test_the_pairs_lie_inside_the_block_on_every_axis_where_the_port_did_not_clip():
    for query in (PORT["query64"], PORT["query32"]):
        for axis, knots in enumerate((PORT["rho"], PORT["rho"], PORT["angle"], PORT["offset"])):
            held = knots.astype(query.dtype)
            assert held[0] <= query[:, axis].min() and query[:, axis].max() <= held[-1]


@pytest.mark.parametrize(("precision", "roundings"), [("64", 64), ("32", 128)])
def test_the_evaluator_reads_the_ports_lookup_on_its_own_grid(precision, roundings):
    query, port = PORT[f"query{precision}"], PORT[f"lookup{precision}"]
    missed = misses(block(), query, port, roundings)
    found = np.asarray(evaluate(block(), jnp.asarray(query), owner=OWNER).value, dtype=np.float64)
    worst = np.abs(found - port).max() / (np.finfo(query.dtype).eps * np.abs(PORT["data"]).max())
    print(f"\nfloat{precision}: the largest difference from the port's lookup is {worst:.3g} roundings of the block's "
          f"largest value (declared: {roundings})")
    assert not missed.any(), int(missed.sum())


class _OneBelow:
    def __getattr__(self, name):
        return getattr(switch, name)

    @staticmethod
    def searchsorted(breaks, x, **settings):
        return jnp.maximum(switch.searchsorted(breaks, x, **settings) - 1, 0)


def test_control_a_cell_one_below_the_right_one_is_seen(monkeypatch):
    monkeypatch.setattr(evaluate_module, "switch", _OneBelow())
    found = evaluate(block(), jnp.asarray(PORT["query64"]), owner=OWNER)
    scale = float(np.abs(PORT["data"]).max())
    missed = ~(np.abs(np.asarray(found.value) - PORT["lookup64"]).max(axis=(-1, -2)) <= 64 * np.finfo(float).eps
               * scale)
    print(f"\na cell one below: {int(missed.sum())} of {missed.size} pairs miss")
    assert missed.any()


def test_control_a_rho_knot_moved_by_a_part_in_a_million_is_seen():
    rho = PORT["rho"].copy()
    rho[4] *= 1 + 1e-6
    missed = misses(block(rho), PORT["query64"], PORT["lookup64"], 64)
    print(f"\na rho knot moved by a part in a million: {int(missed.sum())} of {missed.size} pairs miss")
    assert missed.any()
