"""The table gate's read check (``scripts/tube_table/full.py: read_errors``) compares the table's read with the builder
in the frame the evaluator that ships holds and reads: each block's rows turned from the target's cylindrical components
into the source's (``mime.nodes.hydro.mrs._tube``). It once compared with the builder's unturned rows, which misses by
about 2 for any table, the control and the defective one alike, so it checked nothing.

On a small table built by the builder (``tube_cases.builder_table``), read at drawn points between its knots against the
builder's own blocks there, 20 runs of points of their own: the read is within the gate's declared 1e-2. Controls, each
failing in every run: the table scaled by 1.05 (a 5% error) is over the tolerance, and the old comparison, with the
unturned rows, is over it for the same table, which is the mismatch the turn removes."""

import sys
from pathlib import Path

import numpy as np
import pytest
from tube_cases import builder, builder_table

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "tube_table"
sys.path.insert(0, str(SCRIPTS))
import full  # noqa: E402

from mime.nodes.hydro.mrs import unit_tube_table  # noqa: E402

RADII = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
ANGLES = tuple(float(a) for a in np.linspace(0.0, np.pi, 13))
OFFSETS = tuple(float(z) for z in np.linspace(0.0, 0.6, 7))
RUNS = 20


@pytest.fixture(scope="module")
def tables():
    table = builder_table(RADII, ANGLES, OFFSETS)
    scaled = unit_tube_table(1.0, np.asarray(RADII), np.asarray(ANGLES), np.asarray(OFFSETS),
                             1.05 * _data(), readable_rho=None)
    return table, scaled


def _data() -> np.ndarray:
    from tube_cases import builder_data

    return builder_data(RADII, ANGLES, OFFSETS)


def points(run: int):
    """A source, and targets on radii, angles and offsets between the knots, with the builder's blocks there."""
    rng = np.random.default_rng(700 + run)
    source = float(rng.uniform(0.05, 0.45))
    rho = np.sort(rng.uniform(0.03, 0.47, 3))
    phi, z = np.sort(rng.uniform(1.0, 3.0, 3)), np.sort(rng.uniform(0.02, 0.58, 3))
    build = builder()
    reference = build.image_grid(source, rho, phi, z, build.series_truncation(15, 80, 64, 0.9), threads=1)
    rt, ph, zz = np.meshgrid(rho, phi, z, indexing="ij")
    return rt.ravel(), np.full(rt.size, source), ph.ravel(), zz.ravel(), reference.reshape(-1, 3, 3)


def test_the_read_check_passes_the_table_in_the_evaluators_frame_and_fails_the_controls_every_run(tables):
    table, scaled = tables
    tolerance = full.TOLERANCES["read"]
    found = []
    for run in range(RUNS):
        rt, rs, ph, zz, reference = points(run)
        found.append((full.read_errors(table, rt, rs, ph, zz, reference).max(),
                      full.read_errors(scaled, rt, rs, ph, zz, reference).max(),
                      full.relative(full.read_at(table, rt, rs, ph, zz), reference).max()))
    right, five_per_cent, unturned = (np.array(column) for column in zip(*found))
    print(f"\nthe read against the builder: {right.max():.2e} at most; scaled by 1.05: {five_per_cent.min():.2e} at "
          f"least; the unturned rows: {unturned.min():.2e} at least; the declared {tolerance:g}")
    assert right.max() <= tolerance
    assert five_per_cent.min() > tolerance and unturned.min() > tolerance
