"""The ``tube`` wall kind, registered on the shipped asset ``unit_tube_R0.997`` (``mime.walls.TubeWall``; the regularised-Stokeslet
method's correction ``mime.nodes.hydro.mrs._tube.TUBE_CORRECTION``): a small tube experiment builds and runs with it, the
confined sphere drags as Haberman and Sayre say, a read past the table's readable range (0.997 of the radius, the
file's own ``readable_rho``) is refused, a body longer than the table reaches is refused when planned, and an asset that
is not in the cache fails by name, naming the command that stores it, and downloads nothing.

The asset must be in the cache (``mime assets fetch unit_tube_R0.997``; CI stores it once per job). The tests never
download it."""

import jax.numpy as jnp
import numpy as np
import pytest
import tube_asset
from mrs_cases import drag_node, free_space, resistance_at
from tube_cases import TUBE_RADIUS, grid_sphere, haberman_sayre, polar_spacing, sphere_in_a_tube

import mime.nodes.hydro.mrs._tube as tube_module
from mime.build import BuildRefused
from mime.build._maddening import node_state, parameter_leaves
from mime.interactions import Drag
from mime.nodes.hydro.mrs import wall_correction_named
from mime.registry import EntryKind, lookup
from mime.scene import Capsule
from mime.units import m, mm, ms

RATIO = 0.1
#: The confined sphere's tolerance and resolutions, as ``test_tube_wall.py`` states them (the oracle tube's).
SPHERE_TOLERANCE, POLAR = 5e-3, (10, 11)


def test_the_tube_wall_kind_is_registered_on_the_shipped_asset_with_the_files_readable_range():
    record = lookup(EntryKind.WALL_KIND, "tube", "0.1.0")
    assert record is not None and record.cls.matches == "cylinder" and record.cls.reads == ("centre", "axis", "radius")
    assert wall_correction_named("tube", "0.1.0") is tube_module.TUBE_CORRECTION
    assert tube_module.shipped_unit_tube().axes[0].readable_range == (0.0, 0.997)
    assert tube_module.shipped_unit_tube() is tube_module.shipped_unit_tube()
    assert tube_asset.path().name == tube_module.assets.reference(tube_module.UNIT_TUBE).sha256


def published_at(built, offset: float) -> tuple[bool, np.ndarray]:
    """``wall_accepted`` and the resistance of the drag node with the ball moved ``offset`` of the tube's radius along x."""
    node = drag_node(built, "ball")
    pose = jnp.asarray((offset * TUBE_RADIUS.value, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=jnp.float64)
    out = node.update(node_state(built.graph, "drag:ball"), {"pose": pose}, 1e-3,
                      params=parameter_leaves(built.graph)["drag:ball"])
    return bool(out["wall_accepted"]), np.asarray(out["resistance"])


@pytest.fixture(scope="module")
def built():
    return sphere_in_a_tube("registered", radius=RATIO, spacing=polar_spacing(RATIO, 6), wall_kind="tube").build()


def test_the_registered_tube_wall_kind_builds_runs_and_refuses_a_read_past_the_readable_range(built):
    """A sphere of a tenth of the radius reaches 0.9 of it at an offset of 0.8 (read) and 1.05 at 0.95 (refused: NaN,
    not accepted, as the file's range ends at 0.997)."""
    built.run(ms(3))
    inside, value = published_at(built, 0.8)
    outside, beyond = published_at(built, 0.95)
    print(f"\naccepted at 0.8 of the radius: {inside}; at 0.95: {outside}")
    assert inside and np.all(np.isfinite(value))
    assert not outside and not np.all(np.isfinite(beyond))


def drag_ratio(polar: int) -> float:
    spacing = polar_spacing(RATIO, polar)
    confined = resistance_at(sphere_in_a_tube("confined", radius=RATIO, spacing=spacing, wall_kind="tube").build(),
                             "ball")
    free = resistance_at(free_space("free", {"ball": grid_sphere(m(RATIO * TUBE_RADIUS.value))},
                                    Drag.MRS(spacing=spacing, blob=0.5, wall_kind="free_space")).build(), "ball")
    return float(confined[2, 2] / free[2, 2])


def test_the_confined_sphere_on_the_shipped_table_drags_as_haberman_and_sayre_say():
    expected = haberman_sayre(RATIO)
    coarse, fine = POLAR
    found = [drag_ratio(polar) / expected - 1.0 for polar in POLAR]
    extrapolated = (fine * found[1] - coarse * found[0]) / (fine - coarse)
    print(f"\nK = {expected:.5f}; misses at {POLAR} polar nodes {np.round(found, 5)}, extrapolated {extrapolated:.5f}")
    assert expected - 1 >= 3 * SPHERE_TOLERANCE and abs(extrapolated) <= SPHERE_TOLERANCE


def test_a_body_longer_than_the_shipped_table_reaches_is_refused_when_planned():
    with pytest.raises(BuildRefused) as refused:
        sphere_in_a_tube("too_long", radius=RATIO, spacing=mm(1.0), wall_kind="tube",
                         hull=Capsule(radius=mm(0.3), height=mm(30.0), axis=(0.0, 0.0, 1.0))).plan()
    assert "drag.wall-range" in refused.value.codes()


def test_an_asset_that_is_not_in_the_cache_fails_by_name_and_downloads_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path))
    monkeypatch.setenv("MIME_OFFLINE", "1")
    with pytest.raises(BuildRefused) as refused:
        sphere_in_a_tube("missing", radius=RATIO, spacing=mm(1.0), wall_kind="tube").plan()
    message = str(refused.value)
    print(f"\n{message}")
    assert refused.value.codes() == ["asset.missing"]
    assert "unit_tube_R0.997" in message and "mime assets fetch unit_tube_R0.997" in message
    assert list(tmp_path.iterdir()) == []
