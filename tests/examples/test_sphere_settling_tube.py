"""``examples/sphere_settling_tube.py``: the steel sphere of ``sphere_settling.py`` released at the centre of a closed
horizontal tube of water, run as the example runs it (its table, ``unit_tube_R0.997``, stored by CI before the tests).

**What this bounds.** That the ball comes to rest on the tube's floor, held by the contact at its margin, and never
leaves the lumen; and that early in its fall, at the tube's centre, its speed sits below Stokes's free-space speed by
far more than the rung's own error at this spacing, where the same sphere at the same spacing in free space
(``sphere_settling.py``, the control) sits within it: the wall is in the drag. It does not bound the confined speed's
accuracy: no published value for a sphere moving across a tube at its centre is compared.

And the reason the docstring gives for leaving the end caps out of the physics: the slowest decay of Stokes flow along
a circular tube, solved here from its eigenvalue equation (azimuthal order one: 2.568 + 1.123i per radius; the same
equation's axisymmetric root, 4.466 + 1.467i, beside it), is the decay of MIME's table of the tube's image system
(Liron and Shahar's series) along the axis for a force across it, to 0.1% from 1.5 to 4.5 radii; the control, either
other root, misses it by more than its own size."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import tube_asset
from scipy.optimize import fsolve
from scipy.special import jv, jvp

from mime.units import m_per_s

ROOT = Path(__file__).resolve().parents[2]
#: The rung's error at the example's spacing, a quarter of the radius: the settled speed 1.75% below Stokes's in free
#: space (docs/data/spacing_tables.json; test_sphere_settling.py holds it within 2%).
RUNG_ERROR = 0.02


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"example_{name}", ROOT / "examples" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example():
    return load("sphere_settling_tube")


@pytest.fixture(scope="module")
def written(example, tmp_path_factory):
    tube_asset.path()  # stored by CI; a missing table fails here by name, with the command that stores it
    return example.run(tmp_path_factory.mktemp("tube") / "run")


def test_the_ball_comes_to_rest_on_the_floor_within_the_margin_and_never_leaves_the_lumen(example, written):
    centre = example.heights(written)
    off_axis = np.hypot(centre[:, 1], centre[:, 2]) * 1e6
    gaps = example.TUBE_UM - example.RADIUS_UM - off_axis
    assert np.all(gaps > 0.0), "the ball left the lumen"
    assert gaps[-1] == pytest.approx(example.MARGIN_UM, rel=1e-2) and centre[-1, 2] < 0.0
    assert np.all(np.abs(centre[:, 0]) < 1e-9), "the fall is across the tube, with no drift along it"
    resting = int(np.argmax(np.abs(gaps - example.MARGIN_UM) < 1e-2 * example.MARGIN_UM))
    assert 60.0 <= resting * example.STEP_MS <= 72.0, "the fall takes about 66 ms, as the docstring says"
    speed = np.asarray(written.result["swim_speed:ball"].speed.values.to(m_per_s))
    assert abs(speed[-1]) < 1e-3 * example.stokes_speed()


def test_early_in_the_fall_the_speed_is_below_stokess_by_far_more_than_the_rungs_error(example, written):
    ratio = example.early_speed(written) / example.stokes_speed()
    assert round(ratio, 2) == 0.70, "the docstring's number"
    assert ratio < 1.0 - 10.0 * RUNG_ERROR
    # The control: the same sphere at the same spacing in free space settles within the rung's error of Stokes's law.
    _, free = load("sphere_settling").run(10.0)
    assert 1.0 - RUNG_ERROR < free < 1.0


def pipe_determinant(k: complex, order: int) -> complex:
    """No slip at the unit tube's wall for Stokes flow ``~ exp(i order theta - k z)``: a poloidal potential
    ``a J(kr) + b kr J'(kr)`` and a toroidal ``c J(kr)``, the velocity's three components at r = 1."""
    j = [jvp(order, k, m) * k ** m if m else jv(order, k) for m in range(4)]
    a = (j[0], j[1], j[2])
    b = (k * jvp(order, k, 1), k * jvp(order, k, 1) + k * k * jvp(order, k, 2),
         2 * k * k * jvp(order, k, 2) + k ** 3 * jvp(order, k, 3))
    n = order
    rows = np.array([[-k * a[1], -k * b[1], 1j * n * j[0]],
                     [-1j * n * k * a[0], -1j * n * k * b[0], -j[1]],
                     [-(a[2] + a[1] - n * n * a[0]), -(b[2] + b[1] - n * n * b[0]), 0.0]], dtype=complex)
    return complex(np.linalg.det(rows if n else rows[np.ix_([0, 2], [0, 1])]))


def slowest(order: int) -> complex:
    roots: set[tuple[float, float]] = set()
    for start in [(re, im) for re in np.arange(1.0, 6.0, 0.5) for im in (0.5, 1.0, 1.5)]:
        found, _, done, _ = fsolve(lambda v: [pipe_determinant(v[0] + 1j * v[1], order).real,
                                              pipe_determinant(v[0] + 1j * v[1], order).imag], start,
                                   full_output=True)
        if done == 1 and found[0] > 0.5 and abs(pipe_determinant(found[0] + 1j * found[1], order)) < 1e-9:
            roots.add((round(float(found[0]), 6), round(abs(float(found[1])), 6)))
    re, im = min(roots)
    return complex(re, im)


def tail_miss(rate: complex) -> float:
    """How far the table's whole Green's function on the axis, a force across it, from 1.5 to 4.5 radii, is from
    ``exp(-Re(rate) z)`` times a sinusoid of ``Im(rate) z``, fitted: the largest miss over the fit's size."""
    with np.load(tube_asset.path()) as held:
        along, data = held["dz_grid"], held["data"][0, 0, 0, :, 0, 0]
    chosen = (along >= 1.5) & (along <= 4.5)
    z = along[chosen]
    whole = (data[chosen] + 1.0 / (8.0 * math.pi * z)) * np.exp(rate.real * z)
    basis = np.stack([np.cos(rate.imag * z), np.sin(rate.imag * z)], axis=1)
    fitted = basis @ np.linalg.lstsq(basis, whole, rcond=None)[0]
    return float(np.max(np.abs(whole - fitted)) / np.max(np.abs(fitted)))


def test_a_disturbance_decays_along_the_tube_as_the_slowest_mode_the_docstring_names(example):
    across, axisymmetric = slowest(1), slowest(0)
    assert (round(across.real, 3), round(across.imag, 3)) == (2.568, 1.123)
    assert (round(axisymmetric.real, 3), round(axisymmetric.imag, 3)) == (4.466, 1.467)
    assert tail_miss(across) < 1e-3
    assert f"exp(-2.568 z / R)" in example.__doc__ and math.exp(-2.568 * 4.0) == pytest.approx(3.5e-5, rel=0.02)
    assert example.CAPS_UM / example.TUBE_UM == 4.0
    # The control: the axisymmetric root and the next order's do not describe the table's tail.
    assert tail_miss(axisymmetric) > 1.0 and tail_miss(slowest(2)) > 1.0


def test_its_save_splits_each_field_into_the_blocks_of_the_units_its_build_declares(written):
    """``mime verify``'s blocks by unit, on the demo's own save: each field's resolution is recorded by the blocks the
    units the artefact records give (``tests/saved/test_verify_blocks.py`` judges the demo's arrays by them), so the
    build still declares those units."""
    import json

    from mime.saved import LOCK_FILE, read_folder
    from mime.saved._scales import blocks

    measured = json.loads((ROOT / "tests" / "saved" / "artefacts" / "tube_demo_resolution.json").read_text(
        encoding="utf-8"))
    resolution = json.loads((written.experiment / LOCK_FILE).read_text(encoding="utf-8"))["reference"]["resolution"]
    reference = read_folder(written.experiment).reference
    assert sorted(resolution) == sorted(measured["resolution"]["float32"])
    for label, held in resolution.items():
        shape = reference[f"history/{label.replace('.', '/', 1)}"].shape[1:]
        assert sorted(held) == sorted(name for name, _ in blocks(measured["units"].get(label), shape)), label
    assert sorted(resolution["ball.pose"]) == ["1", "m"] and sorted(resolution["ball.twist"]) == ["m/s", "rad/s"]


def test_the_run_and_its_view_state_the_limit(example, written, tmp_path):
    from mime import viz

    for line in ("down to a gap of about one sample spacing", "not modelled in 0.3.0.dev1",
                 "the contact, not the fluid, holds the ball on the floor", "are planned for 0.3.0.dev2"):
        assert line in " ".join(example.__doc__.split()) and line in example.LIMIT, line
    assert written.recording is not None
    page, result = viz.web(written.recording.parent, tmp_path / "view.html", caption=example.LIMIT)
    text = page.read_text(encoding="utf-8")
    assert "planned for 0.3.0.dev2" in text and result.withheld == {} and result.samples > 30
    shown = written.view(caption=example.LIMIT)  # the notebook's form, the viewer from jsDelivr
    assert "planned for 0.3.0.dev2" in shown.html and shown.result.glb == result.glb
