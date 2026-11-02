"""Review 7l-c's F1, held: the walk's T4 screw (``scripts/inertia_arms.py``'s experiment: FL-9 under the de Jongh
surrogate drag and a field turning at 5 Hz in script A's tube) with the contact 0.2.0 attached, at its default spacing.

- **An anisotropic arm** (the parts' diagonal: the axial moment the smallest, as a real screw's is) turns broadside
  about half a second in, as the review found, and is held at the bore's limit: no point of its body, read from a
  dense quadrature of it (a sixtieth of its size apart, every tenth step, judged from outside the contact), passes the
  wall, and its tilt stops where its hull meets both walls. The control: the same arm with no contact, the review's,
  passes through the wall.
- **The isotropic arm** (script A's inertia) stays on the axis, where no point of its hull reaches the wall: its run
  is the one with no contact to the last bit (``np.array_equal``). The control: the anisotropic arm's run with and
  without the contact differs."""

import importlib.util
import sys
from functools import cache
from pathlib import Path

import numpy as np
from hull_cases import FL9_SETTINGS, TUBE_RADIUS, kind_points, rotation

from mime.interactions import Contact
from mime.units import mm, s

ROOT = Path(__file__).resolve().parents[2]
#: The run, from rest on the axis: the review's turn has passed 10 degrees at 0.47 to 0.49 s and is broadside by
#: 0.6 s with no contact.
SECONDS = 0.7
#: The dense quadrature's spacing, a sixtieth of FL-9's crests' radius (35 um, its farthest point within about 0.2 um of
#: its nearest point, against the margin's 0.1 mm): the body itself, read from outside the contact's own points.
DENSE = 2.0748e-3 / 60


def arms():
    spec = importlib.util.spec_from_file_location("_inertia_arms", ROOT / "scripts" / "inertia_arms.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ARMS = arms()


@cache
def history(inertia: str, held: bool) -> tuple[np.ndarray, np.ndarray]:
    """The arm's poses and twists over the run, every step's."""
    experiment = ARMS.experiment(ARMS.ARMS[inertia])
    if held:
        experiment.attach(Contact.Projection(margin=mm(0.1)), body="umr", surface="tube.inner_surface")
    found = experiment.build().run(s(SECONDS), history=True).history
    assert found is not None
    return np.asarray(found["umr"]["pose"], dtype=np.float64), np.asarray(found["umr"]["twist"], dtype=np.float64)


@cache
def dense_points() -> np.ndarray:
    return kind_points(ARMS.Screw(radius=mm(1.56), length=mm(7.47), wavenumber=ARMS.dimensionless(2.33),
                                  modulation=ARMS.dimensionless(0.33), starts=2), FL9_SETTINGS, DENSE)


def deepest(poses: np.ndarray, every: int = 10) -> float:
    """The least gap of the dense body to the wall itself over the run (negative: through it), every ``every``th
    step."""
    least = np.inf
    for pose in poses[::every]:
        placed = pose[:3] + dense_points() @ rotation(pose[3:]).T
        least = min(least, float(TUBE_RADIUS - np.hypot(placed[:, 1], placed[:, 2]).max()))
    return least


def tilt(poses: np.ndarray) -> np.ndarray:
    """The angle of the body's axis to the tube's, whichever way it points, degrees."""
    along = 1.0 - 2.0 * (poses[:, 5] ** 2 + poses[:, 6] ** 2)
    return np.degrees(np.arccos(np.clip(np.abs(along), 0.0, 1.0)))


def test_an_anisotropic_screw_turns_broadside_and_is_held_at_the_bores_limit():
    poses, _ = history("diagonal", held=True)
    turned = tilt(poses)
    print(f"\nthe largest tilt {turned.max():.3f} degrees, at the end {turned[-1]:.3f}; the body's least gap to the "
          f"wall {deepest(poses):.3e} m")
    assert turned.max() > 15.0, "the arm did not turn: the check could not discriminate"
    assert turned.max() < 22.0, "held past where its hull meets both walls"
    assert deepest(poses) > 0.0


def test_control_without_the_contact_the_screw_goes_through_the_wall():
    poses, _ = history("diagonal", held=False)
    assert deepest(poses) < -1e-4


def test_the_isotropic_screw_runs_as_with_no_contact_to_the_last_bit():
    for held, free in zip(history("isotropic", held=True), history("isotropic", held=False), strict=True):
        assert np.array_equal(held, free)


def test_control_the_anisotropic_screw_runs_otherwise_with_the_contact():
    held, free = history("diagonal", held=True)[0], history("diagonal", held=False)[0]
    assert not np.array_equal(held, free)
