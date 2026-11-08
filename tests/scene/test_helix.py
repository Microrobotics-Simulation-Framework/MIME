"""The helix kind (``mime.scene.Helix``): its samples, drawing and closed forms by the kind-conformance harness (the
samples on its level set, the distance from its centreline's arc, with its outward normal; the weights summing to each
patch's area; the volume and the inertia by the divergence theorem over the samples against its closed forms), at
settings of either hand, whole and part turns, thin and thick filaments; its centroid, the closed form, against the
first moment its samples integrate; and the settings it refuses (a filament folding on its inner side, neighbouring
turns meeting)."""

import numpy as np
import pytest

from mime.scene import Helix
from mime.testing import drawing_problems, kind_problems
from mime.units import DIMENSIONLESS, Quantity, m

SETTINGS = (
    {"radius": 0.15e-3, "pitch": 0.6e-3, "turns": 2.5, "filament_radius": 0.03e-3},
    {"radius": 0.3e-3, "pitch": 1.0e-3, "turns": 1.0, "filament_radius": 0.05e-3},
    {"radius": 0.2e-3, "pitch": -0.5e-3, "turns": 3.25, "filament_radius": 0.02e-3},
    {"radius": 10e-6, "pitch": 80e-6, "turns": 0.6, "filament_radius": 2e-6},
)


def helix(settings):
    return Helix(radius=m(settings["radius"]), pitch=m(settings["pitch"]),
                 turns=Quantity(settings["turns"], DIMENSIONLESS), filament_radius=m(settings["filament_radius"]))


@pytest.mark.parametrize("settings", SETTINGS, ids=range(len(SETTINGS)))
def test_the_harness_holds_its_samples_and_closed_forms(settings):
    kind = helix(settings)
    assert kind.setting_problems(settings) == []
    assert kind_problems(kind, settings, settings["filament_radius"] / 2.0) == []


@pytest.mark.parametrize("settings", SETTINGS[:2], ids=range(2))
def test_the_harness_holds_its_drawing(settings):
    assert drawing_problems(helix(settings), settings, 32) == []


@pytest.mark.parametrize("settings", SETTINGS, ids=range(len(SETTINGS)))
def test_its_origin_is_its_centroid(settings):
    kind = helix(settings)
    spacing = settings["filament_radius"] / 2.0
    first = np.zeros(3)
    for patch in kind.patches:
        samples = kind.samples(settings, patch, kind.sample_counts(settings, spacing, patch))
        points, normals, weights = (np.asarray(v, dtype=np.float64) for v in
                                    (samples.points, samples.normals, samples.weights))
        # the first moment by the divergence theorem: int x dV = 1/2 int x (x . n) dA
        first += 0.5 * np.einsum("ni,n,n->i", points, np.sum(points * normals, axis=1), weights)
    scale = kind.volume(settings) * kind.enclosing_radius(settings)
    assert np.max(np.abs(first)) <= 1e-12 * scale


def test_it_refuses_a_filament_that_folds_or_turns_that_meet():
    folding = {"radius": 0.15e-3, "pitch": 0.1e-3, "turns": 2.0, "filament_radius": 0.16e-3}
    assert any("folds" in problem or "meet" in problem for problem in helix(folding).setting_problems(folding))
    meeting = {"radius": 0.15e-3, "pitch": 0.05e-3, "turns": 2.0, "filament_radius": 0.03e-3}
    assert any("neighbouring turns meet" in problem for problem in helix(meeting).setting_problems(meeting))
    flat = {"radius": 0.15e-3, "pitch": 0.0, "turns": 2.0, "filament_radius": 0.03e-3}
    assert any("handedness" in problem for problem in helix(flat).setting_problems(flat))
