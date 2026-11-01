"""The MRS library (``mime.nodes.hydro.mrs``) against the 0.2 line's free-space MRS, the weakest oracle,
on the 0.2 line's own meshes. The port's MRS ran once, in the port's own environment, before any of it was ported
(``generate_port_mrs_references.py``; ``references/port_mrs.npz`` and ``port_mrs_full.npz`` with their provenance:
MIME-port041 at 6485516, float64, CPU), and its outputs are read here; the port's ``mime`` is never imported beside
this one.

**What each frozen case pins.** Each body's 0.2 samples (triangle centroids, flat normals, areas) and blob (half the
mesh's mean spacing), with what the 0.2 code made of them: the resistance its free-space standalone node holds
(``compute_resistance_matrix``, not symmetrised), its free-space Schwarz node's symmetrised resistance and its outputs
for each unit rigid motion (force, torque and the traction at every sample), and for the helix the free-space form of
the MRS reference the G0 anchor matched (``dejongh_benchmark.compute_freespace_R``). The cases: the helix (de Jongh's
FL-9) at 12 x 10 vertices, a sphere and a prolate spheroid (semi-axes 2, 1, 1) at two subdivisions of the 0.2
icosphere, run on every push; and, marked ``slow``, the helix at G0's 24 x 36 and at the surrogate training data's
40 x 40 (the 9,360-unknown system), and the sphere at G0-a's three subdivisions.

**Reproduced through the 0.2 discretisation** (Q-13's option, labelled: the 0.2 mesh's own samples frozen beside the
outputs rather than a tessellated rule added to the kind for the reproduction alone). The mesh is tied to the de Jongh
kind instead: its vertices lie on the kind's lateral surface, and the 0.2 triangle rule applied to them gives the
frozen samples.

**Declared before the first run.** The library's resistance equals each frozen one within 1e-10 of the case's largest
entry, and its point forces divided by the weights equal the frozen tractions within 1e-10 of their largest: the same
algebra in another formulation (point forces against tractions; a vectorised assembly against the port's chunked one;
one LU against SciPy's), so nothing beyond the rounding of a solve of condition number in the hundreds is allowed.

**Measured, not asserted against a bound.** The matrix's symmetry: in point forces it is exact; the 0.2 line's
traction matrix is not, with non-uniform weights. And the model differences the 0.3 kind brings: its own quadrature
(exact surface points, Gauss-Legendre along the axis, the trapezoid round the turn) at the 0.2 counts, and the capped,
closed surface, each printed against the frozen lateral resistance.

The controls, each on the fixed references, so that each fails every run: a sample's weight dropped (it reaches the
resistance through the blob alone), and the 0.2 traction matrix transposed."""

import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.core import SCREW
from mime.nodes.hydro.mrs import blob_squared, point_forces, resistance, rigid_motions, stokeslet_matrix
from mime.scene import Screw
from mime.units import dimensionless, m

REFERENCES = Path(__file__).parent / "references"
SHA256 = {"port_mrs": "aa0ddbb02268f5a4f71035ed40a9b4fc312b561a67a3d4f804a1502ad297169c",
          "port_mrs_full": "cb71eb9270b56714b837cb231b21d1c3e1eff24128413fc4cf0f4f099ccc56ed"}
TOLERANCE = 1e-10
BLOB = 0.5
FILES = {name: np.load(REFERENCES / f"{name}.npz") for name in SHA256 if (REFERENCES / f"{name}.npz").exists()}
PROVENANCE = {name: json.loads((REFERENCES / f"{name}_provenance.json").read_text()) for name in FILES}
CI_CASES = ("helix_coarse", "sphere_coarse", "spheroid_coarse")
FULL_CASES = ("helix_g0", "helix_training", "sphere_g0")
#: FL-9 with its lengths divided by its radius, as the frozen helix is.
LENGTH = 7.47 / 1.56
KIND = Screw(radius=m(1.0), length=m(LENGTH), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33),
             starts=2)


def frozen(case: str) -> dict[str, np.ndarray]:
    held = FILES["port_mrs" if case in CI_CASES else "port_mrs_full"]
    return {key.split("/", 1)[1]: held[key] for key in held.files if key.startswith(f"{case}/")}


def cases(*names: str) -> list:
    """The CI cases, and the full ones marked slow."""
    return [*names, *(pytest.param(case, marks=pytest.mark.slow) for case in FULL_CASES
                      if case.split("_")[0] in {n.split("_")[0] for n in names})]


def worst(found: np.ndarray, expected: np.ndarray) -> float:
    return float(np.abs(np.asarray(found) - expected).max() / np.abs(expected).max())


def library(points: np.ndarray, weights: np.ndarray, blob: float = BLOB) -> np.ndarray:
    return np.asarray(resistance(jnp.asarray(points), blob_squared(jnp.asarray(weights), blob), 1.0, jnp.zeros(3)))


def test_the_frozen_files_are_the_pinned_ones_from_the_port_at_its_commit():
    for name, pinned in SHA256.items():
        if pinned is None:
            continue
        assert hashlib.sha256((REFERENCES / f"{name}.npz").read_bytes()).hexdigest() == pinned
        provenance = PROVENANCE[name]
        assert provenance["sha256"] == pinned and provenance["commit"].startswith("6485516") and provenance["x64"]
        assert set(provenance["cases"]) == set(CI_CASES if name == "port_mrs" else FULL_CASES)


@pytest.mark.parametrize("case", cases(*CI_CASES))
def test_the_library_reproduces_the_02_resistance_on_the_02_samples(case):
    held = frozen(case)
    found = library(held["points"], held["weights"])
    assert math.isclose(float(np.sqrt(blob_squared(jnp.asarray(held["weights"]), BLOB))), float(held["epsilon"]),
                        rel_tol=1e-14)
    misses = {"the standalone node's": worst(found, held["R_library"]),
              "the Schwarz node's, symmetrised": worst(found, held["R_node"])}
    if "R_benchmark" in held:
        misses["compute_freespace_R's, symmetrised"] = worst(found, held["R_benchmark"])
    print(f"\n{case}, {len(held['points'])} samples: " + "; ".join(f"{k} {v:.2e}" for k, v in misses.items()))
    assert all(miss <= TOLERANCE for miss in misses.values()), misses


@pytest.mark.parametrize("case", cases(*CI_CASES))
def test_the_point_forces_are_the_02_nodes_tractions_times_the_weights_and_sum_to_its_force_and_torque(case):
    held = frozen(case)
    points, weights = jnp.asarray(held["points"]), jnp.asarray(held["weights"])
    motions = rigid_motions(points, jnp.zeros(3))
    forces = np.asarray(point_forces(stokeslet_matrix(points, blob_squared(weights, BLOB), 1.0), motions))
    tractions = forces.reshape(-1, 3, 6).transpose(2, 0, 1) / held["weights"][None, :, None]
    loads = np.asarray(motions).T @ forces
    assert worst(tractions, held["traction"]) <= TOLERANCE
    assert worst(loads[:3], held["force"].T) <= TOLERANCE and worst(loads[3:], held["torque"].T) <= TOLERANCE


@pytest.mark.parametrize("case", cases("helix_coarse"))
def test_the_02_helix_mesh_is_a_tessellation_of_the_de_jongh_kinds_lateral_surface(case):
    """Its vertices, turned from the 0.2 frame (the axis along z from its end) into the kind's (along x from its
    centroid), lie on the kind's screw within rounding and on its span; and the 0.2 triangle rule on them gives the
    frozen samples."""
    held = frozen(case)
    vertices = held["vertices"]
    turned = np.stack([vertices[..., 2] - LENGTH / 2, vertices[..., 0], vertices[..., 1]], axis=-1).reshape(-1, 3)
    screw = KIND.region_parameters("screw", _si())
    distance = np.asarray(jax.jit(jax.vmap(lambda p: SCREW.signed_distance(screw, p)))(jnp.asarray(turned)))
    assert np.abs(distance).max() <= 64 * np.finfo(float).eps * LENGTH
    assert np.abs(turned[:, 0]).max() <= LENGTH / 2 * (1 + 1e-15)
    n_theta, n_zeta = vertices.shape[:2]
    corners = [(vertices[i, j], vertices[(i + 1) % n_theta, j], vertices[(i + 1) % n_theta, j + 1], vertices[i, j + 1])
               for i in range(n_theta) for j in range(n_zeta - 1)]
    triangles = np.asarray([tri for v00, v10, v11, v01 in corners for tri in ((v00, v10, v11), (v00, v11, v01))])
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2.0
    radial = triangles.mean(axis=1) * np.array([1.0, 1.0, 0.0])
    normal = cross / (2.0 * area[:, None]) * np.sign(np.sum(cross * radial, axis=1))[:, None]
    assert worst((triangles[:, 0] + triangles[:, 1] + triangles[:, 2]) / 3.0, held["points"]) <= 1e-15
    assert worst(area, held["weights"]) <= 1e-14 and worst(normal, held["normals"]) <= 1e-14


def _si() -> dict[str, float]:
    return {"radius": 1.0, "length": LENGTH, "wavenumber": 2.33, "modulation": 0.33}


@pytest.mark.parametrize("case", CI_CASES)
def test_the_point_force_matrix_is_symmetric_and_the_02_traction_matrix_is_not(case):
    held = frozen(case)
    matrix = np.asarray(stokeslet_matrix(jnp.asarray(held["points"]),
                                         blob_squared(jnp.asarray(held["weights"]), BLOB), 1.0))
    traction = matrix * np.repeat(held["weights"], 3)[None, :]
    skew = float(np.abs(traction - traction.T).max() / np.abs(traction).max())
    spread = float(held["weights"].max() / held["weights"].min())
    eigenvalues = np.linalg.eigvalsh(matrix)
    print(f"\n{case}: point forces |A - A^T| = {np.abs(matrix - matrix.T).max():.1e}; the 0.2 traction matrix "
          f"|A - A^T| / |A| = {skew:.3f}, its weights spread {spread:.2f} to 1; the point-force matrix's eigenvalues "
          f"from {eigenvalues[0]:.3g} to {eigenvalues[-1]:.3g}, condition {eigenvalues[-1] / eigenvalues[0]:.0f}")
    assert np.array_equal(matrix, matrix.T)
    assert skew > 1e-3


@pytest.mark.parametrize("case", CI_CASES)
def test_control_a_samples_weight_dropped_moves_the_resistance_through_the_blob(case):
    held = frozen(case)
    weights = held["weights"].copy()
    weights[len(weights) // 2] = 0.0
    miss = worst(library(held["points"], weights), held["R_library"])
    print(f"\n{case}: one weight dropped moves the resistance by {miss:.2e} of its largest entry")
    assert miss > 100 * TOLERANCE


@pytest.mark.parametrize("case", CI_CASES)
def test_control_the_02_traction_matrix_transposed_is_seen(case):
    held = frozen(case)
    points, weights = jnp.asarray(held["points"]), jnp.asarray(held["weights"])
    matrix = stokeslet_matrix(points, blob_squared(weights, BLOB), 1.0)
    transposed = jnp.repeat(weights, 3)[:, None] * matrix
    motions = rigid_motions(points, jnp.zeros(3))
    forces = jnp.linalg.solve(transposed, motions) * jnp.repeat(weights, 3)[:, None]
    miss = worst(np.asarray(motions.T @ forces), held["R_library"])
    print(f"\n{case}: the traction matrix transposed moves the resistance by {miss:.2e}")
    assert miss > 100 * TOLERANCE


def in_the_02_frame(points: np.ndarray) -> np.ndarray:
    """The kind's points (axis x, origin at the centroid) in the 0.2 frame (axis z, origin at the end)."""
    return np.stack([points[:, 1], points[:, 2], points[:, 0] + LENGTH / 2], axis=-1)


def kind_samples(patch: str, counts: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    held = KIND.samples(_si(), patch, counts)
    return in_the_02_frame(np.asarray(held.points)), np.asarray(held.weights)


@pytest.mark.parametrize("case", cases("helix_coarse"))
def test_the_kinds_quadrature_and_its_caps_are_model_differences_from_the_02_lateral_mesh(case):
    """Reported, not bounded: the kind's lateral samples at the 0.2 vertex grid's counts and at its count of samples,
    and the closed surface (the caps sampled at the lateral's spacing), each against the frozen resistance; every one
    differs by more than the reproduction's tolerance, so each is a model difference and not a rounding."""
    held = frozen(case)
    n_theta, n_zeta = held["vertices"].shape[:2]
    expected = held["R_library"]
    rows = {}
    for label, counts in (("the 0.2 vertex counts", (n_theta, n_zeta)),
                          ("the 0.2 sample count", (2 * n_theta, n_zeta - 1))):
        points, weights = kind_samples("lateral", counts)
        rows[f"lateral at {label} {counts}"] = (library(points, weights), len(points))
    points, weights = kind_samples("lateral", (2 * n_theta, n_zeta - 1))
    spacing = math.sqrt(weights.sum() / len(weights))
    caps = KIND.sample_counts(_si(), spacing, "caps")
    cap_points, cap_weights = kind_samples("caps", caps)
    rows[f"closed, caps at {caps}"] = (library(np.concatenate([points, cap_points]),
                                               np.concatenate([weights, cap_weights])), len(points) + len(cap_points))
    print(f"\n{case}: against the frozen lateral resistance ({len(held['points'])} samples), largest entry "
          f"{np.abs(expected).max():.4g}")
    for label, (found, count) in rows.items():
        print(f"  {label}, {count} samples: largest {worst(found, expected):.3e}; diagonal "
              + " ".join(f"{d:+.3%}" for d in np.diag(found) / np.diag(expected) - 1)
              + f"; axial coupling {found[2, 5] / expected[2, 5] - 1:+.3%}")
        assert worst(found, expected) > 100 * TOLERANCE
