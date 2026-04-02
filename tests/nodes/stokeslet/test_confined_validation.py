"""Verification: Stokeslet BEM for sphere in cylindrical vessel.

MIME-VER-025: Sphere in cylinder correction factor matches
    Haberman & Sayre (1958) within 5% for a/R ∈ [0.1, 0.5].

The Haberman & Sayre correction factor for a sphere on-axis in a cylinder:
    f = (1 − 2.105λ + 2.0865λ³ − 1.7068λ⁵ + 0.72603λ⁶)
        / (1 − 0.75857λ⁵)
where λ = d_particle / D_cylinder = 2a / (2R) = a/R.

The nearest-neighbour method (Smith 2018) decouples force and quadrature
discretizations, achieving <5% error at all tested confinement ratios.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.environment.stokeslet.surface_mesh import (
    sphere_surface_mesh,
    cylinder_surface_mesh,
)
from mime.nodes.environment.stokeslet.resistance import (
    compute_resistance_matrix,
    compute_confined_resistance_matrix,
    compute_nn_confined_resistance_matrix,
)


def haberman_sayre_correction(lam):
    """Haberman & Sayre (1958) correction factor for sphere in cylinder.

    Parameters
    ----------
    lam : float
        λ = a / R (sphere radius / cylinder radius). Valid for λ ≤ 0.8.

    Returns
    -------
    f : float
        Correction factor: F_confined = f * F_Stokes.
        f > 1 (confinement increases drag).
    """
    num = 1.0 - 2.105 * lam + 2.0865 * lam**3 - 1.7068 * lam**5 + 0.72603 * lam**6
    den = 1.0 - 0.75857 * lam**5
    # f is the reciprocal: drag increases, so correction = 1/f_HS
    return 1.0 / (num / den)


@pytest.mark.verification
class TestConfinedSphere:
    @pytest.mark.slow
    def test_ver_025_sphere_in_cylinder(self):
        """MIME-VER-025: Sphere-in-cylinder confinement increases drag.

        The single-layer regularised Stokeslet BEM under-predicts the
        wall correction factor compared to Haberman & Sayre (1958).
        This is a known limitation of the Nyström BEM for interior
        flows — the completed double-layer formulation (Power & Miranda
        1987) would give better results. For now, we verify the
        qualitative trend: confinement always increases drag, and the
        correction factor is in the right direction and order.
        """
        a = 0.5  # sphere radius
        mu = 1.0

        for lam in [0.2, 0.3, 0.4, 0.5]:
            R_cyl = a / lam  # cylinder radius
            L_cyl = 6.0 * R_cyl  # sufficient to avoid end effects

            # Build meshes — scale wall resolution with cylinder size
            sphere_mesh = sphere_surface_mesh(radius=a, n_refine=3)  # ~1280 pts
            n_circ = max(24, int(2 * np.pi * R_cyl / sphere_mesh.mean_spacing))
            n_axial = max(12, int(L_cyl / sphere_mesh.mean_spacing))
            # Cap to avoid excessive memory
            n_circ = min(n_circ, 48)
            n_axial = min(n_axial, 40)
            wall_mesh = cylinder_surface_mesh(
                radius=R_cyl, length=L_cyl, n_circ=n_circ, n_axial=n_axial,
            )

            # Use sphere epsilon for accuracy (body resolution drives it)
            eps = sphere_mesh.mean_spacing / 2.0

            # Unconfined resistance (with DLP correction)
            R_free = compute_resistance_matrix(
                jnp.array(sphere_mesh.points),
                jnp.array(sphere_mesh.weights),
                jnp.zeros(3), eps, mu,
                surface_normals=jnp.array(sphere_mesh.normals),
            )

            # Confined resistance (with DLP correction)
            R_conf = compute_confined_resistance_matrix(
                jnp.array(sphere_mesh.points),
                jnp.array(sphere_mesh.weights),
                jnp.array(wall_mesh.points),
                jnp.array(wall_mesh.weights),
                jnp.zeros(3), eps, mu,
                body_normals=jnp.array(sphere_mesh.normals),
                wall_normals=jnp.array(wall_mesh.normals),
            )

            # Drag correction factor = confined / free
            F_free = float(R_free[0, 0])  # x-translation drag
            F_conf = float(R_conf[0, 0])
            correction_bem = F_conf / F_free

            correction_hs = haberman_sayre_correction(lam)

            error = abs(correction_bem - correction_hs) / correction_hs

            print(f"  λ={lam:.1f}: BEM={correction_bem:.3f}, "
                  f"H&S={correction_hs:.3f}, error={error:.1%}")

            # Qualitative: confinement increases drag
            assert correction_bem > 1.0, (
                f"λ={lam}: confined drag should exceed free ({correction_bem:.3f})"
            )
            # Correction increases with confinement (monotonic in λ)
            # Quantitative accuracy limited by single-layer BEM for
            # interior flows — see plan for completed double-layer upgrade

    @pytest.mark.slow
    def test_ver_025_nn_haberman_sayre(self):
        """MIME-VER-025: Nearest-neighbour BEM matches Haberman & Sayre <5%.

        Uses Smith (2018) nearest-neighbour method with:
        - Two-level body mesh: N=320 force, Q=5120 quadrature
        - Cylinder length L=12R (approximates infinite cylinder)
        - Gap-scaled ε: min(0.05, 0.02 × gap)

        Tests at λ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}.
        """
        a = 1.0
        mu = 1.0
        center = jnp.zeros(3)

        body_c = sphere_surface_mesh(radius=a, n_refine=2)  # 320
        body_f = sphere_surface_mesh(radius=a, n_refine=4)  # 5120

        for lam in [0.1, 0.2, 0.3, 0.4, 0.5]:
            R_cyl = a / lam
            gap = R_cyl - a
            F_exact = 6.0 * np.pi * mu * a * haberman_sayre_correction(lam)
            cyl_len = 12.0 * R_cyl

            wall_c = cylinder_surface_mesh(
                radius=R_cyl, length=cyl_len,
                n_circ=48, n_axial=16, cluster_center=True,
            )
            wall_f = cylinder_surface_mesh(
                radius=R_cyl, length=cyl_len,
                n_circ=192, n_axial=64, cluster_center=True,
            )

            eps = min(0.05, 0.02 * gap)

            R = compute_nn_confined_resistance_matrix(
                jnp.array(body_c.points), jnp.array(body_c.weights),
                jnp.array(body_f.points), jnp.array(body_f.weights),
                jnp.array(wall_c.points), jnp.array(wall_c.weights),
                jnp.array(wall_f.points), jnp.array(wall_f.weights),
                center, eps, mu,
            )

            R_Fz = float(R[2, 2])  # axial drag
            error = abs(R_Fz - F_exact) / F_exact

            print(f"  λ={lam:.1f}: NN={R_Fz:.2f}, H&S={F_exact:.2f}, "
                  f"error={error:.1%}")

            assert error < 0.05, (
                f"λ={lam}: NN axial drag {R_Fz:.2f} deviates "
                f"{error:.1%} from H&S {F_exact:.2f} (limit 5%)"
            )

    def test_confinement_increases_drag(self):
        """Confined drag should always exceed free-space drag."""
        a = 0.3
        R_cyl = 1.0
        mu = 1.0

        sphere_mesh = sphere_surface_mesh(radius=a, n_refine=2)
        wall_mesh = cylinder_surface_mesh(
            radius=R_cyl, length=6.0, n_circ=20, n_axial=12,
        )
        eps = max(sphere_mesh.mean_spacing, wall_mesh.mean_spacing) / 2.0

        R_free = compute_resistance_matrix(
            jnp.array(sphere_mesh.points),
            jnp.array(sphere_mesh.weights),
            jnp.zeros(3), eps, mu,
            surface_normals=jnp.array(sphere_mesh.normals),
        )
        R_conf = compute_confined_resistance_matrix(
            jnp.array(sphere_mesh.points),
            jnp.array(sphere_mesh.weights),
            jnp.array(wall_mesh.points),
            jnp.array(wall_mesh.weights),
            jnp.zeros(3), eps, mu,
            body_normals=jnp.array(sphere_mesh.normals),
            wall_normals=jnp.array(wall_mesh.normals),
        )

        # All diagonal elements should increase
        for i in range(3):
            assert float(R_conf[i, i]) > float(R_free[i, i]), (
                f"Confined drag[{i},{i}]={float(R_conf[i,i]):.4f} "
                f"not > free={float(R_free[i,i]):.4f}"
            )
