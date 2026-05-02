"""A2b — Validate surface_integral_force on the analytical Stokes flow.

Prescribe the exact Stokes flow around a translating sphere and check
that surface_integral_force returns 6πμaU. This isolates the
extraction formula from any simulation transient/convergence issues.

Stokes flow past a stationary sphere (U_inf in +x):
  u_r =  U_inf cos(θ) [1 - (3a)/(2r) + a³/(2r³)]
  u_θ = -U_inf sin(θ) [1 - (3a)/(4r) - a³/(4r³)]
  p   = p_inf - (3/2) μ U_inf cos(θ) a / r²

Drag on sphere: F_x = 6πμaU_inf.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp

from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.ibm import surface_integral_force
from mime.nodes.environment.fvm.sdf import sphere_sdf


def stokes_flow_around_sphere(x, *, U_inf, a, mu):
    """Velocity (u_x,u_y,u_z) and pressure p at points x. Sphere at origin.
    Outside sphere only — inside set u=0, p=p_inf.
    """
    r = np.sqrt(np.sum(x ** 2, axis=-1))
    cos_theta = x[..., 0] / np.maximum(r, 1e-30)
    sin_theta_phi = np.sqrt(1 - cos_theta ** 2)   # sin(θ)
    # Velocity in spherical → cartesian
    inside = r < a
    r_safe = np.where(r > 1e-30, r, 1.0)
    u_r = U_inf * cos_theta * (1 - 3*a/(2*r_safe) + a**3/(2*r_safe**3))
    u_theta = -U_inf * sin_theta_phi * (1 - 3*a/(4*r_safe) - a**3/(4*r_safe**3))

    # cartesian decomposition: r_hat = x/r ; theta_hat = (cos θ x̂ - r̂ cos θ) / sin θ
    # Easier: do the Cartesian decomp via x components only (axisymmetric).
    # u_x = u_r cos θ + u_θ * (- sin θ) — wait theta_hat in xy plane
    # Skip the complication: project u onto cartesian basis.
    # For axisymmetric flow with axis = +x:
    #   r_hat · x̂ = cos θ
    #   theta_hat · x̂ = -sin θ
    #   Other components live in y-z plane.
    u_x = u_r * cos_theta - u_theta * sin_theta_phi
    # In y, z: u has only u_θ (perpendicular to axis), distributed in
    # transverse direction (y, z plane).
    # tangent direction unit vector in (y,z): (y, z)/sqrt(y²+z²)
    rho = np.sqrt(x[..., 1] ** 2 + x[..., 2] ** 2)
    rho_safe = np.where(rho > 1e-30, rho, 1.0)
    sin_phi = x[..., 1] / rho_safe   # really direction in y-z
    cos_phi = x[..., 2] / rho_safe
    u_perp = u_r * sin_theta_phi + u_theta * cos_theta
    u_y = u_perp * sin_phi
    u_z = u_perp * cos_phi

    u = np.stack([u_x, u_y, u_z], axis=-1)
    u = np.where(inside[..., None], 0.0, u)

    p = -1.5 * mu * U_inf * cos_theta * a / np.maximum(r_safe, 1e-30)**2
    p = np.where(inside, 0.0, p)
    return u.astype(np.float32), p.astype(np.float32)


def main():
    print("=" * 78)
    print("A2b — surface_integral_force on analytical Stokes sphere")
    print("=" * 78)
    a = 0.1
    U_inf = 0.01
    mu = 1.0
    L_box = 12 * a
    print(f"  a={a}, U_inf={U_inf}, mu={mu}, box={L_box}")
    print(f"  Analytical drag F_x = 6πμaU = {6*np.pi*mu*a*U_inf:.4e}")

    for cpr in (4, 8, 12):    # 16 OOMs the 6GB GPU
        N = int(round(cpr * L_box / a))
        mesh = make_cartesian_mesh_3d(
            N, N, N, L_box, L_box, L_box,
            origin=(-L_box/2, -L_box/2, -L_box/2),
        )
        dx = mesh.cartesian_spacing[0]
        x = np.asarray(mesh.x)
        u_np, p_np = stokes_flow_around_sphere(x, U_inf=U_inf, a=a, mu=mu)
        u = jnp.asarray(u_np)
        p = jnp.asarray(p_np)

        def sdf(xq):
            return sphere_sdf(xq, center=jnp.zeros(3), radius=a)

        for shell in [(0.5, 2.5), (1.0, 3.0), (0.5, 4.0)]:
            F, _ = surface_integral_force(
                u, p, mesh, sdf, mu=mu, dx=dx,
                shell_inner=shell[0], shell_outer=shell[1],
            )
            err = abs(float(F[0]) - 6*np.pi*mu*a*U_inf) / (6*np.pi*mu*a*U_inf)
            print(f"    cpr={cpr} N={N} shell={shell}: F = {float(F[0]):.4e}  err={err*100:.1f}%")


if __name__ == "__main__":
    main()
