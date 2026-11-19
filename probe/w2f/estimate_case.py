"""The memory proxy of CORE spike 2's arms, needing neither gmsh nor pyamg (the runner installs them after the estimate):
one explicit momentum stage and a Jacobi CG of the pressure operator on a bodiless periodic tube of structured
triangles revolved to at least the arm's cell count, with the first guess's history (2 x 8 fields) among its arguments.
The multigrid set-up and the mesher are not in it, so a measured peak may read above this estimate."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "c2"))
sys.path.insert(0, os.path.dirname(HERE))


def step_case(cells: int = 240000, sectors: int = 24):
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    import flow as fl
    import tube
    from gfv import amg, prims
    radius, length = 1.75e-3, 14e-3
    n2 = int(cells) // int(sectors)
    nr = max(4, int(round((n2 / 2 * radius / length) ** 0.5)))
    nz = max(4, n2 // (2 * nr) + 1)
    r, z = np.linspace(0.0, radius, nr + 1), np.linspace(-length / 2, length / 2, nz + 1)
    R, Z = np.meshgrid(r, z, indexing="ij")
    nodes = np.stack([R.ravel(), Z.ravel()], axis=1)
    idx = np.arange((nr + 1) * (nz + 1)).reshape(nr + 1, nz + 1)
    tris = np.array([t for i in range(nr) for j in range(nz)
                     for t in ((idx[i, j], idx[i + 1, j], idx[i + 1, j + 1]), (idx[i, j], idx[i + 1, j + 1], idx[i, j + 1]))])
    m = tube.revolved(nodes, tris, int(sectors), radius, length, 0.4 * radius, 1.0)
    F = fl.Flow(m, "scatter")
    tab = prims.as_device(F.tab, jnp.float64)
    E = {k: jnp.asarray(v) for k, v in amg.ell(F.A).items()}
    dinv = jnp.asarray(1.0 / F.A.diagonal())
    par = {"nu": jnp.asarray(3.3e-6), "dt": jnp.asarray(1e-6), "force": jnp.zeros(3)}

    def program(tab, E, dinv, u, flux, hist):
        R_, fc = F.rhs(tab, par, u, flux, jnp.zeros(()))
        b = -F._net(tab["net"], fc) + hist[0, 0] - hist[1, 0]
        x, r, k = fl.pcg(lambda v: amg.matvec(E, v), lambda q: dinv * q, b - jnp.mean(b), 1e-12, 50)
        return u + R_, x, hist

    return program, (tab, E, dinv, jnp.zeros((m.n, 3)), jnp.zeros(F.m), jnp.zeros((2, 8, m.n)))


def mac_case(cells: int = 32):
    """One step of MIME's resolved-body node on W2F's MAC box (``cells`` across the tube's diameter)."""
    import math

    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    from mime.nodes.flow.fvm import band_named
    from mime.nodes.flow.fvm._resolved import window_cells
    from mime.nodes.flow.fvm._resolved_node import ResolvedFlowNode, ResolvedSpec
    cells = int(cells)
    radius = 1.75e-3
    a = 0.4 * radius
    h = 2 * radius / cells
    nx, nz = cells + cells // 2, 4 * cells
    band = band_named("smooth_2h_0.3h")
    spec = ResolvedSpec("ball", "drag:ball", "sphere", {"centre": (0.0, 0.0, 0.0), "radius": a}, (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), window_cells(a, h, band), 4.0 / 3.0 * math.pi * a ** 3)
    c = 0.5 * nx * h
    node = ResolvedFlowNode("flow", 1e-5, cells_x=nx, cells_y=nx, cells_z=nz, spacing=h, viscosity=3.3e-6 * 1060.0, density=1060.0,
                            tube_centre=(c, c, 0.0), tube_axis=(0.0, 0.0, 1.0), tube_radius=radius, scheme="imex_euler",
                            convection=True, bodies=[spec], initial_velocity_z=np.zeros((nx, nx, nz)))
    mass = 1060.0 * 4.0 / 3.0 * math.pi * a ** 3
    inputs = {"body_0_pose": jnp.asarray([c, c, 0.5 * nz * h, 1.0, 0.0, 0.0, 0.0]), "body_0_twist": jnp.zeros(6),
              "body_0_mass": jnp.asarray(np.diag([mass] * 3 + [0.4 * mass * a ** 2] * 3)), "body_force": jnp.asarray([0.0, 0.0, 1.0])}
    state = dict(node.initial_state())

    def program(state, inputs):
        return jax.lax.scan(lambda s, _: ({**s, **node.update(s, inputs, 1e-5)}, None), state, None, length=2)[0]

    return program, (state, inputs)
