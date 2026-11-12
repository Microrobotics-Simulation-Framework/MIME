"""The raw-program estimate of the time probe's heaviest program: the reference's walled step at the fine cell (48^3)."""
import sys
sys.path.insert(0, "tests"); sys.path.insert(0, "tests/patch")


def factory():
    import jax, jax.numpy as jnp
    import numpy as np
    from time_cases import Box, TimeCase, _walled
    from patch_cases import mrs_sphere
    mrs_sphere(0.25)
    case = TimeCase(viscosity=0.05, dt=5e-4, gap=0.3)
    box = Box(case.box * case.ratio, case.fine, case.slab * case.ratio, case.viscosity, 1260.0 / case.dt,
              np.zeros((338, 3)) + case.box * case.coarse / 2)
    rhs = box.grid.zeros(jnp.float64)
    return (lambda r, g: _walled(box.grid, box.stopping, box.mu, box.shift, box.sigma, r, g)), (rhs, rhs)
