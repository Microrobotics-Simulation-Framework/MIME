"""The raw estimate for the operating-point probe's largest program: the cost arm's forward product at 120^3 (1.7 M
cells, 338 points, the cell read, every interior face's own field)."""
import sys
sys.path[:0] = ["tests", "tests/patch", "probe"]


def factory():
    import math
    import jax, jax.numpy as jnp
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    from common_op import A, REGIMES, sphere
    from mime.nodes.flow.fvm._patch import RefinementPatch, patch_reflection, patch_stopping, placed, snapped_origin
    from mime.nodes.flow.fvm._patch_transfers import Quadrature
    from mime.nodes.flow.fvm._solve import REPORT_SIZE
    n, R = 120, 2.5e-3
    mu, shift = REGIMES["l079"]

    class Tube:
        region = "wall"

        def signed_distance(self, points):
            return jnp.sqrt(points[..., 1] ** 2 + points[..., 2] ** 2) - R

    s = sphere(mu)
    centre = np.array([0.0, 0.0, -(R - A * 1.1)])
    patch = RefinementPatch((n, n, n), A / 20.0, read="cell")
    q = Quadrature(jnp.asarray(s["points"] + centre), jnp.asarray(s["weights"]), jnp.asarray(s["owners"], dtype=jnp.int32),
                   jnp.asarray(s["area"]))
    placement = placed(patch, [Tube()], snapped_origin(patch, jnp.asarray(centre)), 0, jnp.asarray(s["force"] + centre),
                       jnp.asarray(s["eps2"]), mu, shift=shift, quadrature=q)
    grid = patch.grid()
    stopping = patch_stopping(patch, jnp.float64)

    def run(p, b0, b1, b2, f):
        return patch_reflection(patch, p, (b0, b1, b2), f, None, stopping, jnp.zeros(REPORT_SIZE)).values

    args = (placement,) + tuple(jax.ShapeDtypeStruct((grid.given_count(c),), jnp.float64) for c in range(3)) + (
        jax.ShapeDtypeStruct((338, 3), jnp.float64),)
    return jax.jit(run), args
