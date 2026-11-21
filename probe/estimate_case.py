"""Memory declaration for the numpy/scipy arms of w6s (sparse modal factors, dense film proxies; not a JAX program): a held
block of the declared peak."""
HELD = []


def held_case(gib: float = 3.0):
    import jax
    import jax.numpy as jnp
    import numpy as np
    HELD.append(np.ones(int(float(gib) * 2 ** 30 / 8)))
    return jax.jit(lambda x: x + 1.0), (jnp.zeros(8),)
