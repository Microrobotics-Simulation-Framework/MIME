"""A raw estimate for the bits probe: script A's floor arm built (the heaviest configuration's resident set dominates)."""
import sys
sys.path[:0] = ["tests", "tests/saved"]


def factory():
    import jax, jax.numpy as jnp
    from switch_configurations import configurations
    configurations()["script_a/floor"]().build()
    return jax.jit(lambda x: x + 1.0), (jax.ShapeDtypeStruct((4,), jnp.float32),)
