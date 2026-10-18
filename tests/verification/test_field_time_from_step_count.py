"""ExternalMagneticFieldNode takes its time from the step count, not a float32 accumulator.

Spike 1 (MIME 0.3.0, "where it hurt" item 14): ``sim_time += dt`` in float32
runs the field's clock at a rate set by how dt rounds to the spacing of t:
+4.0e-4 fast for t in [4, 8) s at dt = 5e-4, which moved FL-9's float32 swim
speed by 3.4e-4 against x64.  Computed from the step count, t carries one
rounding, not ten thousand.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.actuation.external_magnetic_field import ExternalMagneticFieldNode

DT = 5e-4
N = 10_000                                  # 5 s, FL-9's run


def _run(n):
    node = ExternalMagneticFieldNode("field", DT)
    bi = {"frequency_hz": jnp.float32(10.0), "field_strength_mt": jnp.float32(1.2)}
    step = jax.jit(lambda s: node.update(s, bi, DT))
    return jax.lax.fori_loop(0, n, lambda _, s: step(s), node.initial_state())


def test_time_after_n_steps_is_n_dt_to_one_rounding():
    state = _run(N)
    t = np.asarray(state["sim_time"])
    exact = N * DT
    # One float32 rounding of N*dt (plus dt's own rounding): a few 1e-8 relative.
    assert abs(float(t) - exact) / exact < 2e-7, (float(t), exact)


def test_field_phase_after_n_steps_matches_the_drive():
    state = _run(N)
    B = np.asarray(state["field_vector"], dtype=np.float64)
    phase = np.arctan2(B[1], B[0])
    expected = np.angle(np.exp(1j * 2 * np.pi * 10.0 * N * DT))
    assert abs(np.angle(np.exp(1j * (phase - expected)))) < 1e-4
