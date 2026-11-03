"""The tube's integrated image (``wall_refinement=2``) is read in chunks cut for ``TABLE_PAIR_VALUES`` floating values a
pair of its kernel (``tube_image``: both reads and their blend), and XLA's account of the kernel's program holds far
fewer: the figure is declared, and this measures it (the profiling pass's stage 2b, design #205).

- **The kernel holds at most an eighth of the declared figure a pair** beside its result: between 960 points on a
  body-sized cylinder 0.3 mm above the floor of the 1/4 in tube (FL-9's size, its count at 0.4 mm) and 2, 32 and 128
  of them as sources, in float32 and in float64 (measured 48 to 64 values a pair; what a pair holds does not depend on
  where the points are). So a chunk holds an eighth to a quarter of ``CHUNK_BYTES``, never more: the declared figure
  is safe, and loose by that factor, which is what ``TABLE_PAIR_VALUES``' comment states.
- **Control**, failing on every run: a kernel that also holds 64 windows of nine values a pair at once, as a read that
  gathered all its corners together would, measures above the bound, so the measure can see a kernel that holds more
  than the bound allows."""

import math
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.hydro.mrs import frame_across
from mime.nodes.hydro.mrs._tube import TABLE_PAIR_VALUES, shipped_unit_tube, tube_image

#: The tube (the 1/4 in tube along +x) and the body's cylinder in it: FL-9's radius and length, its centre 0.8 mm below
#: the axis, so that its lowest points lie 0.3 mm above the floor and every read is inside the table.
TUBE_RADIUS, BODY_RADIUS, BODY_LENGTH, OFFSET = 3.175e-3, 2.07e-3, 7.47e-3, 0.8e-3
SOURCES = (2, 32, 128)
#: The most a pair of the kernel may hold beside its result, against the declared figure.
BOUND = TABLE_PAIR_VALUES // 8


def placed(dtype: Any) -> tuple[Any, dict[str, Any]]:
    """960 points on the body's cylinder (24 around, 40 along), and the tube's constants, in ``dtype``."""
    around = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    along = np.linspace(-0.5, 0.5, 40) * BODY_LENGTH
    a, x = np.meshgrid(around, along, indexing="ij")
    points = np.stack([x.ravel(), BODY_RADIUS * np.cos(a.ravel()), BODY_RADIUS * np.sin(a.ravel()) - OFFSET], axis=1)
    parameters = {"centre": (0.0, 0.0, 0.0), "frame": frame_across((1.0, 0.0, 0.0)), "radius": TUBE_RADIUS,
                  "table": shipped_unit_tube().arrays(dtype)}
    return jnp.asarray(points, dtype=dtype), parameters


def per_pair(kernel: Any, points: Any, sources: Any, parameters: dict[str, Any]) -> float:
    """XLA's temporaries of ``kernel``'s program, in floating values of the points' dtype a pair."""
    analysed = cast(Any, jax.jit(kernel).lower(points, sources, parameters).compile()).memory_analysis()
    return float(analysed.temp_size_in_bytes) / (points.shape[0] * sources.shape[0] * points.dtype.itemsize)


def image(targets: Any, sources: Any, parameters: dict[str, Any]) -> Any:
    return tube_image(targets, sources, parameters, 1e-3).matrix


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("count", SOURCES)
def test_the_kernel_holds_at_most_an_eighth_of_the_declared_values_a_pair(dtype, count):
    points, parameters = placed(jnp.dtype(dtype))
    held = per_pair(image, points, points[:count], parameters)
    print(f"\n{dtype}, {points.shape[0]} targets and {count} sources: {held:.1f} values a pair, declared "
          f"{TABLE_PAIR_VALUES}")
    assert 0.0 < held <= BOUND


def test_control_a_kernel_holding_every_corner_window_at_once_measures_above_the_bound():
    points, parameters = placed(jnp.float64)

    def every_corner(targets: Any, sources: Any, parameters: dict[str, Any]) -> Any:
        found = image(targets, sources, parameters)
        windows = jnp.broadcast_to(found.reshape(*found.shape[:2], 9)[..., None, :] * jnp.arange(1.0, 65.0)[:, None],
                                   (*found.shape[:2], 64, 9))
        return found + jnp.sum(jnp.cumsum(windows, axis=2), axis=(2, 3))[..., None, None]

    held = per_pair(every_corner, points, points[:32], parameters)
    print(f"\nevery corner's window held at once: {held:.1f} values a pair")
    assert held > BOUND
    assert np.isfinite(held)
