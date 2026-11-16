# SPDX-License-Identifier: Apache-2.0
"""A surface field's colours: the view maps a recorded physics value through one colour map over a range (the field's
declared range, else the recording's), and the page draws the same map as its legend. Colour is presentation: the
stage and the recording hold the values.

The map is viridis (van der Walt and Smith, matplotlib; CC0), held as seventeen sRGB stops and joined by straight lines
between them: perceptually ordered, readable in grey and by the common colour-vision deficiencies. glTF's vertex
colours are linear, so a colour is written linear (:func:`linear`) and the viewer turns it back to sRGB; the light on
the surface shades it, so a colour on screen matches the legend only where the surface faces the light."""

from __future__ import annotations

import numpy as np

COLOUR_MAP = "viridis"
#: viridis at 0, 1/16, ..., 1, in sRGB, from matplotlib's table.
STOPS: tuple[tuple[float, float, float], ...] = (
    (0.267, 0.0049, 0.3294), (0.2823, 0.095, 0.4173), (0.2788, 0.1755, 0.4834), (0.259, 0.2515, 0.5247),
    (0.2297, 0.3224, 0.5457), (0.1994, 0.3876, 0.5546), (0.1727, 0.4488, 0.5579), (0.149, 0.5081, 0.5573),
    (0.1276, 0.5669, 0.5506), (0.1206, 0.6258, 0.5335), (0.1579, 0.6838, 0.5017), (0.2461, 0.7389, 0.452),
    (0.3692, 0.7889, 0.3829), (0.516, 0.8312, 0.2943), (0.6785, 0.8637, 0.1895), (0.8456, 0.8873, 0.0997),
    (0.9932, 0.9062, 0.1439))


def srgb(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """``values`` mapped over ``[low, high]`` (clipped) to sRGB colours, one row of three each."""
    span = high - low if high > low else 1.0
    t = np.clip((np.asarray(values, dtype=np.float64) - low) / span, 0.0, 1.0)
    stops = np.asarray(STOPS, dtype=np.float64)
    where = t * (len(stops) - 1)
    below = np.minimum(np.floor(where).astype(np.int64), len(stops) - 2)
    weight = (where - below)[..., None]
    return stops[below] * (1.0 - weight) + stops[below + 1] * weight


def linear(colours: np.ndarray) -> np.ndarray:
    """sRGB colours in linear light, as glTF holds a vertex colour."""
    c = np.asarray(colours, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def css_gradient() -> str:
    """The map as a CSS gradient from left (low) to right (high)."""
    stops = ", ".join(f"rgb({round(r * 255)}, {round(g * 255)}, {round(b * 255)}) {k / (len(STOPS) - 1):.2%}"
                      for k, (r, g, b) in enumerate(STOPS))
    return f"linear-gradient(to right, {stops})"


__all__ = ["COLOUR_MAP", "STOPS", "css_gradient", "linear", "srgb"]
