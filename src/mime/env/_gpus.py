"""The GPU types a run may request, as Modal names them, and how a device JAX reports is held to one.

A run that claims bits names one model of GPU. Modal serves some names with another model (``A100`` may run on an 80 GB
card, ``H100`` on an H200, ``B200+`` on a B300), so those names are refused and the exact ones are named instead. A
type's words are matched in JAX's ``device_kind`` as whole words (``A10`` is not in ``A100-SXM4-40GB``), case aside."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType

from mime.build import BuildRefused, Refusal, RefusalClass

#: Each type a run may request, with the words JAX's ``device_kind`` holds for it.
EXACT_GPUS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "T4": ("T4",),
    "L4": ("L4",),
    "A10": ("A10",),
    "L40S": ("L40S",),
    "RTX-PRO-6000": ("RTX", "PRO", "6000"),
    "A100-40GB": ("A100", "40GB"),
    "A100-80GB": ("A100", "80GB"),
    "H100!": ("H100",),
    "H200": ("H200",),
    "B200": ("B200",),
    "B300": ("B300",),
})
#: The names Modal may serve with another model, each with the exact types to request instead.
LOOSE_GPUS: Mapping[str, str] = MappingProxyType({
    "A100": "A100-40GB or A100-80GB",
    "H100": "H100!",
    "B200+": "B200 or B300",
})


def gpu_type(name: str) -> str:
    """``name`` when it is an exact type; refused as ``env.gpu-type`` otherwise, with the exact names to use."""
    if name in EXACT_GPUS:
        return name
    instead = LOOSE_GPUS.get(name) or LOOSE_GPUS.get(name.upper())
    if instead is not None:
        message = f"the GPU type {name!r} may be served by another model of GPU, so its results name no one machine"
        action = f"request {instead}"
    else:
        message = f"{name!r} is not a GPU type MIME knows as exact"
        action = f"request one of {', '.join(EXACT_GPUS)}"
    if ":" in name:
        message = f"{name!r} asks for several GPUs; a MIME run computes on one"
        action = f"request one of {', '.join(EXACT_GPUS)}, without a count"
    raise BuildRefused([Refusal("env.gpu-type", RefusalClass.USER, message, action)])


def is_kind_of(device_kind: str, name: str) -> bool:
    """Whether JAX's ``device_kind`` is a device of the exact type ``name``."""
    return all(re.search(rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", device_kind, re.IGNORECASE)
               for word in EXACT_GPUS[name])


__all__ = ["EXACT_GPUS", "LOOSE_GPUS", "gpu_type", "is_kind_of"]
