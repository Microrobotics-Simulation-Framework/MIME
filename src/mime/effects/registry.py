"""EffectModel registry (ADR-2026-EFFECT-MODEL §6).

``@register_effect("Family.Backend")`` records a class under a canonical
name. The registry is the single source of truth for YAML ``type:`` string
resolution, MICROROBOTICA's effect-picker dropdowns, and serialisation
round-trips. A ``@register_effect`` decorator (not importlib auto-discovery)
keeps the surface auditable and avoids arbitrary-code config files.
"""

from __future__ import annotations

from typing import Callable, TypeVar

_REGISTRY: dict[str, type] = {}

T = TypeVar("T", bound=type)


def register_effect(name: str) -> Callable[[T], T]:
    """Class decorator registering an EffectModel under ``name``
    (e.g. ``"HydrodynamicModel.LBM"``)."""

    def _decorate(cls: T) -> T:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"effect name {name!r} is already registered to "
                f"{_REGISTRY[name].__name__}"
            )
        _REGISTRY[name] = cls
        cls._effect_name = name  # type: ignore[attr-defined]
        return cls

    return _decorate


def list_registered_effects() -> dict[str, type]:
    """Return a copy of the name → class registry."""
    return dict(_REGISTRY)


def get_effect(name: str) -> type:
    """Resolve a registered effect name to its class."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no effect registered under {name!r}; known: "
            f"{sorted(_REGISTRY)}"
        ) from None
