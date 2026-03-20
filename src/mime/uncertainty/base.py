"""UncertaintyModel ABC and concrete implementations.

All models are pure functions of (state_or_inputs, t, rng_key) — they
carry configuration but no mutable state, making them JAX-friendly and
serialisable.

Composition: model_a + model_b creates a ComposedUncertainty that
applies both in sequence (observe: a then b, actuate: a then b).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp


# -- Type alias for graph state dicts --------------------------------------

GraphState = dict[str, dict[str, Any]]
"""Structure: {node_name: {field_name: jax_array, ...}, ...}"""


# -- ABC -------------------------------------------------------------------

class UncertaintyModel(ABC):
    """Abstract base class for sensing and actuation uncertainty.

    Sits at the boundary between true simulation state and the controller.
    Two channels:
    - observe(true_state, t, rng_key) -> observed_state
    - actuate(commanded_inputs, t, rng_key) -> applied_inputs

    Subclasses implement _observe and _actuate. The public methods
    handle RNG key splitting.
    """

    @abstractmethod
    def _observe(
        self, true_state: GraphState, t: float, rng_key: jax.Array,
    ) -> GraphState:
        ...

    @abstractmethod
    def _actuate(
        self, commanded: GraphState, t: float, rng_key: jax.Array,
    ) -> GraphState:
        ...

    def observe(
        self, true_state: GraphState, t: float, rng_key: jax.Array,
    ) -> GraphState:
        """Apply sensing uncertainty to the true state."""
        return self._observe(true_state, t, rng_key)

    def actuate(
        self, commanded: GraphState, t: float, rng_key: jax.Array,
    ) -> GraphState:
        """Apply actuation uncertainty to the commanded inputs."""
        return self._actuate(commanded, t, rng_key)

    def __add__(self, other: UncertaintyModel) -> ComposedUncertainty:
        """Compose: (a + b) applies a then b on both channels."""
        models = []
        for m in (self, other):
            if isinstance(m, ComposedUncertainty):
                models.extend(m.models)
            else:
                models.append(m)
        return ComposedUncertainty(models=tuple(models))


# -- Identity (no noise) --------------------------------------------------

class IdentityUncertainty(UncertaintyModel):
    """Perfect sensing and actuation — no noise. Baseline."""

    def _observe(self, true_state, t, rng_key):
        return true_state

    def _actuate(self, commanded, t, rng_key):
        return commanded


# -- Actuation uncertainty -------------------------------------------------

@dataclass(frozen=True)
class ActuationUncertainty(UncertaintyModel):
    """Noise on actuation commands before they reach the physics graph.

    Models real-world imperfections in the actuation hardware:
    - Motor encoder jitter (frequency noise)
    - Field inhomogeneity (field strength noise)
    - Robotic arm positioning error (pointing noise)
    - Slow thermal drift on frequency

    Parameters
    ----------
    frequency_noise_std : float
        Std dev of Gaussian noise on frequency commands (Hz).
    field_strength_noise_std : float
        Std dev of Gaussian noise on field strength commands (mT).
    pointing_error_std : float
        Std dev of Gaussian noise on field direction angles (rad).
    frequency_drift_rate : float
        Linear drift rate on frequency (Hz/s). Accumulates over time.
    target_node : str
        Name of the actuation node whose inputs are perturbed.
    """
    frequency_noise_std: float = 0.0
    field_strength_noise_std: float = 0.0
    pointing_error_std: float = 0.0
    frequency_drift_rate: float = 0.0
    target_node: str = "external_field"

    def _observe(self, true_state, t, rng_key):
        return true_state  # Actuation uncertainty doesn't affect sensing

    def _actuate(self, commanded, t, rng_key):
        if self.target_node not in commanded:
            return commanded

        result = dict(commanded)
        node_inputs = dict(commanded[self.target_node])
        k1, k2, k3 = jax.random.split(rng_key, 3)

        if "frequency_hz" in node_inputs:
            perturbation = 0.0
            if self.frequency_noise_std > 0:
                perturbation = perturbation + jax.random.normal(k1) * self.frequency_noise_std
            if self.frequency_drift_rate != 0.0:
                perturbation = perturbation + self.frequency_drift_rate * t
            if not isinstance(perturbation, float) or perturbation != 0.0:
                node_inputs["frequency_hz"] = node_inputs["frequency_hz"] + perturbation

        if "field_strength_mt" in node_inputs and self.field_strength_noise_std > 0:
            noise = jax.random.normal(k2) * self.field_strength_noise_std
            node_inputs["field_strength_mt"] = node_inputs["field_strength_mt"] + noise

        if "field_direction" in node_inputs and self.pointing_error_std > 0:
            noise = jax.random.normal(k3, shape=(3,)) * self.pointing_error_std
            node_inputs["field_direction"] = node_inputs["field_direction"] + noise

        result[self.target_node] = node_inputs
        return result


# -- Localisation uncertainty ----------------------------------------------

@dataclass(frozen=True)
class LocalisationUncertainty(UncertaintyModel):
    """Noise on state observations — what the controller sees.

    Models limitations of the localisation/imaging system:
    - Position noise (MRI resolution, ultrasound speckle)
    - Velocity noise (derived from noisy position)
    - Tracking dropouts (frames where the robot is lost)
    - Tracking confidence field injected into observed state

    Parameters
    ----------
    position_noise_std_mm : float
        Std dev of Gaussian position noise (mm).
    velocity_noise_std : float
        Std dev of Gaussian velocity noise (mm/s).
    dropout_probability : float
        Probability [0,1] that tracking is lost on any given step.
        During dropout, last known position is held (zero-order hold).
    target_node : str
        Name of the node whose state is observed noisily.
    position_field : str
        Name of the position field in the target node's state.
    velocity_field : str
        Name of the velocity field in the target node's state.
    """
    position_noise_std_mm: float = 0.0
    velocity_noise_std: float = 0.0
    dropout_probability: float = 0.0
    target_node: str = "robot"
    position_field: str = "position"
    velocity_field: str = "velocity"

    def _observe(self, true_state, t, rng_key):
        if self.target_node not in true_state:
            return true_state

        result = dict(true_state)
        node_state = dict(true_state[self.target_node])
        k1, k2, k3 = jax.random.split(rng_key, 3)

        # Dropout: with some probability, don't update (hold last known)
        # We model this by zeroing out the noise and flagging confidence=0
        is_tracking = jax.random.uniform(k3) >= self.dropout_probability
        tracking_confidence = jnp.where(is_tracking, 1.0, 0.0)

        # Position noise (only applied when tracking)
        if self.position_field in node_state and self.position_noise_std_mm > 0:
            pos = node_state[self.position_field]
            noise = jax.random.normal(k1, shape=pos.shape) * self.position_noise_std_mm
            noisy_pos = pos + jnp.where(is_tracking, noise, jnp.zeros_like(noise))
            node_state[self.position_field] = noisy_pos

        # Velocity noise (only applied when tracking)
        if self.velocity_field in node_state and self.velocity_noise_std > 0:
            vel = node_state[self.velocity_field]
            noise = jax.random.normal(k2, shape=vel.shape) * self.velocity_noise_std
            noisy_vel = vel + jnp.where(is_tracking, noise, jnp.zeros_like(noise))
            node_state[self.velocity_field] = noisy_vel

        # Inject tracking confidence
        node_state["tracking_confidence"] = tracking_confidence

        result[self.target_node] = node_state
        return result

    def _actuate(self, commanded, t, rng_key):
        return commanded  # Localisation uncertainty doesn't affect actuation


# -- Model uncertainty -----------------------------------------------------

@dataclass(frozen=True)
class ModelUncertainty(UncertaintyModel):
    """Fractional noise on observed state fields.

    Models parameter uncertainty: patient-to-patient variability,
    fabrication tolerances, environmental fluctuations. Applied as
    multiplicative noise: observed = true * (1 + noise).

    For full model UQ (ensemble over parameter distributions), use
    MADDENING's GraphManager.run_sweep() via jax.vmap instead.

    Parameters
    ----------
    noise_fraction : float
        Std dev of multiplicative Gaussian noise. 0.1 means ±10%.
    target_nodes : tuple of str
        Which nodes' state fields to perturb. Empty = all nodes.
    exclude_fields : tuple of str
        Field names to never perturb (e.g., discrete flags).
    """
    noise_fraction: float = 0.0
    target_nodes: tuple[str, ...] = ()
    exclude_fields: tuple[str, ...] = ("tracking_confidence",)

    def _observe(self, true_state, t, rng_key):
        if self.noise_fraction <= 0:
            return true_state

        result = {}
        key_idx = 0
        keys = jax.random.split(rng_key, max(len(true_state) * 10, 1))

        for node_name, fields in true_state.items():
            if self.target_nodes and node_name not in self.target_nodes:
                result[node_name] = fields
                continue

            noisy_fields = {}
            for field_name, value in fields.items():
                if field_name in self.exclude_fields:
                    noisy_fields[field_name] = value
                    continue
                if hasattr(value, 'shape'):
                    k = keys[key_idx % len(keys)]
                    key_idx += 1
                    noise = jax.random.normal(k, shape=value.shape) * self.noise_fraction
                    noisy_fields[field_name] = value * (1.0 + noise)
                else:
                    noisy_fields[field_name] = value
            result[node_name] = noisy_fields

        return result

    def _actuate(self, commanded, t, rng_key):
        return commanded  # Model uncertainty affects observation, not actuation


# -- Composed uncertainty --------------------------------------------------

class ComposedUncertainty(UncertaintyModel):
    """Applies multiple uncertainty models in sequence.

    Created via the + operator: ``model_a + model_b + model_c``.
    Observe: applies a, then b, then c to the state.
    Actuate: applies a, then b, then c to the commands.
    """

    def __init__(self, models: tuple[UncertaintyModel, ...]) -> None:
        self.models = models

    def _observe(self, true_state, t, rng_key):
        keys = jax.random.split(rng_key, len(self.models))
        state = true_state
        for model, key in zip(self.models, keys):
            state = model.observe(state, t, key)
        return state

    def _actuate(self, commanded, t, rng_key):
        keys = jax.random.split(rng_key, len(self.models))
        inputs = commanded
        for model, key in zip(self.models, keys):
            inputs = model.actuate(inputs, t, key)
        return inputs
