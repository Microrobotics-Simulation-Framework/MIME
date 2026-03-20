"""PolicyRunner — orchestrates GraphManager + ControlInput + UncertaintyModel.

The step loop:
  1. observed_state  = uncertainty.observe(true_state, t, rng_key)
  2. external_inputs = control_input.compute(t, dt, observed_state)
  3. applied_inputs  = uncertainty.actuate(external_inputs, t, rng_key)
  4. true_state      = graph.step(applied_inputs)

The controller is genuinely flying partially blind. The true state
evolves from applied_inputs. The controller only ever sees observed_state.

PolicyRunner is transport-agnostic — it depends on ControlInput (a protocol),
not on any specific policy class or network transport.

State publishing
----------------
PolicyRunner can optionally publish state to observers after each step.
This enables:
- MICROBOTICA's viewport to display live state
- A ZMQ publisher to stream state to a remote controller
- A ROS bridge to publish state as ROS messages
- A logger to record trajectories

The observer interface is a simple callable:
  ``(t, dt, true_state, observed_state, external_inputs, applied_inputs) -> None``
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from mime.control.input_source import ControlInput


# -- UncertaintyModel protocol --------------------------------------------

@runtime_checkable
class UncertaintyModel(Protocol):
    """Protocol for sensing and actuation uncertainty injection.

    Sits at the boundary between true simulation state and the controller.
    """

    def observe(
        self,
        true_state: dict[str, dict[str, Any]],
        t: float,
        rng_key: jax.Array,
    ) -> dict[str, dict[str, Any]]:
        """Apply sensing uncertainty to the true state.

        Returns the observed_state that the controller sees.
        """
        ...

    def actuate(
        self,
        commanded_inputs: dict[str, dict[str, Any]],
        t: float,
        rng_key: jax.Array,
    ) -> dict[str, dict[str, Any]]:
        """Apply actuation uncertainty to the commanded inputs.

        Returns the applied_inputs that the physics graph sees.
        """
        ...


class IdentityUncertainty:
    """Perfect sensing and actuation — no noise. Baseline."""

    def observe(
        self,
        true_state: dict[str, dict[str, Any]],
        t: float,
        rng_key: jax.Array,
    ) -> dict[str, dict[str, Any]]:
        return true_state

    def actuate(
        self,
        commanded_inputs: dict[str, dict[str, Any]],
        t: float,
        rng_key: jax.Array,
    ) -> dict[str, dict[str, Any]]:
        return commanded_inputs


# -- Step observer type ----------------------------------------------------

StepObserver = Callable[
    [
        float,                          # t
        float,                          # dt
        dict[str, dict[str, Any]],      # true_state
        dict[str, dict[str, Any]],      # observed_state
        dict[str, dict[str, Any]],      # external_inputs (commanded)
        dict[str, dict[str, Any]],      # applied_inputs (post-uncertainty)
    ],
    None,
]
"""Callback fired after each PolicyRunner step.

Used for state publishing to remote controllers, viewports, loggers, etc.
"""


# -- PolicyRunner ----------------------------------------------------------

@dataclass
class RunResult:
    """Result of a PolicyRunner.run() call."""
    final_state: dict[str, dict[str, Any]]
    n_steps: int
    wall_time_s: float
    sim_time_s: float


class PolicyRunner:
    """Orchestrates the observe-decide-actuate-step loop.

    Parameters
    ----------
    graph_manager : GraphManager
        MADDENING graph manager (compiled). PolicyRunner calls
        graph_manager.step(external_inputs) each timestep.
    control_input : ControlInput
        Source of external inputs — can be a local policy, remote
        controller, callback, or constant.
    uncertainty : UncertaintyModel, optional
        Sensing and actuation uncertainty injection.
        Default: IdentityUncertainty (perfect sensing/actuation).
    rng_seed : int
        JAX PRNG seed for uncertainty models.
    observers : list of StepObserver
        Callbacks fired after each step. For state publishing,
        logging, visualisation, etc.
    """

    def __init__(
        self,
        graph_manager: Any,  # GraphManager — Any to avoid hard import
        control_input: ControlInput,
        uncertainty: Optional[UncertaintyModel] = None,
        rng_seed: int = 0,
        observers: Optional[list[StepObserver]] = None,
    ) -> None:
        self.graph_manager = graph_manager
        self.control_input = control_input
        self.uncertainty = uncertainty or IdentityUncertainty()
        self.rng_key = jax.random.PRNGKey(rng_seed)
        self.observers: list[StepObserver] = list(observers or [])
        self._t: float = 0.0

    @property
    def t(self) -> float:
        """Current simulation time."""
        return self._t

    def step(self, dt: float) -> dict[str, dict[str, Any]]:
        """Execute one step of the observe-decide-actuate-step loop.

        Parameters
        ----------
        dt : float
            Timestep (used for time tracking; actual graph dt is set
            in GraphManager).

        Returns
        -------
        true_state : dict
            The graph state after stepping.
        """
        # Split RNG key
        self.rng_key, obs_key, act_key = jax.random.split(self.rng_key, 3)

        # 1. Get current true state
        true_state = self.graph_manager.get_state()

        # 2. Apply sensing uncertainty
        observed_state = self.uncertainty.observe(true_state, self._t, obs_key)

        # 3. Compute control commands
        external_inputs = self.control_input.compute(
            self._t, dt, observed_state
        )

        # 4. Apply actuation uncertainty
        applied_inputs = self.uncertainty.actuate(
            external_inputs, self._t, act_key
        )

        # 5. Step the physics graph
        true_state = self.graph_manager.step(applied_inputs)

        # 6. Advance time
        self._t += dt

        # 7. Notify observers
        for obs in self.observers:
            obs(
                self._t, dt,
                true_state, observed_state,
                external_inputs, applied_inputs,
            )

        return true_state

    def run(self, n_steps: int, dt: float) -> RunResult:
        """Run multiple steps.

        Parameters
        ----------
        n_steps : int
            Number of steps to execute.
        dt : float
            Timestep per step.

        Returns
        -------
        RunResult
            Final state, step count, wall time, simulation time.
        """
        t_start = time.monotonic()
        sim_t_start = self._t

        state = {}
        for _ in range(n_steps):
            state = self.step(dt)

        return RunResult(
            final_state=state,
            n_steps=n_steps,
            wall_time_s=time.monotonic() - t_start,
            sim_time_s=self._t - sim_t_start,
        )

    def reset(self, rng_seed: Optional[int] = None) -> None:
        """Reset the runner for a new episode.

        Parameters
        ----------
        rng_seed : int, optional
            New PRNG seed. If None, keeps the current key.
        """
        self._t = 0.0
        self.control_input.reset()
        if rng_seed is not None:
            self.rng_key = jax.random.PRNGKey(rng_seed)

    def add_observer(self, observer: StepObserver) -> None:
        """Add a step observer (for state publishing, logging, etc.)."""
        self.observers.append(observer)

    def remove_observer(self, observer: StepObserver) -> None:
        """Remove a step observer."""
        self.observers.remove(observer)
