"""ControlPolicy ABC, ControlPrimitive ABC, ControlSequence, SequentialPolicy.

Policies are stateless classes — all mutable state lives in a policy_state
dict passed in and returned each call. This makes them JAX-friendly
(no hidden state on self) and serialisable (the policy_state dict can be
checkpointed).

A policy never touches GraphManager. It only sees
(t, observed_state, policy_state) and returns
(external_inputs, new_policy_state).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# -- Type aliases ----------------------------------------------------------

ExternalInputs = dict[str, dict[str, Any]]
"""Mapping of {node_name: {field_name: value}} for GraphManager.step()."""

PolicyState = dict[str, Any]
"""Opaque mutable state carried across policy calls."""


# -- ControlPolicy ABC ----------------------------------------------------

class ControlPolicy(ABC):
    """Abstract base class for feedback control policies.

    Subclasses implement __call__ as a pure function of
    (t, observed_state, policy_state). Constructor params are
    configuration only (gains, target node names) — nothing mutable.

    Policies operate on observed_state (post-uncertainty), never on
    the true simulation state.
    """

    @abstractmethod
    def __call__(
        self,
        t: float,
        observed_state: dict[str, dict[str, Any]],
        policy_state: PolicyState,
    ) -> tuple[ExternalInputs, PolicyState]:
        """Compute external inputs given observed state.

        Parameters
        ----------
        t : float
            Current simulation time in seconds.
        observed_state : dict
            Full graph state as seen through the UncertaintyModel.
            Structure: {node_name: {field_name: jax_array, ...}, ...}.
        policy_state : dict
            Mutable policy state carried across calls.

        Returns
        -------
        external_inputs : ExternalInputs
            Commands for GraphManager.step().
        new_policy_state : PolicyState
            Updated policy state for next call.
        """
        ...

    @abstractmethod
    def initial_policy_state(self) -> PolicyState:
        """Return the initial policy state dict."""
        ...

    def __or__(self, other: ControlPolicy) -> SequentialPolicy:
        """Compose two policies: ``policy_a | policy_b``.

        The resulting SequentialPolicy calls both in order and
        merges their external_inputs dicts (later policy wins on
        conflicts for the same node/field).
        """
        policies = []
        # Flatten nested SequentialPolicy chains
        for p in (self, other):
            if isinstance(p, SequentialPolicy):
                policies.extend(p.policies)
            else:
                policies.append(p)
        return SequentialPolicy(policies=tuple(policies))


class SequentialPolicy(ControlPolicy):
    """Composes multiple policies called in sequence.

    External inputs are merged left-to-right — later policies
    override earlier ones for the same node/field.
    """

    def __init__(self, policies: tuple[ControlPolicy, ...]) -> None:
        self.policies = policies

    def __call__(
        self,
        t: float,
        observed_state: dict[str, dict[str, Any]],
        policy_state: PolicyState,
    ) -> tuple[ExternalInputs, PolicyState]:
        merged_inputs: ExternalInputs = {}
        sub_states = policy_state.get("_sub_states", [{}] * len(self.policies))
        new_sub_states = []

        for i, policy in enumerate(self.policies):
            ext, new_ps = policy(t, observed_state, sub_states[i])
            # Merge: later policies override on conflict
            for node_name, fields in ext.items():
                if node_name not in merged_inputs:
                    merged_inputs[node_name] = {}
                merged_inputs[node_name].update(fields)
            new_sub_states.append(new_ps)

        return merged_inputs, {"_sub_states": new_sub_states}

    def initial_policy_state(self) -> PolicyState:
        return {
            "_sub_states": [p.initial_policy_state() for p in self.policies]
        }


# -- ControlPrimitive ABC -------------------------------------------------

@dataclass
class ControlPrimitive(ABC):
    """A scripting atom — a fixed-duration open-loop command.

    Primitives are composed into ControlSequences. They do not carry
    policy_state — they are stateless descriptors of time-bounded actions.
    """
    duration: float
    target_node: str

    @abstractmethod
    def external_inputs(
        self, t_local: float, dt: float, observed_state: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Compute the boundary inputs for target_node at local time t_local.

        Parameters
        ----------
        t_local : float
            Time since this primitive started (0 to duration).
        dt : float
            Simulation timestep.
        observed_state : dict
            Full graph state (for reactive primitives).

        Returns
        -------
        dict[str, Any]
            Field values for the target node's boundary inputs.
        """
        ...

    def on_start(self, observed_state: dict[str, dict[str, Any]]) -> None:
        """Hook called when this primitive becomes active."""

    def on_end(self, observed_state: dict[str, dict[str, Any]]) -> None:
        """Hook called when this primitive completes."""


# -- ControlSequence -------------------------------------------------------

class ControlSequence(ControlPolicy):
    """Composes ControlPrimitives into a time-ordered script.

    Wraps primitives as a ControlPolicy so it can be used with
    PolicyRunner or composed with | operator.

    Parameters
    ----------
    primitives : list of ControlPrimitive
        Executed in order. Total duration = sum of individual durations.
    loop : bool
        If True, restart from the first primitive after the last completes.
    """

    def __init__(
        self,
        primitives: list[ControlPrimitive],
        loop: bool = False,
    ) -> None:
        self.primitives = list(primitives)
        self.loop = loop
        self._total_duration = sum(p.duration for p in self.primitives)
        # Precompute cumulative start times
        self._start_times: list[float] = []
        t = 0.0
        for p in self.primitives:
            self._start_times.append(t)
            t += p.duration

    @property
    def total_duration(self) -> float:
        return self._total_duration

    def _active_primitive_index(self, elapsed: float) -> int | None:
        """Return the index of the active primitive, or None if past end."""
        if self._total_duration <= 0:
            return None

        if self.loop:
            elapsed = elapsed % self._total_duration
        elif elapsed >= self._total_duration:
            return None

        for i in range(len(self.primitives) - 1, -1, -1):
            if elapsed >= self._start_times[i]:
                return i
        return 0

    def __call__(
        self,
        t: float,
        observed_state: dict[str, dict[str, Any]],
        policy_state: PolicyState,
    ) -> tuple[ExternalInputs, PolicyState]:
        t_start = policy_state.get("sequence_start_t", t)
        prev_index = policy_state.get("active_index", -1)
        elapsed = t - t_start

        idx = self._active_primitive_index(elapsed)
        if idx is None:
            # Past end and not looping — return empty
            return {}, policy_state

        prim = self.primitives[idx]

        # Fire lifecycle hooks on primitive transitions
        if idx != prev_index:
            if 0 <= prev_index < len(self.primitives):
                self.primitives[prev_index].on_end(observed_state)
            prim.on_start(observed_state)

        t_local = elapsed - self._start_times[idx]
        if self.loop:
            t_local = (elapsed % self._total_duration) - self._start_times[idx]

        dt = policy_state.get("dt", 0.001)
        fields = prim.external_inputs(t_local, dt, observed_state)

        ext_inputs: ExternalInputs = {prim.target_node: fields}
        new_state = {
            "sequence_start_t": t_start,
            "active_index": idx,
            "dt": dt,
        }
        return ext_inputs, new_state

    def initial_policy_state(self) -> PolicyState:
        return {
            "sequence_start_t": None,  # Set on first call
            "active_index": -1,
            "dt": 0.001,
        }
