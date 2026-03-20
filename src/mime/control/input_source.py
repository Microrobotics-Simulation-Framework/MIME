"""ControlInput protocol and implementations — transport-agnostic command source.

The ControlInput protocol decouples PolicyRunner from the source of
external inputs. Commands can come from:
- A local ControlPolicy (PolicyControlInput)
- A remote source via a queue (RemoteControlInput — for ZMQ, ROS, WebSocket)
- A plain callable (CallbackControlInput — simplest integration point)
- A constant dict (ConstantControlInput — for testing)

PolicyRunner calls control_input.compute(t, dt, observed_state) each step
and gets back an ExternalInputs dict. It does not know or care where the
commands came from.

Remote control architecture
---------------------------
When the physics graph runs on a cloud GPU (via MADDENING's CloudSession)
and the controller runs locally or on another machine:

    ┌───────────────────────┐          ┌───────────────────────┐
    │  Local / ROS / UI     │          │  Cloud GPU             │
    │                       │  ZMQ     │                        │
    │  ControlPolicy ──────>├─commands─>│  RemoteControlInput    │
    │                       │          │        │                │
    │  observed_state <─────├──state───│  PolicyRunner           │
    │                       │          │        │                │
    │                       │          │  GraphManager.step()   │
    └───────────────────────┘          └───────────────────────┘

The transport layer (ZMQ, ROS topic, gRPC) is NOT part of this module.
This module provides RemoteControlInput which reads from a
threading.Queue — the transport adapter pushes into that queue.
This keeps the control layer transport-agnostic and testable without
any network infrastructure.
"""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from mime.control.policy import ControlPolicy, ExternalInputs, PolicyState


@runtime_checkable
class ControlInput(Protocol):
    """Protocol for anything that produces external inputs each timestep.

    This is the interface PolicyRunner depends on. Implement it to
    integrate any command source — local policy, remote controller,
    hardware-in-the-loop, ROS subscriber, joystick, etc.
    """

    def compute(
        self,
        t: float,
        dt: float,
        observed_state: dict[str, dict[str, Any]],
    ) -> ExternalInputs:
        """Produce external inputs for this timestep.

        Parameters
        ----------
        t : float
            Current simulation time.
        dt : float
            Timestep duration.
        observed_state : dict
            Graph state as seen through the uncertainty model.

        Returns
        -------
        ExternalInputs
            Commands for GraphManager.step().
        """
        ...

    def reset(self) -> None:
        """Reset internal state (e.g., policy_state) for a new episode."""
        ...


class PolicyControlInput:
    """Wraps a ControlPolicy as a ControlInput.

    This is the standard local-policy case. The policy runs in the
    same process as the simulation, with no network hop.
    """

    def __init__(self, policy: ControlPolicy) -> None:
        self.policy = policy
        self._policy_state = policy.initial_policy_state()

    def compute(
        self,
        t: float,
        dt: float,
        observed_state: dict[str, dict[str, Any]],
    ) -> ExternalInputs:
        ext_inputs, self._policy_state = self.policy(
            t, observed_state, self._policy_state
        )
        return ext_inputs

    def reset(self) -> None:
        self._policy_state = self.policy.initial_policy_state()

    @property
    def policy_state(self) -> PolicyState:
        """Read-only access to the current policy state (for logging/debugging)."""
        return self._policy_state


class RemoteControlInput:
    """Receives external inputs from a remote source via a thread-safe queue.

    The transport adapter (ZMQ subscriber, ROS subscriber, WebSocket handler,
    gRPC stream, etc.) pushes ExternalInputs dicts into the command_queue.
    PolicyRunner calls compute() each timestep and gets the most recent
    command. If no command has arrived since the last call, the previous
    command is repeated (zero-order hold).

    Parameters
    ----------
    command_queue : queue.Queue
        Thread-safe queue that the transport adapter pushes into.
    timeout_s : float
        Maximum time to wait for a command on the first call.
        Subsequent calls use non-blocking reads (zero-order hold).
    on_timeout : str
        What to do if no command arrives within timeout_s:
        - "hold": use the last command (or empty if none yet)
        - "raise": raise TimeoutError
        - "empty": return empty ExternalInputs
    stale_threshold_s : float
        If the most recent command is older than this, fire the
        on_stale callback (if provided). 0.0 disables staleness check.
    on_stale : callable, optional
        ``(age_seconds: float) -> None`` — called when command is stale.
    """

    def __init__(
        self,
        command_queue: Optional[queue.Queue[ExternalInputs]] = None,
        timeout_s: float = 1.0,
        on_timeout: str = "hold",
        stale_threshold_s: float = 0.0,
        on_stale: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.command_queue: queue.Queue[ExternalInputs] = (
            command_queue or queue.Queue()
        )
        self.timeout_s = timeout_s
        self.on_timeout = on_timeout
        self.stale_threshold_s = stale_threshold_s
        self.on_stale = on_stale
        self._last_command: ExternalInputs = {}
        self._last_command_time: float = 0.0
        self._first_call = True

    def compute(
        self,
        t: float,
        dt: float,
        observed_state: dict[str, dict[str, Any]],
    ) -> ExternalInputs:
        # Drain the queue — keep only the most recent command
        latest: Optional[ExternalInputs] = None
        try:
            if self._first_call:
                # Block on first call to wait for the remote controller
                latest = self.command_queue.get(timeout=self.timeout_s)
                self._first_call = False
            # Drain any remaining (non-blocking)
            while True:
                latest = self.command_queue.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            self._last_command = latest
            self._last_command_time = time.monotonic()
        elif self._first_call:
            # No command received within timeout on first call
            self._first_call = False
            if self.on_timeout == "raise":
                raise TimeoutError(
                    f"No remote command received within {self.timeout_s}s"
                )
            elif self.on_timeout == "empty":
                return {}
            # "hold" falls through to return self._last_command

        # Staleness check
        if (
            self.stale_threshold_s > 0
            and self._last_command_time > 0
            and self.on_stale is not None
        ):
            age = time.monotonic() - self._last_command_time
            if age > self.stale_threshold_s:
                self.on_stale(age)

        return self._last_command

    def reset(self) -> None:
        self._last_command = {}
        self._last_command_time = 0.0
        self._first_call = True
        # Drain queue
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break

    def push_command(self, command: ExternalInputs) -> None:
        """Convenience method to push a command from the same process.

        In production, the transport adapter calls command_queue.put()
        directly. This method is for testing and simple integrations.
        """
        self.command_queue.put(command)


class CallbackControlInput:
    """Wraps a plain callable as a ControlInput.

    The simplest integration point — pass any function that takes
    (t, dt, observed_state) and returns ExternalInputs.

    Useful for quick experiments, Jupyter notebooks, and bridging
    to external control frameworks that don't use MIME's ControlPolicy ABC.
    """

    def __init__(
        self,
        callback: Callable[
            [float, float, dict[str, dict[str, Any]]],
            ExternalInputs,
        ],
    ) -> None:
        self.callback = callback

    def compute(
        self,
        t: float,
        dt: float,
        observed_state: dict[str, dict[str, Any]],
    ) -> ExternalInputs:
        return self.callback(t, dt, observed_state)

    def reset(self) -> None:
        pass  # No internal state


class ConstantControlInput:
    """Returns the same ExternalInputs every timestep. For testing."""

    def __init__(self, commands: ExternalInputs) -> None:
        self.commands = commands

    def compute(
        self,
        t: float,
        dt: float,
        observed_state: dict[str, dict[str, Any]],
    ) -> ExternalInputs:
        return self.commands

    def reset(self) -> None:
        pass
