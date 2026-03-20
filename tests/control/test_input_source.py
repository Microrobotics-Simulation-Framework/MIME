"""Tests for ControlInput implementations — local, remote, callback, constant."""

import queue
import threading
import time

import pytest

from mime.control.input_source import (
    CallbackControlInput,
    ConstantControlInput,
    ControlInput,
    PolicyControlInput,
    RemoteControlInput,
)
from mime.control.policy import ControlPolicy, ExternalInputs, PolicyState


# -- Test helpers ----------------------------------------------------------

class IncrementingPolicy(ControlPolicy):
    """Returns increasing values each call. For testing state evolution."""

    def __init__(self, node: str = "field", field: str = "freq"):
        self.node = node
        self.field = field

    def __call__(self, t, observed_state, policy_state):
        val = policy_state.get("value", 0.0) + 1.0
        return (
            {self.node: {self.field: val}},
            {"value": val},
        )

    def initial_policy_state(self):
        return {"value": 0.0}


# -- PolicyControlInput tests ---------------------------------------------

class TestPolicyControlInput:
    def test_implements_protocol(self):
        pci = PolicyControlInput(IncrementingPolicy())
        assert isinstance(pci, ControlInput)

    def test_compute_delegates_to_policy(self):
        pci = PolicyControlInput(IncrementingPolicy())
        ext = pci.compute(0.0, 0.001, {})
        assert ext == {"field": {"freq": 1.0}}

    def test_state_persists_across_calls(self):
        pci = PolicyControlInput(IncrementingPolicy())
        pci.compute(0.0, 0.001, {})
        pci.compute(1.0, 0.001, {})
        ext = pci.compute(2.0, 0.001, {})
        assert ext == {"field": {"freq": 3.0}}

    def test_reset_clears_state(self):
        pci = PolicyControlInput(IncrementingPolicy())
        pci.compute(0.0, 0.001, {})
        pci.compute(1.0, 0.001, {})
        pci.reset()
        ext = pci.compute(0.0, 0.001, {})
        assert ext == {"field": {"freq": 1.0}}

    def test_policy_state_readable(self):
        pci = PolicyControlInput(IncrementingPolicy())
        pci.compute(0.0, 0.001, {})
        assert pci.policy_state == {"value": 1.0}

    def test_passes_observed_state_to_policy(self):
        class EchoPolicy(ControlPolicy):
            def __call__(self, t, obs, ps):
                val = obs.get("robot", {}).get("pos", -1.0)
                return {"field": {"echo": val}}, ps
            def initial_policy_state(self):
                return {}

        pci = PolicyControlInput(EchoPolicy())
        ext = pci.compute(0.0, 0.001, {"robot": {"pos": 42.0}})
        assert ext["field"]["echo"] == 42.0


# -- RemoteControlInput tests ---------------------------------------------

class TestRemoteControlInput:
    def test_implements_protocol(self):
        rci = RemoteControlInput()
        assert isinstance(rci, ControlInput)

    def test_push_and_compute(self):
        rci = RemoteControlInput(timeout_s=1.0)
        rci.push_command({"field": {"freq": 10.0}})
        ext = rci.compute(0.0, 0.001, {})
        assert ext == {"field": {"freq": 10.0}}

    def test_zero_order_hold(self):
        """If no new command, repeat the last one."""
        rci = RemoteControlInput(timeout_s=0.1)
        rci.push_command({"field": {"freq": 10.0}})
        rci.compute(0.0, 0.001, {})  # consumes the command
        # Second call — no new command in queue
        ext = rci.compute(1.0, 0.001, {})
        assert ext == {"field": {"freq": 10.0}}

    def test_takes_latest_command(self):
        """If multiple commands queued, take the most recent."""
        rci = RemoteControlInput(timeout_s=0.1)
        rci.push_command({"field": {"freq": 10.0}})
        rci.push_command({"field": {"freq": 20.0}})
        rci.push_command({"field": {"freq": 30.0}})
        ext = rci.compute(0.0, 0.001, {})
        assert ext == {"field": {"freq": 30.0}}

    def test_timeout_hold_returns_empty_on_first_call(self):
        """on_timeout='hold' with no commands returns empty dict."""
        rci = RemoteControlInput(timeout_s=0.01, on_timeout="hold")
        ext = rci.compute(0.0, 0.001, {})
        assert ext == {}

    def test_timeout_raise(self):
        rci = RemoteControlInput(timeout_s=0.01, on_timeout="raise")
        with pytest.raises(TimeoutError):
            rci.compute(0.0, 0.001, {})

    def test_timeout_empty(self):
        rci = RemoteControlInput(timeout_s=0.01, on_timeout="empty")
        ext = rci.compute(0.0, 0.001, {})
        assert ext == {}

    def test_reset_clears_state_and_queue(self):
        rci = RemoteControlInput(timeout_s=0.1)
        rci.push_command({"field": {"freq": 10.0}})
        rci.compute(0.0, 0.001, {})
        rci.push_command({"field": {"freq": 99.0}})
        rci.reset()
        assert rci.command_queue.empty()
        # After reset, first call should block again (timeout)
        rci2 = RemoteControlInput(
            command_queue=rci.command_queue, timeout_s=0.01, on_timeout="empty"
        )
        ext = rci2.compute(0.0, 0.001, {})
        assert ext == {}

    def test_staleness_callback(self):
        stale_ages = []
        rci = RemoteControlInput(
            timeout_s=0.01,
            on_timeout="hold",
            stale_threshold_s=0.01,
            on_stale=lambda age: stale_ages.append(age),
        )
        rci.push_command({"field": {"freq": 10.0}})
        rci.compute(0.0, 0.001, {})
        time.sleep(0.02)
        rci.compute(1.0, 0.001, {})
        assert len(stale_ages) == 1
        assert stale_ages[0] >= 0.01

    def test_threaded_push(self):
        """Simulate a transport adapter pushing from another thread."""
        rci = RemoteControlInput(timeout_s=2.0)
        received = []

        def producer():
            time.sleep(0.05)
            rci.push_command({"field": {"freq": 42.0}})

        t = threading.Thread(target=producer)
        t.start()

        ext = rci.compute(0.0, 0.001, {})
        received.append(ext)
        t.join()

        assert received[0] == {"field": {"freq": 42.0}}

    def test_high_frequency_producer(self):
        """Producer pushes commands faster than consumer reads."""
        rci = RemoteControlInput(timeout_s=0.5)
        for i in range(100):
            rci.push_command({"field": {"freq": float(i)}})

        ext = rci.compute(0.0, 0.001, {})
        # Should get the last one (99.0)
        assert ext == {"field": {"freq": 99.0}}

    def test_custom_queue(self):
        """User can provide their own queue."""
        q: queue.Queue = queue.Queue()
        q.put({"field": {"freq": 7.0}})
        rci = RemoteControlInput(command_queue=q, timeout_s=0.1)
        ext = rci.compute(0.0, 0.001, {})
        assert ext == {"field": {"freq": 7.0}}


# -- CallbackControlInput tests -------------------------------------------

class TestCallbackControlInput:
    def test_implements_protocol(self):
        cci = CallbackControlInput(lambda t, dt, obs: {})
        assert isinstance(cci, ControlInput)

    def test_delegates_to_callback(self):
        def my_cb(t, dt, obs):
            return {"field": {"freq": t * 10.0}}
        cci = CallbackControlInput(my_cb)
        ext = cci.compute(3.0, 0.001, {})
        assert ext == {"field": {"freq": 30.0}}

    def test_receives_observed_state(self):
        def my_cb(t, dt, obs):
            return {"field": {"echo": obs.get("robot", {}).get("x", 0.0)}}
        cci = CallbackControlInput(my_cb)
        ext = cci.compute(0.0, 0.001, {"robot": {"x": 5.0}})
        assert ext["field"]["echo"] == 5.0

    def test_receives_dt(self):
        def my_cb(t, dt, obs):
            return {"field": {"dt_echo": dt}}
        cci = CallbackControlInput(my_cb)
        ext = cci.compute(0.0, 0.042, {})
        assert ext["field"]["dt_echo"] == 0.042


# -- ConstantControlInput tests -------------------------------------------

class TestConstantControlInput:
    def test_implements_protocol(self):
        cci = ConstantControlInput({"field": {"freq": 10.0}})
        assert isinstance(cci, ControlInput)

    def test_returns_same_every_time(self):
        commands = {"field": {"freq": 10.0, "strength": 5.0}}
        cci = ConstantControlInput(commands)
        for t in [0.0, 1.0, 100.0]:
            assert cci.compute(t, 0.001, {}) == commands

    def test_reset_is_noop(self):
        cci = ConstantControlInput({"field": {"freq": 10.0}})
        cci.reset()  # Should not raise
        assert cci.compute(0.0, 0.001, {}) == {"field": {"freq": 10.0}}
