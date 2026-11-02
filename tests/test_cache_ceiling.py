"""``tests/_cache_ceiling.py`` gives JAX's programs back when a worker passes its ceiling, and only then.

The control: with a headroom that the block's growth passes the caches are cleared (a program compiled before the
block is compiled again after it), and with one far over they are not, and a process that already stands over its
headroom when the block begins is not cleared for again and again. A ceiling that did nothing would pass a
test that never exceeds it. And the arm battery, whose ``params_effective`` check held 7.7 GB of programs, has the
fixture that holds it: a guard, with its own control."""

import ast
import time
from pathlib import Path

import _cache_ceiling
import jax
import jax.numpy as jnp
import numpy as np
import pytest

needs_proc = pytest.mark.skipif(_cache_ceiling.resident_mib() is None, reason="needs /proc/self/statm")


def compiles(function) -> int:
    seen: list[str] = []

    def listener(event: str, _duration: float, **_: object) -> None:
        seen.append(event)

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        function()
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)
    return sum(event.endswith("backend_compile_duration") for event in seen)


def wait_for(condition, bound: float = 30.0) -> bool:
    deadline = time.monotonic() + bound
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


def grow_by(mib: int) -> None:
    """Hold ``mib`` MiB of touched pages for a moment, as a test that grows does."""
    held = np.ones(mib * 2**17)  # float64: 8 bytes each
    time.sleep(0.5)  # the watcher looks every 0.01 s here
    del held


@needs_proc
def test_a_headroom_that_growth_passes_clears_the_caches_and_one_it_does_not_pass_does_not():
    @jax.jit
    def program(x):
        return jnp.sin(x) * 3.0 + jnp.cos(x) ** 2

    x = jnp.arange(7.0)
    program(x).block_until_ready()
    assert compiles(lambda: program(x).block_until_ready()) == 0  # warm: the program is held

    with _cache_ceiling.under_ceiling(10_000, poll=0.01) as roomy:
        program(x).block_until_ready()  # compiled again on entry, which gives back what earlier tests kept
        grow_by(200)
    assert roomy.cleared == 0
    assert compiles(lambda: program(x).block_until_ready()) == 0

    with _cache_ceiling.under_ceiling(50, poll=0.01, quiet=0.0) as tight:
        grow_by(200)
    assert tight.cleared >= 1
    assert compiles(lambda: program(x).block_until_ready()) >= 1  # the control: the program had to be compiled again


@needs_proc
def test_a_process_that_stays_over_its_headroom_is_not_cleared_for_again_and_again():
    held = np.ones(300 * 2**17)  # resident for the whole block
    with _cache_ceiling.under_ceiling(50, poll=0.01, quiet=0.0) as steady:
        time.sleep(0.5)
    del held
    assert steady.cleared == 0  # the limit is taken after the entry's clear, with the array already held


# -- the arm battery uses it ---------------------------------------------------------------------------------------------

HEADROOM_LIMIT_MIB = 1024  # a block's headroom over its worker's standing size; a worker stands at 0.6 to 1 GB


def holds_a_ceiling(source: str) -> bool:
    """Whether a module has an autouse fixture that calls ``under_ceiling`` with a module constant at or under
    ``HEADROOM_LIMIT_MIB`` (so that its tests stay under the memory_heavy line of 2 GiB)."""
    tree = ast.parse(source)
    constants = {target.id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float))
                 for target in node.targets if isinstance(target, ast.Name)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not any(
                isinstance(d, ast.Call) and "autouse" in {k.arg for k in d.keywords} for d in node.decorator_list):
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "under_ceiling" and call.args:
                argument = call.args[0]
                value = constants.get(argument.id) if isinstance(argument, ast.Name) else (
                    argument.value if isinstance(argument, ast.Constant) else None)
                if isinstance(value, (int, float)) and value <= HEADROOM_LIMIT_MIB:
                    return True
    return False


def test_the_arm_battery_holds_its_worker_under_the_ceiling():
    """``params_effective`` held 7.7 GB of compiled programs at its end; this fails a battery whose fixture is gone,
    or whose headroom is more than a worker with this file's other tests can spare."""
    source = (Path(__file__).parent / "oracles" / "test_arm_dynamics_battery.py").read_text(encoding="utf-8")
    assert holds_a_ceiling(source)


def test_control_the_guard_of_the_ceiling_sees_a_missing_fixture_and_a_headroom_that_is_too_large():
    fixture = ("import _cache_ceiling\nimport pytest\nHEADROOM_MIB = {value}\n@pytest.fixture(autouse=True)\n"
               "def kept():\n    with _cache_ceiling.under_ceiling(HEADROOM_MIB):\n        yield\n")
    assert holds_a_ceiling(fixture.format(value=1024))
    assert not holds_a_ceiling(fixture.format(value=9000))
    assert not holds_a_ceiling(fixture.format(value=1024).replace("autouse=True", ""))
    assert not holds_a_ceiling("import pytest\ndef test_a(): ...\n")
