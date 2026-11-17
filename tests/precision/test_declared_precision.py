"""The precision an experiment declares (``Experiment(..., precision=...)``): float64 when not stated, float32 declared;
JAX's 64-bit types enabled by the build in both, and ``run_dtype()`` read from the declaration, not JAX's flag.

**Zero wide operations.** Script A and the settling ball built in float32, with 64-bit types enabled, step with no
float64, int64, uint64 or complex128 operation anywhere in their step (the dtype audit, ``mime.testing._dtype_audit``),
and no state field changes dtype in a step. Before the shared widening lines were fixed the same audit read 742 wide
operations of 1521 in script A's step and 398 of 1028 in the ball's (the parameter leaves in float64, ``jnp.eye``'s
and ``jnp.diag``'s int64 ``iota``, a float64 timestep, the MRS's float64 build-time blocks reading float32).

**Controls**, each failing on every run (the plants are deterministic, so one run is the rate): the parameter leaves
left in float64 (MADDENING makes a Python float JAX's canonical float64) are read as wide operations and as drifted
state, and the run's state still keeps float32 (the step's fields narrowed back where they came in); an int64 ``iota``
planted in the rigid body's mass matrix (``jnp.eye``) is read as wide operations."""

from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.build import BuildRefused
from mime.core import run_dtype
from mime.core._precision import FROM_JAX_FLAG
from mime.experiment import Experiment
from mime.scene import Scene
from mime.testing._dtype_audit import audit_graph
from mime.units import ms, s

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def example(name: str, precision: str | None) -> Experiment:
    sys.path.insert(0, str(EXAMPLES))
    try:
        experiment = getattr(importlib.import_module(name), name)()
    finally:
        sys.path.remove(str(EXAMPLES))
    experiment._settings = dataclasses.replace(experiment._settings, precision=precision)  # noqa: SLF001
    return experiment


def state_dtypes(built: Any) -> set[str]:
    return {str(value.dtype) for node, fields in built.current_state().items() if node != "_meta"
            for value in fields.values() if np.asarray(value).dtype.kind == "f"}


def test_an_experiment_that_states_no_precision_runs_in_float64(monkeypatch):
    monkeypatch.delenv(FROM_JAX_FLAG)
    jax.config.update("jax_enable_x64", False)
    built = example("script_a", None).build()
    built.run(s(10 * built.plans.dt))
    assert jax.config.jax_enable_x64
    assert run_dtype() == np.float64
    assert state_dtypes(built) == {"float64"}
    assert built.record.environment["x64"] is True


def test_a_declared_float32_runs_in_float32_with_64_bit_types_enabled():
    jax.config.update("jax_enable_x64", False)
    built = example("script_a", "float32").build()
    assert jax.config.jax_enable_x64, "the build enables 64-bit types whatever the precision"
    assert run_dtype() == np.float32, "run_dtype reads the declared precision, not JAX's flag"
    built.run(s(10 * built.plans.dt))
    assert state_dtypes(built) == {"float32"}
    assert built.record.environment["x64"] is False, "the lock records the declared precision"


def test_a_precision_mime_does_not_run_in_is_refused():
    experiment = Experiment("p", scene=Scene.new(up=(0, 0, 1), units="m"), dt=ms(1), precision="float16")  # type: ignore[arg-type]
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    assert "run.precision" in refused.value.codes()


@pytest.mark.parametrize("name", ["script_a", "sphere_settling"])
def test_a_float32_run_with_64_bit_types_enabled_holds_no_wide_operation(name):
    built = example(name, "float32").build()
    audit = audit_graph(name, built.graph, built.held_inputs())
    print("\n" + audit.text())
    assert audit.total > 0
    assert (audit.wide_count, audit.drifted) == (0, ())


def test_control_float64_parameter_leaves_are_read_wide_and_drifted_and_the_run_keeps_float32(monkeypatch):
    import mime.core._node as node

    monkeypatch.setattr(node, "_leaves_in", lambda precision, tree: tree)  # the leaves as MADDENING makes them
    built = example("script_a", "float32").build()
    audit = audit_graph("script_a", built.graph, built.held_inputs())
    assert audit.wide_count > 0 and audit.by_dtype().get("float64", 0) > 0
    assert any(field.endswith("float32->float64") for field in audit.drifted)
    built.run(s(10 * built.plans.dt))
    assert state_dtypes(built) == {"float32"}


def test_control_an_int64_iota_in_the_mass_matrix_is_read_wide(monkeypatch):
    import mime.nodes.body._rigid_inertial as rigid

    monkeypatch.setattr(rigid, "eye", lambda n, dtype: jnp.eye(n, dtype=dtype))
    built = example("script_a", "float32").build()
    assert audit_graph("script_a", built.graph, built.held_inputs()).by_dtype().get("int64", 0) > 0


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_a_saved_experiment_loads_in_the_precision_its_lock_records(monkeypatch, tmp_path, precision):
    """The lock's ``x64`` records the declared precision; an old lock's ``x64: false`` is float32, as it was: a load
    declares what the lock holds (float32 here is exactly the lock a float32-flag run wrote before precision was declared)."""
    built = example("script_a", precision).build()
    built.save(tmp_path / "saved", steps=2)
    assert built.record.environment["x64"] is (precision == "float64")
    monkeypatch.delenv(FROM_JAX_FLAG)
    jax.config.update("jax_enable_x64", False)
    loaded = Experiment.load(tmp_path / "saved").build()
    assert run_dtype() == np.dtype(precision)
    assert loaded.record.environment["x64"] is (precision == "float64")
    loaded.run(s(2 * loaded.plans.dt))
    assert state_dtypes(loaded) == {precision}


def test_a_controller_whose_start_takes_jax_s_flag_for_its_dtype_builds_in_float32():
    """Its start is float64 under 64-bit types (``jnp.zeros(()).dtype``), its step returns the run's float32: refused
    ``controller.untraceable`` before the start was narrowed, as JAX narrowed it before precision was declared."""
    from allocation_cases import allocating_sphere

    experiment = allocating_sphere()
    experiment._settings = dataclasses.replace(experiment._settings, precision="float32")  # noqa: SLF001
    built = experiment.build()
    built.run(s(2 * built.plans.dt))
    assert state_dtypes(built) == {"float32"}


def test_a_wide_operation_in_a_plugin_s_node_is_credited_to_its_own_file(monkeypatch):
    """A node file outside MIME (a plugin's; here this file) is the origin the audit names for the plant, not MIME's
    ``MimeNode`` wrapper that called it (where it was credited before: the first frame with ``/mime/`` in its path).
    The nodes after it widen at their own lines, which the audit names too. The plant is deterministic, so one run is
    its rate."""
    import mime.nodes.body._gravity as gravity

    held = gravity.GravityNode.advance

    def planted(self: Any, p: Any, state: Any, boundary_inputs: Any, dt: Any) -> Any:
        out = dict(held(self, p, state, boundary_inputs, dt))
        out["wrench"] = out["wrench"] + jnp.zeros(out["wrench"].shape)  # PLANTED: dtype-less, float64
        return out

    monkeypatch.setattr(gravity.GravityNode, "advance", planted)
    built = example("script_a", "float32").build()
    origins = {line for (_, dtypes, line) in audit_graph("script_a", built.graph, built.held_inputs()).origins
               if "float64" in dtypes}
    assert any(line.split(":")[0].endswith(Path(__file__).name) for line in origins), origins
