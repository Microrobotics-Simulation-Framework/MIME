# Quickstart

This guide gets you from `pip install` to a running MIME simulation in
about five minutes.

## 1. Install

```bash
pip install mime-engine     # CPU
# pip install "mime-engine" "jax[cuda12]"   # GPU
```

See the [Installation](installation.md) page for extras and source
installs.

## 2. The library shape

MIME is built on top of [MADDENING](https://microrobotica.org/maddening/)
and lives under the `mime` import root:

```text
mime
├── core/           # NodeMeta extensions, role enums, mime-specific node ABCs
├── nodes/
│   ├── actuation/      # Motors, robot arms
│   ├── environment/    # Fluids: csf_flow, stokeslet, lbm (IB-LBM), fvm (FVM-IBM)
│   ├── robot/          # Rigid bodies, magnetic response, helix geometry
│   ├── sensing/        # Sensor models
│   └── therapeutic/    # Drug-delivery / therapy nodes
├── control/        # Kinematics, controllers
└── experiments/    # Reference graphs (dejongh, dejongh_new_chain, …)
```

You compose a simulation by adding `mime` nodes to a MADDENING
`GraphManager`, wiring them with `add_edge`, then calling
`compile()` and `run_scan_with_history`.

## 3. Your first MIME graph

A minimal rotating-field → permanent-magnet → rigid-body chain. This
is a stripped-down version of the
[`dejongh_confined`](https://github.com/Microrobotics-Simulation-Framework/MIME/tree/main/experiments/dejongh_confined)
experiment.

```python
import jax.numpy as jnp
from maddening import GraphManager

from mime.nodes.environment.external_magnetic_field import (
    ExternalMagneticFieldNode,
)
from mime.nodes.robot.permanent_magnet_response import (
    PermanentMagnetResponseNode,
)
from mime.nodes.robot.rigid_body import RigidBodyNode

DT = 5e-4

gm = GraphManager()
gm.add_node(ExternalMagneticFieldNode(name="field", timestep=DT))
gm.add_node(PermanentMagnetResponseNode(
    name="magnet", timestep=DT,
    n_magnets=2, m_single=8.4e-4,
    moment_axis=(1.0, 0.0, 0.0),
))
gm.add_node(RigidBodyNode(
    name="body", timestep=DT,
    semi_major_axis_m=2.07e-3, semi_minor_axis_m=1.56e-3,
    density_kg_m3=1410.0,
    fluid_viscosity_pa_s=1e-3, fluid_density_kg_m3=1000.0,
))

gm.add_edge("field",  "magnet", "field_vector",   "field_vector")
gm.add_edge("field",  "magnet", "field_gradient", "field_gradient")
gm.add_edge("body",   "magnet", "orientation",    "orientation")
gm.add_edge("magnet", "body",   "magnetic_torque", "magnetic_torque")
gm.add_edge("magnet", "body",   "magnetic_force",  "magnetic_force")

gm.add_external_input("field", "frequency_hz",      shape=())
gm.add_external_input("field", "field_strength_mt", shape=())

gm.compile()
final, history = gm.run_scan_with_history(
    n_steps=2_000,
    external_inputs={
        "field": {
            "frequency_hz":      jnp.full((2_000,), 30.0),
            "field_strength_mt": jnp.full((2_000,), 10.0),
        },
    },
)

print(history["body"]["position"].shape)     # (2000, 3)
print(history["body"]["orientation"].shape)  # (2000, 4) — quaternion
```

## 4. Build the full benchmark graph

For the complete confined-helical-UMR benchmark there's a single-call
builder under `mime.experiments`:

```python
from mime.experiments.dejongh import build_graph

gm = build_graph(
    design_name="FL-9",
    vessel_name='1/4"',
    mu_Pa_s=1e-3,
    delta_rho=410.0,
    dt=5e-4,
    use_lubrication=True,
    lubrication_epsilon_mm=0.05,
)
gm.compile()
```

This is exactly the graph the
[`dejongh_confined`](https://github.com/Microrobotics-Simulation-Framework/MIME/tree/main/experiments/dejongh_confined)
experiment hands to the MICROROBOTICA IDE.

## 5. Differentiating through the simulation

The compiled graph is just an {term}`XLA` program — `jax.grad`,
`jax.vmap`, and `jax.jit` work as expected:

```python
import jax

def loss(frequency_hz):
    gm.set_node_state("body", {
        "position": jnp.zeros(3),
        "orientation": jnp.array([1.0, 0.0, 0.0, 0.0]),
    })
    n = 1000
    inputs = {"field": {
        "frequency_hz":      jnp.full((n,), frequency_hz),
        "field_strength_mt": jnp.full((n,), 10.0),
    }}
    final, _ = gm.run_scan_with_history(n_steps=n, external_inputs=inputs)
    return final["body"]["position"][2]  # z-displacement

print(jax.grad(loss)(30.0))
```

## 6. Run an experiment from the MICROROBOTICA IDE

MIME's experiment format (`experiment.yaml` + `physics/setup.py` +
`scene/world.usda`) is what
[MICROROBOTICA](https://microrobotica.org/) loads via
**File → Open Experiment**. See
[Using the libraries](https://microrobotica.org/user_guide/using_the_libraries.html)
for the IDE side.

## Next steps

- [Algorithm Guide](../algorithm_guide/defect_correction.md) — every
  node's governing equations, discretisation, and verification
  evidence.
- [`MIME/experiments/`](https://github.com/Microrobotics-Simulation-Framework/MIME/tree/main/experiments)
  — fully worked, end-to-end reference graphs.
- [MADDENING Quickstart](https://microrobotica.org/maddening/user_guide/quickstart.html)
  — the underlying graph runtime.
