# MIME Rendering Implementation Plan

This plan details the implementation of the viewport system described in
ARCHITECTURE_PLAN.md §15. It covers two deliverables: `PyVistaViewport`
(local development) and `HydraStormViewport` (production headless/cloud).

---

## Architecture Recap

```
Simulation Loop (JAX)                    Rendering Pipeline
┌─────────────────────┐                  ┌─────────────────────────┐
│ GraphManager.step()  │                  │                         │
│        │             │                  │  USDViewport.render()   │
│        ▼             │   state→USD      │        │                │
│  JAX state dicts ────┼──────────────►   │  Read USD stage         │
│                      │                  │        │                │
│  PolicyRunner        │                  │  Convert to renderable  │
│        │             │                  │        │                │
│        ▼             │                  │  Produce pixels         │
│  StepObserver ───────┼───── notify ──►  │        │                │
│                      │                  │  Return HxWx3 uint8     │
└─────────────────────┘                  └─────────────────────────┘
```

Key insight: the simulation loop and the rendering pipeline are decoupled
via the USD stage. The simulation writes to the stage; the viewport reads
it. They share no other state.

---

## Component 1: State-to-USD Bridge (`src/mime/viz/stage_bridge.py`)

Before either viewport can render, simulation state must be written to a
USD stage. This bridge converts JAX state dicts to USD prim attributes.

### What it does

- Creates an in-memory USD stage (`Usd.Stage.CreateInMemory()`)
- For each node with position/orientation state, creates a USD Xform prim
- Each timestep, updates prim transforms from state dicts
- Supports parametric geometry (sphere, ellipsoid, cylinder from GeometrySource)
- Does NOT own the simulation — it's a passive observer (StepObserver callback)

### Dynamic vs. static geometry separation

The `StageBridge` handles **dynamic state only** — quantities that change every timestep (robot position, orientation, field vector). Static anatomy (the cylindrical tube in the demo, or a Neurobotika ventricular mesh for B4-T2) is loaded **once** into the stage as a USD reference or sublayer, not re-written each frame. This separation maps naturally to USD's composition architecture:

- **Static anatomy**: loaded via `stage.GetRootLayer().subLayerPaths.append(mesh_path)` or `prim.GetReferences().AddReference(mesh_usd)` at setup time. Never modified by `StageBridge.update()`.
- **Dynamic robot**: `UsdGeom.Xform` with `xformOp:translate` + `xformOp:orient` updated each timestep.
- **Dynamic field glyph**: single `UsdGeom.Mesh` arrow with `xformOp:orient` + `xformOp:scale` updated each timestep.

This separation is critical for performance (static geometry is never re-traversed by Hydra) and for the eventual Neurobotika mesh integration (the mesh is a static USD reference, not streamed through the state bridge).

### API

```python
class StageBridge:
    """Writes simulation state to a live USD stage each timestep."""

    def __init__(self, stage: Optional[Usd.Stage] = None):
        """If no stage provided, creates an in-memory one."""

    @property
    def stage(self) -> Usd.Stage:
        """The live USD stage."""

    def register_robot(self, node_name: str, geometry: GeometrySource,
                       prim_path: str = "/World/Robot") -> None:
        """Register a robot body for visualisation."""

    def register_field(self, node_name: str,
                       prim_path: str = "/World/Field") -> None:
        """Register an external field for vector visualisation."""

    def add_static_geometry(self, geometry: GeometrySource,
                            prim_path: str = "/World/Channel") -> None:
        """Add static environment geometry (tube, mesh) — loaded once."""

    def update(self, state: dict) -> None:
        """Write current dynamic state to USD stage.

        Performance note: use Sdf.ChangeBlock() to batch attribute
        writes when updating multiple prims in a single timestep.
        """

    def as_observer(self) -> StepObserver:
        """Return a PolicyRunner StepObserver callback."""
```

### Implementation approach

- Option (a) from ARCHITECTURE_PLAN.md: in-memory stage, updated each step
- Robot body: `UsdGeom.Xform` with `xformOp:translate` + `xformOp:orient`
- Parametric geometry: `UsdGeom.Sphere`, `UsdGeom.Capsule`, `UsdGeom.Cylinder`
- Field visualisation: single `UsdGeom.Mesh` arrow transformed via `xformOp:orient` (direction) and `xformOp:scale` (magnitude). One prim, not two — reduces stage traversal cost and simplifies instancing if multiple sample points are needed later.
- Camera: `UsdGeom.Camera` at configurable position
- **Performance**: use `Sdf.ChangeBlock()` context manager to batch all per-timestep attribute writes into a single stage notification. This prevents the stage from sending change notices to Hydra for each individual attribute write.

### Dependencies

- `usd-core` (OpenUSD Python bindings) — `pip install usd-core`
- Only needed for visualisation, not for physics simulation

---

## Component 2: `PyVistaViewport` (`src/mime/viz/pyvista_viewport.py`)

### What it does

1. Reads the live USD stage from `StageBridge`
2. Traverses `UsdGeom` prims and converts to VTK `PolyData` meshes
3. Renders via PyVista's offscreen plotter
4. Returns HxWx3 uint8 numpy array

### Why PyVista and not raw VTK

- PyVista wraps VTK with a clean Python API
- Offscreen rendering via OSMesa works without a display server
- Good enough for development; replaced by HydraStorm for production

### Key implementation details

- `UsdGeom.Sphere` → `pyvista.Sphere(radius, center)`
- `UsdGeom.Capsule` → `pyvista.Cylinder` (approximate — see note below)
- `UsdGeom.Cylinder` → `pyvista.Cylinder` (for tube geometry)
- `UsdGeom.Xform` transform → PyVista actor transform matrix
- Arrow mesh for magnetic field vector
- Offscreen: `pyvista.Plotter(off_screen=True)`

**Capsule approximation note**: `UsdGeom.Capsule` is approximated as a cylinder. None of the current robot geometries use Capsule — the prolate ellipsoid body maps to a scaled `pyvista.ParametricEllipsoid()` instead. If Capsule geometry is needed in the future, PyVista's tube filter (`pyvista.Tube`) provides hemispherical end caps and would be a better match.

### Performance

The main performance concern for PyVistaViewport is the USD→VTK geometry conversion. **Geometry meshes must be created once at registration time and only actor transforms updated each frame.** The per-frame loop should:

1. For each registered dynamic prim: read `xformOp:translate` and `xformOp:orient` from USD
2. Update the corresponding PyVista actor's transform matrix
3. Render

It should NOT: re-traverse the full stage, re-create PyVista meshes, or re-add actors to the plotter each frame. This keeps the per-frame cost to O(N_dynamic_prims) attribute reads + one VTK render call.

### Test plan

- Unit test: create stage with sphere, render, check image is non-black
- Unit test: move sphere, render two frames, check images differ
- Integration: run 10 simulation steps, render each, verify all finite

---

## Component 3: `HydraStormViewport` (`src/mime/viz/hydra_viewport.py`)

### What it does

1. Creates an EGL headless OpenGL context (no X11 required)
2. Initialises `pxr.UsdImagingGL.Engine` (Hydra Storm)
3. Renders the live USD stage directly — full USD materials, lighting
4. Reads the framebuffer into a numpy array
5. Returns HxWx3 uint8

### Why Hydra Storm

- USD's native renderer — handles composition, variants, references
- Used by NVIDIA Omniverse and all USD-native tools
- Reads the stage directly (no geometry conversion)
- Incremental scene updates via dirty-bit tracking — only changed prims are re-processed
- The framebuffer feeds the Selkies WebRTC transport layer

### EGL headless setup

The target EGL path is `EGL_PLATFORM=surfaceless` (or `EGL_PLATFORM=device`) with a real NVIDIA/AMD GPU driver. This avoids all X11/Wayland dependencies.

```python
import ctypes
import os
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

from OpenGL import EGL

# 1. Get default display (surfaceless — no window system)
display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
EGL.eglInitialize(display, None, None)

# 2. Choose config for offscreen rendering
config_attribs = [
    EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
    EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
    EGL.EGL_NONE,
]
configs = (EGL.EGLConfig * 1)()
EGL.eglChooseConfig(display, config_attribs, configs, 1, ctypes.byref(ctypes.c_int()))

# 3. Create pbuffer surface
surface = EGL.eglCreatePbufferSurface(display, configs[0], [
    EGL.EGL_WIDTH, width, EGL.EGL_HEIGHT, height, EGL.EGL_NONE
])

# 4. Create context
EGL.eglBindAPI(EGL.EGL_OPENGL_API)
context = EGL.eglCreateContext(display, configs[0], EGL.EGL_NO_CONTEXT, None)
EGL.eglMakeCurrent(display, surface, surface, context)
```

### Hydra rendering

```python
from pxr import UsdImagingGL, Gf, Glf

engine = UsdImagingGL.Engine()
engine.SetRendererAov("color")

params = UsdImagingGL.RenderParams()
params.frame = Usd.TimeCode.Default()

# Set camera from stage camera prim
camera = stage.GetPrimAtPath("/Camera")
engine.SetCameraState(...)

# Render
engine.Render(stage.GetPseudoRoot(), params)

# Read pixels
pixels = Glf.ReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
```

### Performance

Frame latency directly affects WebRTC stream quality. Key bottlenecks:

1. **Hydra scene sync**: Hydra Storm tracks dirty bits — only prims with changed attributes are re-processed. The `StageBridge.update()` + `Sdf.ChangeBlock()` pattern ensures minimal dirty-prim count per frame.

2. **Framebuffer readback**: `glReadPixels` is a GPU→CPU synchronisation point — the CPU blocks until the GPU finishes rendering. For streaming at 30+ FPS, this is the dominant bottleneck. **Mitigation**: use PBO (Pixel Buffer Object) double-buffering to pipeline the readback — start reading frame N-1 while rendering frame N. This is a Phase 2 optimisation; the initial implementation uses synchronous readback.

3. **Stage attribute writes**: `StageBridge.update()` writes attributes from the simulation thread. Wrapping all writes in `Sdf.ChangeBlock()` batches the stage notifications, preventing Hydra from syncing mid-write.

### Dependencies and container image

The PyPI `usd-core` package does **not** include `UsdImagingGL` (built without `PXR_ENABLE_GL_SUPPORT`). For HydraStormViewport, we build our own Docker base image with OpenUSD compiled from source:

```dockerfile
# ghcr.io/microrobotics-simulation-framework/mime:usd-gl
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Build OpenUSD with Python bindings AND GL support
RUN python3 build_scripts/build_usd.py \
    --python \
    --no-examples --no-tutorials --no-docs \
    --no-embree --no-prman \
    --onetbb \
    /opt/usd
# This enables pxr.UsdImagingGL.Engine
```

This image is published as `ghcr.io/microrobotics-simulation-framework/mime:usd-gl` and used by:
- SkyPilot cloud jobs (via MADDENING's cloud launcher)
- CI for HydraStorm viewport tests
- Local development (optional — PyVistaViewport works without it)

The MICROROBOTICA project already publishes its own base image (`ghcr.io/microrobotics-simulation-framework/microrobotica:base`) with USD built for C++. The MIME image is separate because it needs Python bindings + GL, which MICROROBOTICA's image does not include.

**EGL driver requirement**: the Docker image must run on a host with NVIDIA GPU drivers and the NVIDIA Container Toolkit (for GPU passthrough). SkyPilot GPU instances (T4, L4, A10G) provide this. Mesa llvmpipe is NOT an acceptable fallback for production streaming — it is far too slow. llvmpipe is acceptable only for CI testing of the EGL codepath on CPU-only runners, gated behind a performance threshold check (skip if FPS < 5).

### Test plan

- Unit test: create stage, render with Hydra, check non-black output
- Skip test if EGL not available (graceful degradation — CI runners without GPU)
- Performance: render 100 frames, assert > 20 FPS on GPU, log FPS on CPU

---

## Threading and Double-Buffering

### Current design: synchronous

In the initial implementation, rendering is synchronous — `StageBridge.update()` writes to the stage, then `USDViewport.render()` reads from the same stage, all in the same thread (called sequentially in the `StepObserver` callback or after `PolicyRunner.step()`).

### Forward path: double-buffered asynchronous

For live preview during simulation (or for streaming at simulation-independent frame rate), the renderer needs to run in its own thread. The `StepObserver` callback runs in the simulation thread, so the two threads share the USD stage.

**Double-buffering sketch**:

```
StageBridge owns two stages: stage_A, stage_B
Simulation thread writes to stage_A (the "write" stage)
Render thread reads from stage_B (the "read" stage)
After each simulation step: swap(stage_A, stage_B)
```

The swap is a single atomic pointer swap — no stage copying. The render thread always reads a complete, consistent snapshot (never a partially-written frame). This is lock-free as long as:
- The simulation thread does not write to `stage_B` (it writes to `stage_A` only)
- The render thread does not write to `stage_A` (it reads `stage_B` only)
- The swap happens between frames (not mid-write or mid-render)

The existing `PolicyRunner.step()` design is compatible — the `StepObserver` callback fires **after** the step completes (all state is consistent), so the swap can happen inside the observer callback before the next step begins.

This is deferred to Phase 2. The synchronous implementation does not need refactoring to support it — the double-buffering is purely additive (new `StageBridge` constructor parameter `double_buffered=True`).

---

## Implementation Order

### Step 1: `StageBridge` (blocking for both viewports)

Create `src/mime/viz/stage_bridge.py`:
- In-memory USD stage creation
- Robot prim registration (sphere/ellipsoid from GeometrySource)
- Static environment geometry (parametric `UsdGeom.Cylinder` tube)
- State→USD transform update with `Sdf.ChangeBlock()`
- Field arrow as single `UsdGeom.Mesh` with orient/scale ops
- PolicyRunner StepObserver integration

### Step 2: `PyVistaViewport`

Create `src/mime/viz/pyvista_viewport.py`:
- USD→VTK geometry conversion (one-time at registration)
- Per-frame: update actor transforms only (no mesh re-creation)
- Offscreen rendering
- Tests with parametric geometry

### Step 3: Demo script

Create `examples/visualise_helical_robot.py`:
- Set up ExternalField → MagneticResponse → RigidBody chain
- Create a parametric cylindrical tube (`CylinderGeometry`, D=2mm, L=10mm) added to the stage as a static `UsdGeom.Cylinder` — no external mesh files
- Helical microrobot (prolate ellipsoid, a=150um, b=50um) navigating inside the tube
- Attach StageBridge as StepObserver
- Render with PyVistaViewport
- Save animation as GIF or MP4

### Step 4: `HydraStormViewport`

Create `src/mime/viz/hydra_viewport.py`:
- EGL surfaceless context management (real GPU driver, not llvmpipe)
- `UsdImagingGL.Engine` setup
- Synchronous framebuffer readback (PBO double-buffering deferred to Phase 2)
- Integration with Selkies transport (references MADDENING cloud module)

### Step 5: Docker base image

Create `docker/Dockerfile.usd-gl`:
- OpenUSD from source with `--python` and GL support
- Publish as `ghcr.io/microrobotics-simulation-framework/mime:usd-gl`
- Test in CI with Hydra viewport tests

### Step 6: WebRTC streaming integration

Wire HydraStormViewport output to MADDENING's Selkies transport layer:
- Framebuffer → numpy array → Selkies encoder → WebRTC stream
- Configuration via MADDENING's `StreamConfig` presets (PREVIEW/STANDARD/CAPTURE)

---

## File Layout

```
src/mime/viz/
├── __init__.py
├── stage_bridge.py        # State→USD bridge (StepObserver)
├── pyvista_viewport.py    # PyVista offscreen renderer
└── hydra_viewport.py      # Hydra Storm + EGL headless renderer

docker/
└── Dockerfile.usd-gl      # OpenUSD with Python + GL for HydraStorm

examples/
├── visualise_helical_robot.py   # Demo: helical robot in parametric tube
└── stream_simulation.py         # Demo: cloud streaming via Selkies
```

---

## Dependencies to Add

```toml
[project.optional-dependencies]
viz = [
    "pyvista>=0.42",
    "usd-core>=24.8",
]
viz-gpu = [
    # usd-core from PyPI does NOT include UsdImagingGL.
    # For HydraStormViewport, use the ghcr.io/mime:usd-gl Docker image
    # which has OpenUSD built from source with PXR_ENABLE_GL_SUPPORT=ON.
    "pyvista>=0.42",
    "PyOpenGL>=3.1",
]
```

---

## Open Questions

1. **Drug concentration visualisation**: When ConcentrationDiffusionNode exists, how to
   visualise a 3D scalar field on the USD stage? Options: volume rendering
   (HydraStorm supports it via `UsdVol`), isosurface extraction (PyVista can do this),
   or colour-mapped slice planes. Deferred to Phase 2.

2. **Multiple robots**: For swarm simulations (Phase 3), `StageBridge` needs
   to support N robot prims under `/World/Robots/robot_0`, `.../robot_1`, etc.
   The current API registers one robot — extend to `register_robots(count=N)`.
