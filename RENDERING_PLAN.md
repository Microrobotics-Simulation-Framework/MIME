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

    def update(self, state: dict) -> None:
        """Write current simulation state to USD stage."""

    def as_observer(self) -> StepObserver:
        """Return a PolicyRunner StepObserver callback."""
```

### Implementation approach

- Option (a) from ARCHITECTURE_PLAN.md: in-memory stage, updated each step
- Robot body: `UsdGeom.Xform` with `xformOp:translate` + `xformOp:orient`
- Parametric geometry: `UsdGeom.Sphere`, `UsdGeom.Capsule`, `UsdGeom.Cylinder`
- Field visualisation: arrow glyph (cone + cylinder) showing B direction
- Camera: `UsdGeom.Camera` at configurable position

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
- `UsdGeom.Capsule` → `pyvista.Cylinder` (approximate)
- `UsdGeom.Xform` transform → PyVista actor transform matrix
- Arrow glyphs for magnetic field vectors
- Offscreen: `pyvista.Plotter(off_screen=True)`

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
- The framebuffer feeds the Selkies WebRTC transport layer

### EGL headless setup

```python
import ctypes
from OpenGL import EGL

# 1. Get default display
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

### Dependencies

- `usd-core` compiled with `PXR_ENABLE_GL_SUPPORT=ON`
- `PyOpenGL` with EGL support
- GPU with EGL support (NVIDIA/AMD), or Mesa llvmpipe for CPU fallback
- No X11 or display server required

### Test plan

- Unit test: create stage, render with Hydra, check non-black output
- Skip test if EGL not available (graceful degradation)
- Performance: render 100 frames, measure FPS

---

## Implementation Order

### Step 1: `StageBridge` (blocking for both viewports)

Create `src/mime/viz/stage_bridge.py`:
- In-memory USD stage creation
- Robot prim registration (sphere/ellipsoid from GeometrySource)
- State→USD transform update
- PolicyRunner StepObserver integration

### Step 2: `PyVistaViewport`

Create `src/mime/viz/pyvista_viewport.py`:
- USD→VTK geometry conversion
- Offscreen rendering
- Tests with parametric geometry

### Step 3: Demo script

Create `examples/visualise_helical_robot.py`:
- Set up ExternalField → MagneticResponse → RigidBody chain
- Attach StageBridge as StepObserver
- Render with PyVistaViewport
- Save animation as GIF or MP4

### Step 4: `HydraStormViewport`

Create `src/mime/viz/hydra_viewport.py`:
- EGL context management
- UsdImagingGL.Engine setup
- Framebuffer readback
- Integration with Selkies transport (references MADDENING cloud module)

### Step 5: WebRTC streaming integration

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

examples/
├── visualise_helical_robot.py   # Demo: rotating robot in CSF
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
    "pyvista>=0.42",
    "usd-core>=24.8",
    "PyOpenGL>=3.1",
]
```

`usd-core` is the PyPI package for OpenUSD Python bindings. It includes
`pxr.Usd`, `pxr.UsdGeom`, `pxr.UsdImagingGL` (if built with GL support).

---

## Open Questions

1. **USD stage threading**: `StageBridge.update()` writes to the stage from
   the simulation thread. `USDViewport.render()` reads from the same stage.
   If rendering runs in a separate thread (e.g., for live preview), we need
   stage-level locking or double-buffering. For initial implementation,
   rendering is synchronous (called after each step in the same thread).

2. **Field visualisation**: How to visualise a 3D magnetic field vector?
   Options: arrow glyph at robot position, field line streamlines (expensive),
   colour-mapped plane slice. Start with arrow glyph, extend later.

3. **Drug concentration**: When ConcentrationDiffusionNode exists, how to
   visualise a 3D scalar field on the USD stage? Options: volume rendering
   (HydraStorm supports it), isosurface extraction (PyVista can do this),
   or colour-mapped slice planes.
