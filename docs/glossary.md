# Glossary

A shared vocabulary across the three layers of the Microrobotics Simulation
Framework. Terms link in from any page that uses the `` {term}`Name` ``
MyST role; with `sphinx-tippy` enabled, hovering the link pops up the
definition without leaving the page.

This page is **single-source-of-truth in the MICROROBOTICA repo** at
`docs/glossary.md`. Identical mirrors live in `MADDENING/docs/glossary.md`
and `MIME/docs/glossary.md`, kept in sync by
`MICROROBOTICA/scripts/sync_glossary.py`.

```{glossary}
:sorted:

Node
   The fundamental unit of computation in a MADDENING graph: a pure
   function `(state, dt, boundary_inputs) -> new_state`. Subclassed in MIME
   to add physics- and biology-specific metadata.

Edge
   A typed, unit-aware connection between two nodes that carries a named
   field (e.g. `magnetic_force`, `orientation`). Edges encode the
   data-dependency graph the {term}`Graph step` walks. An edge may also
   run from a node to itself: it then reads the node's value from the
   previous step, or, with the node in a coupling group, is iterated with
   the group.

Graph step
   One full traversal of a MADDENING node graph that advances simulation time
   by `dt`. Compiles into a single XLA op so the whole step is autodiff-able.

Coupling
   The strategy by which mutually dependent nodes exchange data within a
   single graph step. MADDENING ships {term}`Gauss-Seidel` and {term}`Jacobi`
   coupling with {term}`Aitken acceleration` and {term}`IQN-ILS`.

Gauss-Seidel
   Iterative coupling scheme where each node updates with the *latest*
   values from its neighbours within an iteration. Converges faster than
   {term}`Jacobi` for many stiff systems but is sequential within a step.

Jacobi
   Iterative coupling scheme where every node updates from the previous
   iteration's values, so all nodes can update in parallel. Slower
   convergence than {term}`Gauss-Seidel` but trivially parallelisable.

Aitken acceleration
   A scalar relaxation factor (re-estimated each iteration from the
   residual sequence) that speeds up {term}`Gauss-Seidel` / {term}`Jacobi`
   convergence — cheap, no Jacobian needed.

IQN-ILS
   *Interface Quasi-Newton with Inverse Least-Squares*. A vector-valued
   acceleration scheme that builds a low-rank approximation of the
   Jacobian from past iterations — usually much faster than
   {term}`Aitken acceleration` for strongly coupled multiphysics.

Richardson extrapolation
   Run a step at two different `dt`s and combine the results to estimate
   the truncation error and a higher-order solution. MADDENING uses it to
   drive an adaptive timestepper.

PI controller
   A *Proportional–Integral* feedback controller. In MADDENING's adaptive
   timestepper it shrinks/grows `dt` based on the current
   {term}`Richardson extrapolation` error and its trend across recent
   steps — gentler than a pure proportional rule.

JIT compilation
   *Just-In-Time* compilation. Each MADDENING {term}`Graph step` is
   `jax.jit`-compiled into a single {term}`XLA` op the first time it is
   called; subsequent calls reuse the cached binary.

XLA
   *Accelerated Linear Algebra* — the optimising tensor-program compiler
   JAX targets. XLA emits CPU / GPU / TPU code from JAX's tracing IR.

vmap
   `jax.vmap` — automatic vectorisation. Lets you run the same
   {term}`Graph step` across a batch of initial conditions (or
   parameters) with no manual loop and no per-instance compilation.

MLP
   *Multi-Layer Perceptron* — a stack of dense layers. The default
   {term}`Surrogate` architecture for short-range physics nodes such as
   `MLPResistanceNode`.

DeepONet
   *Deep Operator Network* — a neural architecture that learns
   function-to-function mappings (operators). MADDENING uses it as a
   {term}`Surrogate` family for nodes that map fields to fields.

FNO
   *Fourier Neural Operator* — a spectral-domain neural operator that is
   often dramatically more sample-efficient than {term}`MLP`s on PDE
   surrogates. Another {term}`Surrogate` option in MADDENING.

Topological sort
   The ordering of graph nodes such that every edge points from earlier
   to later. MADDENING's `GraphManager` topologically sorts each step;
   cycles are broken by *cycle staggering* (run them iteratively under
   {term}`Coupling`).

IBM-FVM
   *Immersed Boundary Method on a Finite-Volume Mesh* — MIME's
   higher-fidelity option for confined low-Reynolds flow. The body is
   represented as forcing terms on a regular fluid grid rather than as a
   conforming mesh.

GNN correction
   *Graph Neural Network* residual learnt on top of the analytical
   {term}`Stokeslet` (or {term}`IBM-FVM`) prediction to absorb modelling
   error. Optional in MIME; toggled via the asset schema.

SkyPilot
   Open-source multi-cloud orchestration framework. MADDENING uses it
   to provision GPU VMs (RunPod, Lambda, AWS, GCP) for distributed
   graph execution and surrogate training.

ZMQ
   *ZeroMQ* — the message-passing library MADDENING uses to wire
   distributed graph fragments together and to stream graph state to
   visualisation clients.

WebRTC
   The browser-native real-time streaming protocol. MADDENING uses it
   to push rendered viewport frames from a remote GPU VM to a local
   browser or to the MICROROBOTICA IDE (`SelkiesSession`, with the
   browser client `api/static/webrtc_client.html`). The stream encodes
   in software (`x264enc`) on every host. What is tested: a GStreamer
   WebRTC peer that signals as the browser client does gets the answer,
   completes ICE, and decodes the frames pushed, on CPU in a network-less
   container (`tests/cloud/test_webrtc_loopback.py`, CI job
   `test-streaming`); the cloud container's entry point, run as the
   container runs it, streams the simulation built and run over its REST
   API to that peer (`tests/cloud/test_entrypoint_streams.py`). On
   2026-10-12 the stream was also tested from a GPU host with the
   simulation on CUDA, and over the internet through a TURN relay. By
   default the media travels over UDP, which a cloud provider may not
   expose; `StreamConfig.ice_servers` is where a STUN or TURN server is
   set.

pybind11
   Header-only C++ ↔ Python binding generator. MICROROBOTICA ships
   *one* `microrobotica` C++ library and exposes it both to the Qt app
   and to the embedded scripting console via pybind11.

ROS
   *Robot Operating System*. The hardware-integration bridge: the
   MICROROBOTICA IDE can publish or subscribe to ROS topics so a
   simulation can drive (or be driven by) a real robotic platform.

ASME V&V 40
   *Verification & Validation 40-2018* — the ASME standard that
   defines the {term}`COU` framing and the credibility levels expected
   of medical-device computational models.

Multi-rate
   A scheduling mode where different nodes advance at different `dt`s,
   reconciled by a GCD-based scheduler. Used for stiff subsystems that need a
   smaller step than the rest of the graph. Between the base steps a slower
   node fires on, its readers see what it holds: its held state, and the
   flux (`compute_boundary_fluxes`) of that held state.

Surrogate
   A neural network (MLP / DeepONet / FNO) trained to mimic a slow physics
   node so it can be hot-swapped in at runtime. See `surrogates/` in MADDENING.

USD layer
   One of the three Pixar OpenUSD layers MICROROBOTICA uses to compose a
   scene: the immutable *anatomy* base, the simulator-written *results*
   layer, and the user *override* layer. Output never overwrites anatomy.

ComponentMeta
   The compile-time annotation MICROROBOTICA attaches to every interface,
   capturing intended use, verification status, and SOUP classification for
   {term}`IEC 62304` compliance.

SOUP
   *Software Of Unknown Provenance* — an IEC 62304 term for any third-party
   software included in a medical device. Each SOUP needs a documented
   classification and risk analysis.

COU
   *Context Of Use* — the FDA/ASME V&V40 framing of "what question is this
   simulation answering, and how much accuracy does that demand?" Drives
   per-node verification rigour.

IEC 62304
   The international standard for medical device software lifecycle
   processes. MICROROBOTICA ships pre-wired traceability (`ComponentMeta`,
   anomaly registry, verification benchmarks) to keep audit overhead low.

EU MDR
   *EU Medical Device Regulation* (2017/745). MDR Annex II.2 in particular
   asks for design rationale, V&V evidence, and traceability — the same
   evidence the framework's regulatory docs are organised around.

Stokeslet
   The fundamental Green's function of the Stokes equations: the flow field
   produced by a point force in a viscous, low-Reynolds fluid. MIME uses
   regularised Stokeslets for confined low-Re hydrodynamics.

Low-Re
   *Low Reynolds number* (Re ≪ 1). The regime where viscous forces dominate
   inertia — i.e. the regime nearly all microrobots in biological fluids
   operate in.

UMR
   *Untethered Microrobot* — a microscale device propelled and steered by
   external fields rather than wires. The framework's reference workload.

Helical UMR
   A {term}`UMR` shaped as a helix and rotated by an external magnetic
   field, converting rotation into translation. See the de Jongh (2025)
   replication in `MIME/docs/validation/umr_deboer2025/`.

Dipole response
   The force and torque a permanent-magnet or paramagnetic body experiences
   in an external field gradient. Modelled by MIME's
   `PermanentMagnetResponseNode`.

Lubrication correction
   A near-wall force correction added on top of bulk drag in confined flows.
   MIME's `LubricationCorrectionNode` adds this to MLP-predicted resistance.

Anatomical regime
   The set of physiological bounds (compartment, flow regime, pH, viscosity,
   temperature) under which a MIME asset has been validated. Enforced at
   load time before a simulation can run.

JAX
   The Python array library MADDENING is built on: NumPy-style code with
   composable transforms — `grad` (autodiff), `jit` ({term}`JIT compilation`
   to {term}`XLA`), `vmap` ({term}`vmap`), and `scan`.

SimulationNode
   The MADDENING base class every physics block subclasses. It declares an
   `initial_state` and an `update(state, boundary_inputs, dt)` method —
   see {term}`Node`.

GraphManager
   The MADDENING object you add nodes and edges to. It validates the wiring,
   runs the {term}`Topological sort`, applies {term}`Coupling`, and compiles
   the whole {term}`Graph step` into one {term}`XLA` program.

JAX-traceable
   Code that JAX can trace into its IR: no Python-level side effects, no
   data-dependent Python control flow, no in-place mutation. Node `update`
   methods must be JAX-traceable — use `jnp.where` instead of `if/else`.

Pure function
   A function whose output depends only on its inputs and which has no
   side effects. MADDENING nodes are pure functions of
   `(state, boundary_inputs, dt)`; this is what makes them
   {term}`JAX-traceable` and safe to {term}`vmap`.

Boundary inputs
   The data a {term}`Node` receives from its upstream neighbours on a given
   {term}`Graph step` — the values flowing in along its incoming
   {term}`Edge`s.

LBM
   *Lattice Boltzmann Method* — a fluid solver that evolves particle
   distribution functions on a regular lattice rather than solving the
   Navier-Stokes equations directly. Well suited to confined,
   complex-geometry flow.

IB-LBM
   *Immersed-Boundary Lattice Boltzmann Method* — {term}`LBM` with the
   microrobot represented as immersed-boundary forcing points rather than
   a body-fitted mesh. One of MIME's confined-flow fluid nodes.

Reynolds number
   The dimensionless ratio of inertial to viscous forces in a flow.
   Microrobots operate at {term}`Low-Re` (Re ≪ 1), where viscosity
   dominates and motion is effectively time-reversible.

CSF
   *Cerebrospinal fluid* — the clear fluid in the brain ventricles and
   spinal canal. A primary target environment for the microrobots MIME
   models; its flow and viscosity define one {term}`Anatomical regime`.

Verification benchmark
   A reproducible test that checks a node or graph against a known
   analytical solution or reference dataset. Every MADDENING / MIME node
   ships verification benchmarks; their results feed the audit trail.

Defect correction
   An iterative scheme that solves a hard problem by repeatedly solving an
   easier approximate one and correcting with the residual ("defect").
   MIME uses it to add fidelity to fast fluid nodes without paying the
   full solver cost every step.

Quasi-Newton
   A family of iterative solvers that approximate the Jacobian from past
   iterates instead of computing it directly. {term}`IQN-ILS` is the
   quasi-Newton scheme MADDENING uses to accelerate {term}`Coupling`.

Cycle staggering
   How MADDENING handles feedback loops in the graph: a cyclic dependency
   is broken into an iterative {term}`Gauss-Seidel` sweep rather than
   rejected, so strongly coupled subsystems still compile.

NodeMeta
   The compliance-metadata record attached to every MADDENING
   {term}`SimulationNode`: stability level, validated regimes, references.
   MIME's {term}`MimeNode` extends it with biomedical fields; MICROROBOTICA's
   {term}`ComponentMeta` is the IDE-side analogue.

MimeNode
   MIME's base node class — a {term}`SimulationNode` whose {term}`NodeMeta`
   is extended with domain metadata (biocompatibility, {term}`Anatomical
   regime`, {term}`SOUP` classification).

OpenUSD
   *Open Universal Scene Description* — Pixar's open scene-graph format and
   composition engine. The framework's interchange format: MADDENING
   serialises graphs to USD, MICROROBOTICA composes and renders them via
   {term}`USD layer`s.

Step-out frequency
   The rotation rate above which a magnetically driven {term}`UMR` can no
   longer keep up with the rotating field and desynchronises — its
   propulsion speed collapses past this point. A key validation metric
   for {term}`Helical UMR` experiments.

CFL number
   *Courant–Friedrichs–Lewy number* — a dimensionless stability limit for
   explicit time-stepping: information must not cross more than one grid
   cell per step. Exceeding it makes a solver blow up; it bounds the
   timestep of MIME's fluid nodes.

Womersley number
   A dimensionless number for pulsatile flow, comparing the oscillation
   frequency to viscous diffusion. It characterises time-varying
   physiological flows such as pulsing {term}`CSF`.

RFT
   *Resistance-Force Theory* — a fast analytical model for slender-body
   propulsion in {term}`Low-Re` flow: drag is decomposed into components
   parallel and perpendicular to the body. Used as a cheap baseline
   against the full hydrodynamic nodes.

Bouzidi scheme
   An interpolated bounce-back boundary condition for {term}`LBM` that
   places curved walls accurately between lattice nodes, instead of
   stair-stepping them onto the grid. Improves confined-flow fidelity.

Momentum exchange
   The standard way to read the hydrodynamic force on an immersed body out
   of an {term}`LBM` simulation: sum the momentum transferred as
   distribution functions bounce off the body's boundary links.

PISO
   *Pressure-Implicit with Splitting of Operators* — a segregated
   pressure–velocity solver for incompressible flow, used by MIME's
   finite-volume ({term}`IBM-FVM`) fluid node.

Schwarz coupling
   A domain-decomposition method: two solvers covering overlapping
   sub-regions exchange boundary values and iterate to agreement. MIME
   uses it to couple a boundary-element near field to an {term}`LBM` or
   FVM far field.

CRBA
   *Composite-Rigid-Body Algorithm* — an O(n²) method for computing a
   robot's joint-space mass matrix. Used by MIME's `RobotArmNode`.

RNEA
   *Recursive Newton–Euler Algorithm* — an O(n) method for robot inverse
   dynamics (joint torques from motion). Paired with {term}`CRBA` in
   MIME's robot-arm nodes.

ISO 14971
   The international standard for *risk management* of medical devices.
   The framework's hazard-hint metadata is structured to feed an
   ISO 14971 risk file.

ISO 13485
   The international standard for a medical-device *quality management
   system*. Governs the processes around the software, complementary to
   the product-focused {term}`IEC 62304`.

IEC 62366-1
   The international standard for *usability engineering* of medical
   devices — the use-related risk and human-factors process that sits
   alongside {term}`IEC 62304` and {term}`ISO 14971`.

MDCG 2019-11
   EU Medical Device Coordination Group guidance on *qualification and
   classification of software* under the {term}`EU MDR`. Determines
   whether and at what class a piece of software is regulated.

MDCG 2019-16
   EU Medical Device Coordination Group guidance on *cybersecurity* for
   medical devices under the {term}`EU MDR`.

Notified Body
   The independent organisation that audits a manufacturer and its
   technical documentation before a medical device may be CE-marked under
   the {term}`EU MDR`.

SaMD
   *Software as a Medical Device* — software that is itself a medical
   device, rather than software embedded in hardware. The regulatory
   category the simulation framework's downstream uses fall under.

SBOM
   *Software Bill of Materials* — a machine-readable inventory of every
   dependency in a build. MADDENING commits one per covered install
   (CycloneDX format, `docs/validation/sbom/`) so downstream {term}`SOUP`
   analysis has a complete component list.

Anomaly registry
   MADDENING's record of its known defects,
   `docs/validation/known_anomalies.yaml`: one entry per anomaly, with an
   ID (`MADD-ANO-NNN`), a severity, the versions it affects and its status.
   CI checks it with `scripts/check_anomalies.py`, and the known-anomaly
   tables of the {term}`SOUP` package are generated from it.

Checkpoint
   A running graph saved to disk so the run can resume from it: every
   node's state, the graph's `params` and its internal bookkeeping (the
   multi-rate step counter, coupling warm starts) in one `.npz` archive.
   `GraphManager.save_state` writes one and `load_state` reads it back; the
   cloud entry point resumes from the one `RESUME_FROM_URL` names.

Coupling group
   A set of nodes whose mutual dependence is solved by iteration inside
   each {term}`Graph step`: the members run repeatedly, reading each
   other's current-iteration values, until the change in a pass meets the
   group's convergence norm or `max_iterations` is reached. Declared with
   `GraphManager.add_coupling_group`; one node with an edge to itself can
   form a group.

Cramér–Rao bound
   The smallest variance any unbiased estimator of a parameter can have,
   given the data: the diagonal of the inverse of the
   {term}`Fisher information matrix`. `maddening.sysid.fim` reports it per
   parameter as `crb`, and reports `inf` for a parameter the data cannot
   identify.

DEPRECATED
   The {term}`Stability level` of a surface scheduled for removal. Its
   docstring names the replacement, and using it warns
   (`DeprecationWarning`, or `FutureWarning` where the result will change).

EVOLVING
   The {term}`Stability level` of a surface whose signature and wire format
   are settled, though additions are expected. An incompatible change needs
   a minor release and one minor release of `DeprecationWarning` first.

EXPERIMENTAL
   The {term}`Stability level` of an opt-in surface that may change or
   disappear in any minor release, without a deprecation period. Each such
   change still gets a CHANGELOG entry.

External input
   A value fed into a node's {term}`Boundary inputs` from outside the
   graph (a controller, a sensor, a user command) rather than along an
   {term}`Edge`. Declared with `GraphManager.add_external_input` and passed
   to each `step` or `run` call in `external_inputs`; a field takes an
   external input or edges, never both.

Fisher information matrix
FIM
   `F = JᵀJ`, for `J` the Jacobian of a residual (model minus data) with
   respect to the parameters: how strongly the data constrain each
   combination of parameters. An eigenvalue near zero names a combination
   the data cannot distinguish; `maddening.sysid.fim` reports the
   eigenpairs, the rank and the {term}`Cramér–Rao bound`.

Fixed point
   A state that one more pass of an iteration leaves unchanged:
   `x* = F(x*)`. A {term}`Coupling group` iterates towards the fixed point
   of its coupling pass in every step, and its default solver
   differentiates that fixed point with the {term}`IFT` rather than through
   the iterations.

FMI
   *Functional Mock-up Interface* — the open standard for exchanging models
   between simulation tools as packaged {term}`FMU`s and stepping them
   together (co-simulation). MADDENING exports to FMI 3.0.

FMU
   *Functional Mock-up Unit* — a `.fmu` zip holding a model description and
   a compiled binary that an {term}`FMI` importer (FMPy, OpenModelica, ...)
   loads and steps. A MADDENING FMU is a co-simulation FMU whose thin C
   wrapper forwards every call to a Python {term}`Sidecar` holding the
   compiled graph.

Geometry-dependent mapping
   An {term}`Interface mapping` that computes its weights at every step from
   positions held in the graph's state, so the points it maps between may
   move (markers carried by a flow, a body crossing a grid). The edge names
   the positions with `geometry=(anchor, field)`; `multilinear_grid` is the
   built-in kind.

Halo
   The ghost cells around a {term}`Shard`, copied in from the neighbouring
   shards, so that a {term}`Stencil` update near the shard's border can read
   its neighbours. A node declares how many it needs per axis with
   `halo_width()`; at the edges of the global grid a boundary mode
   (`"edge"`, `"periodic"` or `"zero"`) fills them.

IFT
Implicit function theorem
   *Implicit function theorem*. A {term}`Coupling group`'s default solver
   (`solver="ift"`) uses it to differentiate the group's converged
   {term}`Fixed point` directly, by solving a linear system in `I - dF/dx`,
   instead of differentiating through every iteration. The derivative is
   exact only at a converged fixed point.

Interface mapping
   The transfer of a field between two nodes that discretise a shared
   interface differently (a coarse and a fine rod, a mesh and a point
   cloud), applied on the {term}`Edge` before its transform. A
   `"consistent"` mapping interpolates a value field; a `"conservative"`
   one transfers an integral quantity, such as a force, so that its total
   is preserved.

Interface norm
   The convergence norm `convergence_norm="interface"` of a
   {term}`Coupling group`: the RMS change, relative to `rtol` times each
   value's magnitude, of only the fields the group's internal edges carry
   between members. It often needs fewer
   iterations than `"l2"`, but it cannot see a member's other fields, so a
   group can stop while the rest of its state is still moving.

ParamSpec
   The metadata for one leaf of a graph's parameter {term}`Pytree`
   (`maddening.core.params.ParamSpec`): whether an optimiser may change it
   (`trainable`), its physical `bounds`, and the `transform` (`"log"`,
   `"logit"`) that keeps an unconstrained optimiser inside them. Nodes
   declare specs for their own constants, and a graph can override them
   with `GraphManager.set_param_spec`.

Precision floor
   The size below which a computed difference is float rounding rather than
   signal, set by the working precision (the `eps` of the dtype). A
   {term}`Coupling group`'s diagnostics report the residual's floor as
   `residual_precision_floor` and flag a residual at or below it as
   `precision_limited`; `maddening.sysid.fim` warns when its rank verdict
   lands near the cutoff the precision sets.

Pytree
   JAX's name for a nested structure of Python containers (dicts, lists,
   tuples) whose leaves are arrays; `jax.jit`, `jax.grad` and `jax.vmap`
   work leaf by leaf. A MADDENING graph's parameters are one pytree,
   `GraphManager.params`, so the same object flows into a run, an optimiser
   and the {term}`System identification` tools.

Reason code
   A short string in a {term}`Coupling group`'s diagnostics report that
   says why a flag such as `spectral_usable` is `False`
   (`maddening.core.coupling.reason_codes`), so that code can branch on the
   cause instead of parsing the sentence in `not_usable_reason`. Every
   cause that holds is listed; a flag that is `True` has none.

Shard
Sharding
Sharded
   One device's slice of a node's state when the state is split across
   devices along the axes of a JAX device mesh; *sharding* is the
   splitting. `ShardedPointwiseNode` and `ShardedStencilNode` wrap a node so
   that each device updates its own shard, the stencil wrapper exchanging a
   {term}`Halo` between neighbours.

Sidecar
   The long-running Python process behind a MADDENING {term}`FMU`: it holds
   the compiled graph and answers the FMU's calls, forwarded over TCP, so
   that XLA's compile cost is paid once rather than on every instantiation
   (`maddening.fmi.sidecar.FmuSidecar`).

Spectral radius
   The largest modulus among a matrix's eigenvalues. For a fixed-point
   iteration it is the factor by which the error shrinks per pass in the
   long run, so the iteration converges locally when it is below 1. A
   {term}`Coupling group` built with `diagnostics=True` reports an estimate
   for its Jacobian `dF/dx` as `rho_spectral`.

Stability level
   The promise a public MADDENING surface makes about change, set with
   `@stability(StabilityLevel.X)`: {term}`STABLE`, {term}`EVOLVING`,
   {term}`EXPERIMENTAL` or {term}`DEPRECATED`. The level decides in which
   release an incompatible change may land and how much notice it needs
   (MADDENING's deprecation policy).

STABLE
   The {term}`Stability level` whose signature and documented behaviour
   hold for the life of the major version. An incompatible change needs a
   major release and two minor releases of `DeprecationWarning` first;
   MADDENING's `STABLE` set is first frozen at 1.0.

Stencil
   The neighbouring grid points a computation reads to produce one value:
   the five points of a 2-D diffusion step, or the lattice cells a
   multilinear interpolation weights. A node whose update reads neighbours
   declares how far its stencil reaches per axis with `halo_width()`, which
   sizes the {term}`Halo` when the node is sharded.

Sub-cycling
Sub-cycled
   Letting the faster members of a {term}`Coupling group` take several
   smaller steps inside one step of the group (`subcycling=True`): a member
   with timestep `node_dt` takes `macro_dt / node_dt` sub-steps, where
   `macro_dt`, the group's largest timestep, must be a whole multiple of
   every member's. It is the in-group counterpart of {term}`Multi-rate`
   scheduling.

System identification
sysid
   Estimating a model's parameters from measured data. `maddening.sysid`
   does it on a graph's parameter {term}`Pytree`: a windowed trajectory
   loss (`windowed_loss`), a fitter (`fit`) that keeps to each leaf's
   {term}`ParamSpec`, and the {term}`Fisher information matrix` (`fim`),
   which says which parameter combinations the data cannot identify.
```
