- **The iterative allocation, `ctx.allocator`'s choice of path, and a controller's run flags** (apparatus lane, AP4a,
  dev2).
  - **`mime.control.iterative_allocation(rows, start, *, lower, upper, owner)`** (`IterativeAllocation`): the commands
    within bounds (possibly infinite) that minimise half the squared rows a JAX function of them makes, for any number
    of commands and any model, by projected Newton: Newton's step on the exact Hessian where it is positive definite on
    the free commands, Gauss-Newton's where it is not, the bounds a projection, Armijo's test nonmonotone for Newton's
    step on a positive definite Hessian that no bound cuts (against the largest objective of the last four
    points: in the curved valley of an arm's redundant joints that step first raises the objective, and a monotone test
    took a warm start to the cap of 50 iterations; over 2220 such solves the most went from 47 to 35; a step the box
    cuts or a Gauss-Newton one stays monotone, since with the allowance a float32 OctoMag draw cycled between two
    corners of the box), Levenberg's damping (1e-3, times 10 at each further cut) after a step that had to be cut, raised at each further cut and
    lowered at each whole step (a warm start crawling by halvings met the cap 0.027 rad from the minimiser; over the
    2220 solves, at most 15 iterations), the step's path bent with the rows (Transtrum and Sethna's geodesic
    acceleration, the least-squares fit of the rows' second derivative along the step, zero on an affine model: in the
    curved valley the straight step was cut and the damped one after it overshot again, which took 3 of 7474 of the
    follower's warm starts to the cap; with the bend none, at most 15 iterations), the decrease measured as a
    difference of squares, a decrease within what the
    objective resolves passing (each row's rounding judged on its terms' scale, the problem's size, not on the row's).
    **Its guards, declared**: at most `ITERATIONS` (50) iterations, read at a point already weighed; `TOLERANCE` on the
    stationarity (the projected gradient's length in commands scaled to unit columns, over the rows' miss plus what the
    commands make), 1e-10 in float64 and 1e-5 in float32, at or above the floor measured; a stagnation stop, counting only steps that leave the commands within their rounding (counting every step whose decrease the objective could not resolve, it stopped 2 of 1202 float32 solves on unreachable targets far from their tolerance while the commands still moved); and **a
    local phase**, from the first point within the tolerance: full Newton steps with no line search while they are
    longer than the commands' rounding (at most 8), returning the point within the tolerance whose
    step was shortest, so that the derivative is taken at the minimiser. The tolerance alone left a warm-started, weakly
    regularised arm 4e-8 rad from it, its derivative 8e-5 off the minimiser's and a closed loop's (A0's rig, five ticks)
    7.6e-4 off central differences (1e-8 with the local phase), and a solve 0.2 mm further 2e-3 rad from it, where
    Armijo's test refused the full step that converges (in the curved valley of redundant joints it first raises the
    objective). **`converged` is read at the commands returned**: the stationarity within its tolerance there and the
    free commands' rows of full rank. Where it did not converge the commands are the last iterate inside the bounds, and
    `converged` says so. **Its derivative** is the implicit function's (`jax.lax.custom_root` on the projected
    gradient's fixed point): the exact Hessian's tangent system on the free commands, a held command's derivative its
    bound's, one dense LU of order the commands; forward and reverse mode. Status: `piecewise_kink` on an affine model
    (a convex quadratic), `piecewise_jump` on one that is not (the local minimiser followed from the start can change
    branch).
  - **`ctx.allocator` chooses the path when the law is made**, from the model's declaration and the commands' count:
    exact where the model is affine in every command asked for, at most `EXACT_COMMANDS` of them, each quantity's miss
    the difference of its rows; iterative otherwise. The refusals `allocation.exact-size` and `allocation.model`'s "not
    affine" hand over to the iterative path there (`exact_allocation()` and `allocate()` keep `allocation.exact-size`,
    naming `iterative_allocation`); an unregularised iterative allocation whose rows cannot fix its commands is refused
    (`allocation.objective`, naming `effort=`). The build report names each allocation's path and why.
    `AllocationReport` gains `path`, `converged` and `stationarity` (on the exact path `"exact"`, true and zero).
  - **A controller's run flags** (the core of the flags a law raises, for controllers): a controller declares the flags
    its step raises in its class's `raises` (a `mime.registry.Flag` each), or its law's calls declare them as it is made;
    its step raises one with `mime.control.raise_flag(name, value)`, at the step's own level. Inside the compiled step
    its node holds each as `flag_<name>`; as a callback a holder node of the build's own, fed at each tick. The run
    reads each at one path in both modes, `<controller>.<name>`, lays them out and counts them as a law's (steps
    raised, first, last, a warning for one that warns), and a study's members and `Study.gradient`'s carry them.
    Refused, each naming its fix: a name not declared, a raise inside a `jax.lax` body (`cond`, `while_loop`, `scan`,
    `fori_loop`), and a step compiled with nothing underneath it (a step compiled by `jax.jit` on a function is traced
    through it). **`allocation_unconverged`** is the first: declared by a law that allocates on the iterative path,
    raised at a tick whose solve did not converge, its value the stationarity. A controller that declares none builds
    nothing new. `Study.gradient`'s and a fit's flags now read the run's paths (a named flag's name, not its field).
  - **`ForwardModel.misses`**: a quantity's own miss where it is not the difference of its rows; the kinematic arm's
    model gives each frame's `<frame>.position` (m) and `<frame>.orientation` (a quaternion whose miss is the rotation
    vector between the two, rad), so an objective names one frame.
  - `mime.core.switch.not_collected()`: a switch collector hidden from what an iterative solve traces into programs
    of its own, whose switches its caller evaluates once more on the solution.
  - **What the iterative path bounds, and what it estimates**: against the exact path on the arrays MIME ships and
    against SciPy's BVLS above eight commands, within the line its claim derives (the oracle's perturbation bound plus
    the natural residual's error bound); against the rigid-body library's inverse kinematics on reachable poses where
    it holds no joint at a limit, and, where it does, by the first-order conditions there from the objective's own
    gradient (unregularised near the AR4's wrist singularity the minimiser reached from the start can be a joint's
    limit far from it, off the pose, where the library's damped solve met the pose close by: the solve reaches a local
    minimiser, not the nearest; a slew box bounds how far it goes); its
    derivative against central differences of converged reference solves, reverse against forward mode, float32
    against float64 (`tests/control/test_iterative_allocation.py`); through a built controller in both modes, a study's
    sweep and `Study.gradient` (`tests/control/test_iterative_controller.py`). Software oracles: nothing here says a
    model is the hardware. The pilot (`tests/control/predictions_ap4_pilot.json`) found the first solve's float32 floor
    and its unreachable-target stall, which the decrease's resolution and the stall rule above answer.
