# Tutorial A0: a magnet on an arm, in a closed loop

Experiment A0 replaces A's rotating field with a permanent magnet carried by a robot arm. The arm is an AR4. A motor on its
last link turns a rotor that carries de Jongh's N45 magnet, and the magnet is a point dipole above the same screw in the
same tube. The screw is no longer given a field: it is pulled and turned by the magnet's, and a **controller** moves the
arm every step to keep the magnet above the screw as it swims. That is the closed loop.

The file is `examples/tutorial_a0.py`. It reuses the screw and the declared values of
[tutorial A](tutorial_a.md), so read that first. The definitions of the new ideas are in [the multibody
note](../multibody.md): a chain, its segments, and what a part mounted on a frame means.

## What this tutorial claims

That the experiment builds and runs as written, and that its output is the output shown. The run is short, one second of the
screw's motion, so it says nothing of the screw's steady swim.

What the test suite holds about A0 is narrow too. `tests/differential/test_a0.py` holds it to the earlier line's A0, frozen,
with every difference declared before the comparison ran, in float32. It checks the framework against its reference *on the
same drag weights*; those weights inherit the old wall tables' defect (MIME-ANO-117), so it does not check the drag's
physics, and the earlier code is the weakest oracle.

## The apparatus: an arm, a motor and a magnet

The arm and the motor are two segments of one chain. The magnet is a part mounted on the motor's rotor, and the field it
makes is a point dipole:

<!-- mime-include: examples/tutorial_a0.py#apparatus -->

The arm is `Arm.Kinematic`: a **kinematic** segment prescribes its motion ([the multibody note](../multibody.md) defines the
word), so it moves where it is told. Its joint positions have no drive here, because the controller writes them.

## The experiment

The tube, water and the screw are A's. The magnet's moment is declared with its reason, and the laws are A's with the
magnet's force added: the magnet's field is not uniform, so it pulls the screw as well as turning it:

<!-- mime-include: examples/tutorial_a0.py#experiment -->

## Closing the loop

The controller reads the screw's pose and twist and the arm's joint positions, and writes the arm's joint positions. It
reads things a real rig could not: it reads the physics state directly, the ground truth (the build report labels each read
`state, ground truth, not deployable`), which is why it is a controller for simulating the loop and not one to deploy.

<!-- mime-include: examples/tutorial_a0.py#controller -->

`mode="graph"` puts the controller inside the compiled step. MIME has other places to put a controller, with different costs;
the build report says which it chose and what that means for when it reads and when its writes act. The controller computes in
single precision whatever the run's precision, so a run in double precision with it is mixed ([the precision
rule](install.md#precision-follows-the-dtype-of-the-run)).

## Run it

Run it from a source checkout of MIME, in an environment where MIME is installed: `python examples/tutorial_a0.py`.
`examples/` is not part of the installed package, and the arm's description is the AR4's URDF in the checkout's
`examples/descriptions/ar4/`, under the MIT licence beside its meshes' manifest (a lab uses its own). The regions below use names the file defines at its top
or takes from `script_a.py`: `AR4` (that path), `BASE` and `HOME` (where the arm stands and its joint positions),
`FREQUENCY` (the field's rate) and the screw, the reason string and the contact (`FL9`, `REFERENCE`, `CentreContact`).

Running it prints `DescriptionWarning`s on stderr, apart from the output below, each more than once: about the AR4's file, the visual
and collision elements MIME does not simulate, and links with geometry and no `<inertial>`, which it reads as massless.
They are about the description and not about the experiment: the arm is kinematic, so it moves where the controller writes
and not under forces.

<!-- mime-include: examples/tutorial_a0.py#run -->

<!-- mime-output: examples/tutorial_a0.py -->
```text
controller: arm_tracking 0.1.0, mode 1
after a short run of 1 s, the magnet is 151.0 mm from the body
```

The first line is the build report's line for the controller. The second is the distance from the magnet to the screw at the
end of the short run. The controller holds the magnet at its `standoff` setting above the line of the tube's axis,
at the screw's place along it. The printed distance is that
standoff, plus the screw's centre lying about a millimetre below the axis (it rests on the tube's floor, and starts there),
plus a tracking error of a fraction of a millimetre along the tube and across it. (`tests/docs/test_tutorials.py` holds the
excess over the standoff to the screw's starting depth.)

`tests/differential/test_a0.py` holds the swim speed, the rotation rates and the axial tracking error, among other
quantities, to the earlier line's, in float32, at two frequencies. The distance printed here is not one of them: the test
reports it and does not gate it, because it cannot tell the two frequencies apart and so tells nothing the others do not.
This run only shows that the loop is closed and the arm followed.

## Where next

The tutorial for experiment G waits for a1. [The results note](../results.md) says how to read a run's outputs beyond the
distance printed here.
