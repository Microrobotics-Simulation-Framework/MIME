---
describes: "0.1.0"
---
# A magnet carried by an arm, following a body

**What it models.** The controller of an arm that carries a magnet above a robot in a tube and keeps it there as the
robot swims: `exp.set_controller(ArmTracking(description=asset("arm.urdf"), end_effector="link_6", home=rad(...),
axis=X, up=Z, at=m(...)), reads={"body_pose": "umr.pose", "body_twist": "umr.twist", "joint_positions":
"rpm.arm.joint_positions"}, writes={"joint_positions": "rpm.arm.joint_positions"}, mode="callback")`, or `mode="graph"`
to run it inside the compiled step. Each control period it reads the body's pose, its twist and the arm's joint
positions, sets a target for the arm's end effector at the standoff along `up` from where the body is along `axis`, that
distance held within `reach`, and writes the joint positions that a few damped-least-squares steps reach from the arm's
present ones: the command of a kinematic arm. The target moves a fraction `smoothing` of the way towards that point each
tick, from the end effector's pose at `home`; with `orientation_feedback`, its orientation also turns so that the end
effector's z follows the body's.

It is the v0.2 line's controller of the AR4 rig, ported: the same target, filter and inverse kinematics, compared with
the v0.2 controller on a recorded input sequence. It computes in single precision whatever the run's precision, as the
v0.2 controller did, so a run in double precision with it is mixed.

**When not to use it.**
- When what the controller reads matters: it reads the body's pose and twist and the arm's joint positions straight
  from the physics state, which the build report lists as ground truth, not deployable (see MIME-ANO-104). No
  instrument gives a real controller the robot's pose inside a body.
- For a dynamic arm: it writes joint positions, and the joint torques of the v0.2 line's computed-torque control wait
  for the dynamic arm.
- When a derivative of the run is needed and it runs as a callback: between compiled chunks no derivative passes
  through its writes. Inside the compiled step one does, and it is `piecewise_jump`: the reach is a clip (a kink), and
  the inverse kinematics' half turn, the orientation feedback's hemisphere and its fallback where the body's z lies
  along the home pose's x are jumps. A tracking target never comes near a half turn of the end effector.
- For a tube that is not straight through the origin along `axis`: it follows the body's distance along that one line.
- When the target filter's time constant must stay fixed as the rate changes: `smoothing` is a fraction per tick, so
  the same value at another rate is another filter.

**Versions.** `arm_tracking` 0.1.0 is its first: as a callback or inside the compiled step, writing a kinematic
arm's joint positions.
