---
describes: ["0.2.0", "0.3.0", "0.4.0"]
---
# A rigid body with mass and inertia

**What it models.** A rigid body whose state is the pose and the twist of its centre of mass in the world frame, the
dynamics a body takes when its inertia matters: `dynamics=RigidInertial()` on a body whose `mass` and `inertia` are
declared or derived from its parts, with an initial `position`, `orientation` (a unit quaternion, body to world),
`velocity` and `angular_velocity`. The `inertia` is about the centre of mass in the body's frame: three principal
moments about the body's axes, or the symmetric tensor's nine numbers, row-major, its products of inertia
carried as they are. A body whose parts give products (a screw whose turns end part of the way round) runs with them in
every profile. It publishes the pose and the twist, and accepts the applied wrench, the background load, the
resistance and the confinement.

Each step it sums every wrench applied to the body and every background load, takes every resistance delivered to it
implicitly, with the step solved for the change in velocity, and takes the gyroscopic term linearly implicit at the
midpoint. The equations are on the node's page, and the entry does not repeat them. There is no setting for the
implicit step: it is not optional. The body is also where a confinement is held. A confinement the contact plans is
taken at construction and applied after the step, its centre by its extent or its hull by impulses through the step's
system and a projection after it: see [the inequality constraint](../../mechanisms/inequality_constraint.md) for what
happens between the nodes, and the contact's own page for what it holds.

An inertia that is not positive definite (a moment not above zero) or a tensor that is not symmetric is refused at
build (`value.not-positive`), since the step divides by it. A research build may still declare the diagonal of a body's
tensor in place of its parts' products, with a reason: that is a modelling choice the build report names, the tensor
it takes the place of beside it.

**When not to use it.**
- For a body rotating fast against the step: the gyroscopic term is resolved only while the rotation rate times the
  step stays below the limit the entry declares (`limits`, printed by `mime registry --json`), and past it a run is
  finite and wrong with nothing to refuse it (see MIME-ANO-129). That limit is a declaration from one body's
  measurement, not a bound for every body.
- Where a solve needs the step's matrix symmetric and positive definite: for a body of unequal moments it is not
  symmetric (see MIME-ANO-129).
- For a body with no inertia to speak of: it integrates the mass and the inertia it is given, and each must be
  positive.
- Where a confinement that holds the body's centre by its extent must also turn the body: a centre held by an extent
  never reads the orientation (see MIME-ANO-131). A body whose own inertia turns its axis across a tube under that
  contact is flagged after the run; the hull contact holds its pose.
- For a body whose fluid's inertia matters: no added mass is carried here.

**Versions.** 0.4.0 takes the full inertia tensor in the body's frame, where 0.3.0 took three principal moments about
the body's axes and refused products of inertia or, in exploratory, dropped them with a stamp. The gyroscopic term's
Jacobian keeps 0.3.0's form for the diagonal and adds the products' part alone, so a body with no products moves as it
did on 0.3.0, bit for bit per compiled program (the released programs are held so); a body with products now runs with
them. The tensor is never turned into principal moments and a rotation, whose derivative is not finite at equal
moments, which every axisymmetric body has. 0.3.0 stays registered, taking the diagonal and refusing a tensor with
products, so that a folder saved with it loads and reproduces. 0.3.0 takes the gyroscopic term linearly implicit at
the midpoint, where 0.2.0 took it explicit: the explicit term gained kinetic energy at every step for a body of three
different moments, and a rig with nothing to damp it went non-finite (MIME-ANO-129). Results of a rotating body of
unequal moments move between them; those of an isotropic body do not, bit for bit as a test holds it in float64 and for
script A's compiled program on hosts with fused multiply-add; bit for bit is a property of one compiled program, not of
the law. 0.2.0 stays registered so that a folder saved with it loads and reproduces, and a script that writes
`RigidInertial()` gets 0.4.0. 0.2.0 solves the step for the change in velocity, where the first version solved it for
the velocity after the step, which in single precision rounded a small load against the inertial term (MIME-ANO-113). A
document naming that first version is refused at load (`load.version`).
