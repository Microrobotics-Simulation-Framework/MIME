"""MIME's own interface names: the body's state and targets and the magnetic field, as the bodies, laws and fields of
the first experiment use them. Every vector and operator is in the world frame; a body's wrench and twist are about
its centre of mass; a composite field lists its parts in order, separated by ``;`` in its units."""

from mime.vocabulary._vocabulary import Interface, ReadKind, TargetKind, define

POSE = define(Interface(
    "body.pose", ReadKind.STATE, "m; 1", "world", (7,), magnitude=1e-3,
    description="[position of the centre of mass; orientation as a unit quaternion (w, x, y, z), body to world]"))
TWIST = define(Interface(
    "body.twist", ReadKind.STATE, "m/s; rad/s", "world", (6,), magnitude=1e-2,
    description="[velocity of the centre of mass; angular velocity]"))
MASS_MATRIX = define(Interface(
    "body.mass_matrix", ReadKind.STATE, "kg; kg m; kg m^2", "world", (6, 6), magnitude=1e-6,
    description="the body's mass on the translations' diagonal and its inertia about the centre of mass, turned to "
                "the body's pose, on the rotations' block, published only where a plan reads it (a medium's near "
                "field, whose lagged gain needs every body's); a step that takes the gyroscopic term implicitly "
                "solves with this less half the step times its stiffness, which vanishes for an isotropic body"))
APPLIED_WRENCH = define(Interface(
    "body.applied_wrench", TargetKind.LOAD, "N; N m", "world", (6,), sign="on the body, about its centre of mass",
    magnitude=1e-6, description="[force; torque], summed over every law that loads the body"))
RESISTANCE = define(Interface(
    "body.resistance", TargetKind.OPERATOR, "N s/m; N s; N m s", "world", (6, 6),
    sign="the load on the body is -R [velocity; angular velocity] + the background load; v.Rv >= 0",
    magnitude=1e-6, description="the resistance to the body's motion, integrated implicitly by an inertial body"))
BACKGROUND_LOAD = define(Interface(
    "body.background_load", TargetKind.LOAD, "N; N m", "world", (6,),
    sign="on the body, held still in the ambient flow, about its centre of mass", magnitude=1e-6,
    description="[force; torque] from the ambient flow, paired with a resistance"))
CONFINEMENT = define(Interface(
    "body.confinement", TargetKind.INEQUALITY_CONSTRAINT, "m", "world", (),
    description="a fixed surface the body stays on the free side of, with a margin and a velocity rule, applied by "
                "the body's dynamics after its step; at most one per body"))
FLUX_DENSITY = define(Interface(
    "field.magnetic.flux_density", ReadKind.FIELD, "T", "world", (3,), magnitude=1e-3,
    description="the magnetic flux density B at a point"))
GRADIENT = define(Interface(
    "field.magnetic.gradient", ReadKind.FIELD, "T/m", "world", (3, 3), magnitude=1e-1,
    description="G_ij = dB_i/dx_j at a point, so that a moment's force is G^T m"))
FRAME_POSES = define(Interface(
    "apparatus.frame_poses", ReadKind.STATE, "m; 1", "world", (-1, 7), magnitude=1e-1,
    description="the world pose of every frame an apparatus chain's segment names, one row per frame in the order "
                "the segment lists them: [position of the frame's origin; orientation as a unit quaternion (w, x, y, "
                "z), frame to world]. Every rung of a segment publishes it, so mounts read it alike"))
FRAME_ANGULAR_VELOCITIES = define(Interface(
    "apparatus.frame_angular_velocities", ReadKind.STATE, "rad/s", "world", (-1, 3), magnitude=1.0,
    description="the angular velocity in the world of every frame an apparatus chain's segment names, one row per "
                "frame in the order the segment lists them, as frame_poses lists their poses: what a segment mounted "
                "on one of them reads to know how its mount turns (a spinning rotor's bearing and gyroscopic torque). "
                "Every MIME segment publishes it, on every rung"))
JOINT_POSITIONS = define(Interface(
    "apparatus.joint_positions", ReadKind.STATE, "1", None, (-1,), magnitude=1.0,
    description="the angles in radians of a chain segment's moving joints, in the order its description lists them. "
                "The radian is dimensionless in MIME; a prismatic joint's position is a length in metres that this "
                "unit cannot tell from an angle, so a controller declared in this unit takes revolute joints only"))
JOINT_VELOCITIES = define(Interface(
    "apparatus.joint_velocities", ReadKind.STATE, "rad/s", None, (-1,), magnitude=1.0,
    description="the rates of a chain segment's moving joints, in the order its description lists them: radians per "
                "second, or metres per second on a prismatic joint, which this unit cannot tell from an angular "
                "rate; a dynamic segment publishes them, a kinematic one does not"))
LINK_WRENCH = define(Interface(
    "apparatus.link_wrench", TargetKind.LOAD, "N; N m", "world", (-1, 6),
    sign="on the body each frame is fixed to, about the frame's origin", magnitude=1e-3,
    description="the loads on a chain segment's frames, one row per frame in the order the segment lists them: "
                "[force; torque about the frame's origin], summed over every delivery. The load target a dynamic "
                "segment offers, which a kinematic one does not"))
LINK_MOMENTUM = define(Interface(
    "apparatus.link_momentum", TargetKind.OPERATOR, "N m s", "world", (-1, 3),
    sign="the gyroscopic torque on the body each frame is fixed to is h x omega, omega the frame's angular velocity",
    magnitude=1e-3,
    description="the angular momentum about its spin axis of each rotor a chain segment's frame carries, one row per "
                "frame in the order the segment lists them, summed over every delivery: what a spinner hands the "
                "frame it is mounted on, whose gyroscopic torque the segment applies at its own velocity. A dynamic "
                "segment takes it, a kinematic one does not"))
LINK_INERTIA = define(Interface(
    "apparatus.link_inertia", TargetKind.OPERATOR, "kg m^2", "world", (-1, 2, 3, 3),
    sign="added to the inertia of the body each frame is fixed to, about its centre of mass, at the step's start "
         "and at its end",
    magnitude=1e-5,
    description="the inertia each of a chain segment's frames carries beyond its own inertial, one row per frame in "
                "the order the segment lists them, at the step's start and at its end, summed over every delivery: "
                "what a rotor unequal across its spin axis hands the frame it is mounted on, its half difference "
                "across the axis, which turns with it. A dynamic segment takes it, a kinematic one does not"))
FLOW_VELOCITY = define(Interface(
    "field.flow.velocity", ReadKind.FIELD, "m/s", "world", (3,), magnitude=1e-3,
    description="the velocity of a medium's flow at a point: the quantity a flow field kind provides"))
FLOW_BODY_FORCE = define(Interface(
    "flow.body_force", TargetKind.LOAD, "m/s^2", "world", (3,), sign="on the fluid of a medium, per unit mass",
    magnitude=1e-3,
    description="a uniform body force per unit mass on a medium's fluid (none on its solid), summed over every law "
                "that drives it: on a periodic box, the drive that stands in for a pressure drop"))
FLOW_MOMENTUM = define(Interface(
    "flow.momentum", ReadKind.STATE, "kg m/s", "world", (3,), magnitude=1e-8,
    description="a medium's fluid's total momentum, the integral of rho u over its fluid (none over its solid, as a body "
                "force acts on its fluid alone): the flow a uniform body force per unit mass on it is paired with"))
FLOW_FACES = define(Interface(
    "flow.faces", ReadKind.STATE, "m/s", "grid", (-1, -1, -1), magnitude=1e-3,
    description="a medium's flow on its grid: the velocity on one set of the grid's faces, in the grid's own shape and "
                "coordinates, which an exchange law's gather reads at a partner's points (each face set its own "
                "field)"))
FLOW_EXCHANGE_X, FLOW_EXCHANGE_Y, FLOW_EXCHANGE_Z = (define(Interface(
    f"flow.exchange_{axis}", TargetKind.LOAD, "N/m^3", "grid", (-1, -1, -1),
    sign="on a medium's fluid and solid alike, per unit volume", magnitude=1e3,
    description=f"the force density an exchange law scatters onto the grid's faces normal to {axis} (the "
                f"partners' forces on the medium, by its kernel), summed over every partner's law")) for axis in "xyz")
FLOW_PARTNERS = define(Interface(
    "flow.partners", TargetKind.REGISTRATION, "", None, (),
    description="the exchange partners a medium's field takes deliveries from through an exchange law (a point body's "
                "drag): each registers with the field, which then takes their scattered force on its grid"))
#: The registration kind of an exchange partner (:data:`FLOW_PARTNERS`).
EXCHANGE_PARTNER = "flow.exchange_partner"
FLOW_WAVEFORM = define(Interface(
    "flow.waveform", TargetKind.EQUALITY_CONSTRAINT, "1", "world", (7,),
    sign="on the fluid of a medium, per unit mass", magnitude=1.0,
    description="a medium's drive as its waveform: a unit direction, the steady part and one harmonic's amplitude "
                "(accelerations), frequency and phase, the force per unit mass on its fluid, which the field kind "
                "evaluates at its own instants or responds to analytically. At most one law delivers it: a waveform is "
                "not summed"))
