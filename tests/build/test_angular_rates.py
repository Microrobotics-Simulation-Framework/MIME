"""A frequency (Hz, turns per second) and an angular velocity (rad/s) share a dimension and differ by 2 pi, and no site
takes one for the other. Each kind of site is built with the other kind and refused, in every profile, with both
conversions written into its own line; the kind it counts is built as a control:

- **a field's command** (a new user's case, ``"frequency": rad_per_s(5.0)``, which ran as 5 Hz), written as a
  quantity and as ``Constant(...)``;
- **a body's initial state** (the other, ``angular_velocity=Hz((5, 0, 0))``, which ran as 5 rad/s);
- **an entry's setting** (a motor's ``top_speed`` and ``initial_velocity``) and **a segment's command** (a motor's
  prescribed ``velocity``);
- **a rate** (the controller's, a frequency), and **a controller's write**, whose signal's units must be those of what
  the command counts (rad/s for a motor's velocity, Hz for a field's frequency);
- **a declaration's default** (a lab entry's), refused at registration.

Every site MIME declares is listed with the kind it counts (``SITES``), read from the registry and the motor's
prescriptions, so a new site is a decision. The controls fail on every run: a site declared as a frequency where the
table says an angular velocity (a site that takes the wrong kind) and a site that says neither are each reported, and
registration refuses an entry with a rate per second that says neither, as a setting, a command or an initial state."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
import pytest
from chain_cases import MAGNET_MOMENT, add_body, experiment, magnet_part
from control_cases import _ABSENT

from mime import Apparatus, Body, Experiment, State
from mime.build import BuildRefused
from mime.drives import Constant
from mime.dynamics import Motor, RigidInertial
from mime.dynamics.apparatus._motor import PRESCRIPTIONS
from mime.fields import PointDipole, UniformRotating
from mime.gradients import GradientStatus
from mime.interactions import Magnetic
from mime.registry import (
    PHYSICAL,
    Command,
    ControlLaw,
    Controller,
    Dynamics,
    EntryKind,
    Signal,
    dimensioned_settings,
    rate_problems,
    register_controller,
    register_dynamics,
    registered,
)
from mime.scene import Scene, Z
from mime.things import PermanentMagnet, ThingKind
from mime.units import (
    ANGULAR_VELOCITY,
    DIMENSIONLESS,
    FREQUENCY,
    TIME,
    A_m2,
    Dimension,
    Hz,
    N_m,
    Quantity,
    UnitError,
    kg,
    kg_m2,
    m,
    m_per_s,
    mT,
    ms,
    rad,
    rad_per_s,
    si,
)

ANGLE = "differ by 2 pi"


def refusals(build) -> list[Any]:
    """The refusals of a build, or none where it builds."""
    try:
        build()
    except BuildRefused as refused:
        return list(refused.refusals)
    return []


def angle_refusals(build) -> list[Any]:
    return [refusal for refusal in refusals(build) if ANGLE in refusal.message]


def field_rig(*, frequency: object = Hz(5.0), spin: Quantity = rad_per_s((0.0, 0.0, 0.0)),
              profile: str = "exploratory") -> Experiment:
    """A magnetised body turned by a uniform field, its drive and its initial spin as given: T2's sites, without its
    drag."""
    exp = Experiment("rates", scene=Scene.new(up=Z, units="m"), dt=ms(1), profile=profile)
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=spin)))
    exp.declare("umr.mass", kg(1e-6), reason="a millimetre body")
    exp.declare("umr.inertia", kg_m2((1e-12, 1e-12, 1e-12)), reason="a millimetre body")
    exp.declare("umr.magnetic_moment", A_m2((0.0, 1e-3, 0.0)), reason="a millimetre body's magnet")
    exp.add_apparatus(Apparatus("field", fields={"magnetic": UniformRotating(rotation_axis=(1.0, 0.0, 0.0),
                                                                             initial_direction=(0.0, 1.0, 0.0))},
                                commands={"frequency": frequency, "amplitude": mT(1.0)}))
    exp.attach(Magnetic.Torque(), body="umr", field="field.magnetic")
    return exp


def motor_rig(*, motor: Motor | None = None, commands: Mapping[str, object] | None = None) -> Experiment:
    """A motor alone in the world, 10 cm above the body, turning the magnet on its rotor at a prescribed velocity."""
    exp = experiment("motor_rates")
    exp.add_apparatus(Apparatus(
        "spin", chain={"motor": motor if motor is not None else Motor.Kinematic(
            axis=Z, prescribed="velocity", initial_angle=rad(0.0), at=m((0.0, 0.0, 0.1)))},
        parts={"magnet": magnet_part("motor.rotor")}, magnets={"magnet": PermanentMagnet(direction=(1.0, 0.0, 0.0))},
        fields={"magnetic": PointDipole(part="magnet")},
        commands=dict(commands) if commands is not None else {"motor.velocity": Constant(rad_per_s(6.0))}))
    exp.declare("spin.magnetic_moment", A_m2(MAGNET_MOMENT), reason="de Jongh's rotor magnet")
    add_body(exp)
    exp.attach(Magnetic.Torque(), body="umr", field="spin.magnetic")
    return exp


def test_si_refuses_each_kind_where_the_other_goes_and_takes_its_own_and_a_rate_that_knows_neither():
    with pytest.raises(UnitError, match=ANGLE) as frequency:
        si(rad_per_s(5.0), FREQUENCY, what="x")
    assert [q for q, _ in frequency.value.alternatives] == ["Hz(5.0)", "Hz(5.0 / (2 * pi))"]
    with pytest.raises(UnitError, match=ANGLE) as angular:
        si(Hz((5.0, 0.0, 0.0)), ANGULAR_VELOCITY, what="x")
    assert [q for q, _ in angular.value.alternatives] == ["rad_per_s((2 * pi * 5.0, 0.0, 0.0))",
                                                         "rad_per_s((5.0, 0.0, 0.0))"]
    assert si(Hz(5.0), FREQUENCY, what="x") == 5.0 and si(rad_per_s(5.0), ANGULAR_VELOCITY, what="x") == 5.0
    assert si(rad_per_s(1.0) + Hz(1.0), FREQUENCY, what="x") == 2.0, "a sum knows neither, and is taken as the site's"
    assert FREQUENCY == ANGULAR_VELOCITY and hash(FREQUENCY) == hash(ANGULAR_VELOCITY), "one dimension"
    assert str(rad_per_s(5.0)) == "5.0 rad/s" and str(Hz(5.0)) == "5.0 Hz"
    # each conversion written is the number the other reading gives
    assert eval(frequency.value.alternatives[1][0], {"Hz": Hz, "pi": math.pi}).value == pytest.approx(5.0 / (2 * math.pi))
    assert eval(angular.value.alternatives[0][0], {"rad_per_s": rad_per_s, "pi": math.pi}).value[0] == \
        pytest.approx(10.0 * math.pi)


@pytest.mark.parametrize("profile", ["exploratory", "research"])
@pytest.mark.parametrize("written", [rad_per_s(5.0), Constant(rad_per_s(5.0))], ids=["quantity", "constant"])
def test_a_fields_frequency_in_rad_per_s_is_refused_with_hz_to_write(profile, written):
    (refused,) = angle_refusals(lambda: field_rig(frequency=written, profile=profile).build())
    assert refused.code == "setting.units" and "frequency (Hz)" in refused.message and "5.0 rad/s" in refused.message
    assert refused.action.endswith(": commands={'frequency': Hz(5.0)} if you meant turns per second, or "
                                   "commands={'frequency': Hz(5.0 / (2 * pi))} for the radians per second written "
                                   "(pi from math)")


@pytest.mark.parametrize("profile", ["exploratory", "research"])
def test_a_bodys_initial_spin_in_hz_is_refused_with_rad_per_s_to_write(profile):
    (refused,) = angle_refusals(lambda: field_rig(spin=Hz((5.0, 0.0, 0.0)), profile=profile).build())
    assert refused.code == "state.units" and "angular velocity (rad/s)" in refused.message
    assert ": State(angular_velocity=rad_per_s((2 * pi * 5.0, 0.0, 0.0))) for the turns per second written" in \
        refused.action
    assert "State(angular_velocity=rad_per_s((5.0, 0.0, 0.0))) if you meant radians per second" in refused.action


def test_control_the_field_rig_in_its_own_kinds_builds_and_turns_at_the_frequency_written():
    built = field_rig(frequency=Hz(5.0), spin=rad_per_s((1.0, 0.0, 0.0))).build()
    assert built.report.refusals == ()
    assert refusals(lambda: field_rig(frequency=Constant(Hz(5.0))).build()) == []


@pytest.mark.parametrize("setting", ["top_speed", "initial_velocity"])
def test_a_motors_angular_velocity_setting_in_hz_is_refused(setting):
    motor = Motor.Dynamic(axis=Z, initial_angle=rad(0.0), inertia=kg_m2(1e-5), at=m((0.0, 0.0, 0.1)),
                          **{setting: Hz(10.0)})
    found = angle_refusals(lambda: motor_rig(motor=motor, commands={"motor.torque": Constant(N_m(0.0))}).build())
    (refused,) = [r for r in found if f"setting {setting!r}" in r.message]
    assert refused.code == "setting.units" and \
        f": {setting}=rad_per_s(2 * pi * 10.0) for the turns per second written" in refused.action
    # the control: in rad/s, nothing is refused for its angle
    held = Motor.Dynamic(axis=Z, initial_angle=rad(0.0), inertia=kg_m2(1e-5), at=m((0.0, 0.0, 0.1)),
                         **{setting: rad_per_s(10.0)})
    assert angle_refusals(lambda: motor_rig(motor=held, commands={"motor.torque": Constant(N_m(0.0))}).build()) == []


def test_a_motors_prescribed_velocity_in_hz_is_refused_and_in_rad_per_s_builds():
    (refused,) = angle_refusals(lambda: motor_rig(commands={"motor.velocity": Constant(Hz(1.0))}).build())
    assert refused.code == "setting.units" and "'motor.velocity'" in refused.message
    assert ": commands={'motor.velocity': rad_per_s(2 * pi * 1.0)} for the turns per second" in refused.action
    assert refusals(lambda: motor_rig().build()) == []


@register_controller
@dataclass(frozen=True, kw_only=True)
class HertzSpeed(Controller):
    """Writes a motor's velocity, its write declared in Hz: the wrong units for an angular velocity."""

    name = "oracle.hertz_speed"
    version = "0.1.0"
    differentiability = GradientStatus.CUT
    gradient = _ABSENT
    builds = {"": ()}
    reads = {"pose": Signal("m; 1", (7,))}
    writes = {"speed": Signal("Hz", ())}

    def law(self, ctx: Any) -> ControlLaw:
        return ControlLaw(state={}, held={"speed": np.asarray(0.0)},
                          step=lambda state, reads: (state, {"speed": np.asarray(1.0)}))


@register_controller
@dataclass(frozen=True, kw_only=True)
class RadianSpeed(HertzSpeed):
    """The same, its write declared in rad/s."""

    name = "oracle.radian_speed"
    writes = {"speed": Signal("rad/s", ())}


def controlled(controller: Controller, *, rate: object = ms(3)) -> Experiment:
    exp = motor_rig(commands={})
    exp.set_controller(controller, reads={"pose": "umr.pose"}, writes={"speed": "spin.motor.velocity"},
                       mode="callback", rate=rate)
    return exp


def test_a_controllers_write_to_an_angular_velocity_in_hz_is_refused_and_in_rad_per_s_is_taken():
    (refused,) = angle_refusals(lambda: controlled(HertzSpeed()).build())
    assert refused.code == "controller.signal" and "takes 'rad/s'" in refused.message
    assert "Signal('rad/s', ...)" in refused.action
    assert [r for r in refusals(lambda: controlled(RadianSpeed()).build()) if r.code == "controller.signal"] == []


def test_a_controllers_rate_in_rad_per_s_is_refused_and_in_hz_is_taken():
    (refused,) = angle_refusals(lambda: controlled(RadianSpeed(), rate=rad_per_s(2000.0 * math.pi / 3.0)).build())
    assert refused.code == "run.rate-units"
    assert ": rate=Hz(2094.3951023931954) if you meant turns per second, or rate=Hz(2094.3951023931954 / (2 * pi))" \
        in refused.action
    assert [r for r in refusals(lambda: controlled(RadianSpeed(), rate=Hz(1000.0 / 3.0)).build())
            if r.code.startswith("run.rate")] == []


def test_an_entrys_default_of_the_other_kind_is_refused_at_registration():
    @dataclass(frozen=True, kw_only=True)
    class Spinning(Dynamics):
        name = "toy.spinning_default"
        version = "0.1.0"
        setting_roles = {"spin": PHYSICAL}
        differentiability = GradientStatus.EXACT
        references = ()
        builds = {"": ()}
        applies_to = frozenset({ThingKind.BODY})
        spin: Annotated[Quantity, ANGULAR_VELOCITY] = Hz(1.0)

    with pytest.raises(TypeError, match=ANGLE):
        register_dynamics(Spinning)


#: Every site of MIME's that takes a rate per second, with what it counts: (kind, entry, where) -> angle.
SITES = {
    ("dynamics", "rigid_inertial", "initial angular_velocity"): "radian",
    ("dynamics", "motor", "setting initial_velocity"): "radian",
    ("dynamics", "motor", "setting top_speed"): "radian",
    ("dynamics", "motor", "prescribed velocity"): "radian",
    ("dynamics", "arm", "setting initial_velocities"): "radian",
    ("field provider", "uniform_rotating", "command frequency"): "turn",
    ("controller", "controller", "rate"): "turn",
}


def rate_sites(entries) -> dict[tuple[str, str, str], str | None]:
    """The rate-per-second sites ``entries`` (``(kind, name, class)``) declare, each with the angle it counts: its
    dimensioned settings, its commands and its initial state."""
    found: dict[tuple[str, str, str], str | None] = {}

    def held(kind: str, name: str, where: str, dimension: object) -> None:
        if isinstance(dimension, Dimension) and dimension == FREQUENCY:
            found[(kind, name, where)] = dimension.angle

    for kind, name, cls in entries:
        hints = dataclasses.fields(cls) if dataclasses.is_dataclass(cls) else ()
        settings, _ = dimensioned_settings(cls) if hints else ({}, [])
        for setting, declared in settings.items():
            held(kind, name, f"setting {setting}", declared.dimension)
        for command, spec in dict(getattr(cls, "commands", None) or {}).items():
            held(kind, name, f"command {command}", getattr(spec, "dimension", None))
        initial = getattr(cls, "initial_state", None)
        for state, dimension in dict(initial if isinstance(initial, Mapping) else {}).items():
            held(kind, name, f"initial {state}", dimension)
    return found


def mime_sites() -> dict[tuple[str, str, str], str | None]:
    entries = [(record.kind.value, record.name, record.cls) for kind in EntryKind for record in registered(kind)
               if record.namespace == "mime"]
    found = rate_sites(entries)
    found |= {("dynamics", "motor", f"prescribed {name}"): dimension.angle
              for name, dimension in PRESCRIPTIONS.items() if dimension == FREQUENCY}
    found[("controller", "controller", "rate")] = FREQUENCY.angle  # the planner's divider takes a rate as a frequency
    return found


def site_problems(found: Mapping[tuple[str, str, str], str | None]) -> list[str]:
    problems = [f"{site}: says neither turns nor radians" for site, angle in found.items() if angle is None]
    problems += [f"{site}: counts {angle}, where the table says {SITES[site]}" for site, angle in found.items()
                 if angle is not None and site in SITES and SITES[site] != angle]
    problems += [f"{site}: a new site, not in the table" for site in found if site not in SITES]
    problems += [f"{site}: in the table, and no longer a site" for site in SITES if site not in found]
    return problems


def test_every_rate_site_of_mimes_counts_what_the_table_says():
    assert site_problems(mime_sites()) == []


def test_control_a_planted_site_that_takes_the_wrong_kind_or_neither_is_reported():
    @dataclass(frozen=True, kw_only=True)
    class Planted:
        initial_state: ClassVar[Mapping[str, Dimension]] = {"angular_velocity": FREQUENCY}

    planted = mime_sites() | rate_sites([("dynamics", "rigid_inertial", Planted)])
    assert site_problems(planted) == ["('dynamics', 'rigid_inertial', 'initial angular_velocity'): counts turn, where "
                                      "the table says radian"]
    bare = mime_sites() | {("dynamics", "motor", "prescribed velocity"): Dimension(FREQUENCY.exponents).angle}
    assert site_problems(bare) == ["('dynamics', 'motor', 'prescribed velocity'): says neither turns nor radians"]


#: A rate per second that says neither turns nor radians.
BARE = DIMENSIONLESS / TIME


def test_control_registration_refuses_a_rate_per_second_that_says_neither():
    @dataclass(frozen=True, kw_only=True)
    class BareSetting(Dynamics):
        name = "toy.bare_rate_setting"
        version = "0.1.0"
        spin: Annotated[Quantity, BARE]

    class BareCommand:
        commands: ClassVar[Mapping[str, Command]] = {"spin": Command("spin", BARE)}

    class BareState:
        initial_state: ClassVar[Mapping[str, Dimension]] = {"spin": BARE}

    assert any("says neither turns nor radians" in p for p in dimensioned_settings(BareSetting)[1])
    assert rate_problems(BareCommand) == ["its command 'spin' is declared with a rate per second that says neither "
                                          "turns nor radians: say which rate it is: mime.units.Frequency for a "
                                          "frequency in Hz (turns per second), or mime.units.AngularVelocity for an "
                                          "angular velocity in rad/s; the two differ by 2 pi"]
    assert len(rate_problems(BareState)) == 1 and "its initial 'spin'" in rate_problems(BareState)[0]
    with pytest.raises(TypeError, match="says neither turns nor radians"):
        register_dynamics(BareSetting)
