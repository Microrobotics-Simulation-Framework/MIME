"""The faults planted in each part the A0 emulator was extended to, and in the motor and the gradient force, for the
comparison's controls: the arm's pose, the dipole's moment and its position, the controller's latency and its law, the
motor's speed and the gradient force alone. Each is a size in units of the comparison's resolution for the quantity it
moves, so that a planted fault is above what the comparison can see by a stated factor (``test_a0_controls.py`` plants
three resolutions; ``measure_a0_controls.py`` sweeps the factor).

The resolution of a quantity is its declared tolerance over its measured response to the fault. A response that is
assumed is a declaration, not a resolution: the dipole's moment was first given the lag's response of a law
``lag ~ 1 / moment`` and the comparison's lag gate was about 18 times more sensitive than that. The responses below
that are measured are recorded by ``measure_a0_controls.py`` in ``references/a0_controls_measured.json`` under
``sensitivities``, and ``test_a0_controls.py`` checks the constants here against that record. The tracking error's
mean moves by its tolerance times its prediction in metres, and the controller's latency can only be planted in whole
steps, the smallest of which already moves the tracking error by several tolerances."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

from a0_compare import MOTOR_OUTPUT, PREDICTED, TOLERANCES

from chain_cases import AR4_BASE, MAGNET_MOMENT
from mime.nodes.actuation import KinematicMotorNode
from mime.nodes.magnetic import MagneticForceNode
from mime.units import m, ms

KEY = "3hz"
TRACKING = "tracking_error_axial_mean_m"
LAG = "phase_lag_field_minus_moment_rad"
SPEED = "swim_speed_axial_mean_velocity_m_s"
#: Metres: the tracking error's mean at 3 Hz moved by its own tolerance.
LENGTH_RESOLUTION = TOLERANCES[TRACKING] * abs(PREDICTED[KEY][TRACKING])
#: Metres: a dipole shifted along the tube moves no frame the comparison reads (the tracking error is the rotor's), but
#: moves the axial gradient force on the body and so its speed: by 3.97e-4 of itself per micrometre, by the emulation
#: (``predict_a0.emulate`` with ``altered={"magnet_at": (1e-6, 0, 0)}``), so the shift that moves the speed by its
#: tolerance is the tolerance over that.
DIPOLE_POSITION_RESOLUTION = TOLERANCES["swim_speed_axial_mean_velocity_m_s"] / 3.9716e-4 * 1e-6
#: Radians of lag per unit of relative change of the moment, measured (``sensitivities.moment`` of the record): the
#: field's torque goes as the moment, and the lag as the field's torque.
MOMENT_LAG_SENSITIVITY = 0.103
#: Relative: the dipole moment's change that moves the lag by its tolerance.
MOMENT_RESOLUTION = TOLERANCES[LAG] / MOMENT_LAG_SENSITIVITY
#: Relative: the filter's smoothing change that moves the tracking error (its lag, a time constant dt / alpha, times
#: the speed) by its tolerance.
LAW_RESOLUTION = TOLERANCES[TRACKING]
#: Relative change of the gradient force alone (the field and so the torque unmoved) that moves the mean velocity by its
#: tolerance, measured (``sensitivities.gradient_force`` of the record): about 0.3 % of the force is the least the
#: comparison sees, since the force is about 1 % of the swim speed.
GRADIENT_FORCE_RESOLUTION = 2.1e-3
#: Relative: the motor's speed change below which its own angle may not move at all. The angle is a single-precision
#: sum, each step's increment (9.4e-3 rad at 3 Hz) rounded to a whole number of units in the last place (3.8e-6 rad
#: from 32 rad, 7.6e-6 from 64, 1.5e-5 from 128, where the steady window ends), so a speed change moves the angle in
#: jumps of that unit per step or not at all: the smallest change that moves the increment by a whole unit in every
#: binade the window touches is 1.5e-5 / 9.4e-3 = 1.6e-3; a change of 1e-3 does so in all but the last, which is most
#: of the window. Measured in ``sensitivities.motor_speed`` of the record: a change of 1e-6 or 3e-6 leaves the angle
#: as it was (or moves it by one unit, which is 0.8 of the rate gate's tolerance), and 1e-3 moves the rate gate by
#: many tolerances.
MOTOR_RESOLUTION = 1.0e-3
DT = 5e-4


@contextmanager
def _gradient_scaled(factor: float) -> Iterator[None]:
    """``MagneticForceNode`` reading its gradient ``factor`` times too large: the force alone, the torque path (which
    reads the field) unmoved."""
    original = MagneticForceNode.advance

    def advance(self: Any, p: Any, state: Any, boundary_inputs: Any, dt: float) -> Any:
        given = dict(boundary_inputs)
        if "gradient" in given:
            given["gradient"] = given["gradient"] * factor
        return original(self, p, state, given, dt)

    MagneticForceNode.advance = advance  # type: ignore[method-assign]
    try:
        yield
    finally:
        MagneticForceNode.advance = original  # type: ignore[method-assign]


@contextmanager
def _motor_scaled(factor: float) -> Iterator[None]:
    """``KinematicMotorNode`` turning its rotor ``factor`` times too fast, whatever the drive commands: a fault of the
    motor, which the drive's commanded speed does not show."""
    original = KinematicMotorNode.advance

    def advance(self: Any, p: Any, state: Any, boundary_inputs: Any, dt: float) -> Any:
        given = dict(boundary_inputs)
        if "velocity" in given:
            given["velocity"] = given["velocity"] * factor
        return original(self, p, state, given, dt)

    KinematicMotorNode.advance = advance  # type: ignore[method-assign]
    try:
        yield
    finally:
        KinematicMotorNode.advance = original  # type: ignore[method-assign]


@dataclass(frozen=True)
class Plant:
    """``apply(size, sign)`` is the options that plant a fault of ``size`` resolutions on ``script_a0.a0``;
    ``patch(size, sign)``, for a fault of a node rather than of the experiment, a context in which the node is faulty;
    ``quantity`` the gated quantity it moves most."""

    name: str
    part: str
    quantity: str
    apply: Callable[[float, float], dict[str, Any]]
    resolution: str
    patch: Callable[[float, float], AbstractContextManager[None]] | None = None
    #: A fault that cannot be planted below its resolution (a whole step; a speed within one unit in the last place).
    sub_resolution: bool = True

    def planted(self, size: float, sign: float) -> AbstractContextManager[None]:
        return self.patch(size, sign) if self.patch is not None else nullcontext()


def _arm_pose(size: float, sign: float) -> dict[str, Any]:
    x, y, z = AR4_BASE
    return {"arm_at": (x + sign * size * LENGTH_RESOLUTION, y, z)}


def _dipole_position(size: float, sign: float) -> dict[str, Any]:
    # along the rotor's axis, which is the tube's: a shift across it would turn with the rotor
    return {"magnet_at": m((0.0, 0.0, sign * size * DIPOLE_POSITION_RESOLUTION))}


def _dipole_moment(size: float, sign: float) -> dict[str, Any]:
    return {"moment": MAGNET_MOMENT * (1.0 + sign * size * MOMENT_RESOLUTION)}


def _latency(size: float, sign: float) -> dict[str, Any]:
    return {"latency": ms(DT * 1e3 * max(round(size), 1))}


def _law(size: float, sign: float) -> dict[str, Any]:
    return {"smoothing": 0.005 * (1.0 + sign * size * LAW_RESOLUTION)}


def _gradient_force(size: float, sign: float) -> AbstractContextManager[None]:
    return _gradient_scaled(1.0 + sign * size * GRADIENT_FORCE_RESOLUTION)


def _motor_speed(size: float, sign: float) -> AbstractContextManager[None]:
    return _motor_scaled(1.0 + sign * size * MOTOR_RESOLUTION)


PLANTS = (
    Plant("motor turning too fast by a factor, the drive's command unmoved", "the motor's speed", MOTOR_OUTPUT,
          lambda size, sign: {}, f"{MOTOR_RESOLUTION:.3g} of itself", patch=_motor_speed, sub_resolution=False),
    Plant("gradient force scaled, the field and the torque unmoved", "the gradient force", SPEED, lambda size, sign: {},
          f"{GRADIENT_FORCE_RESOLUTION:.3g} of itself", patch=_gradient_force),
    Plant("arm base shifted along the tube", "the arm's pose", TRACKING, _arm_pose,
          f"{LENGTH_RESOLUTION:.3g} m"),
    Plant("magnet shifted along the rotor's axis, the tube's", "the dipole's position", SPEED, _dipole_position,
          f"{DIPOLE_POSITION_RESOLUTION:.3g} m"),
    Plant("magnet moment scaled", "the dipole's moment", LAG, _dipole_moment, f"{MOMENT_RESOLUTION:.3g} of itself"),
    Plant("controller latency in whole steps", "the controller's latency", TRACKING, _latency,
          "one step (the least that can be planted)", sub_resolution=False),
    Plant("filter smoothing scaled", "the controller's law", TRACKING, _law, f"{LAW_RESOLUTION:.3g} of itself"),
)
