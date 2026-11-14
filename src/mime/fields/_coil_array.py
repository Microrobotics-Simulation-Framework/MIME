"""Coil arrays: electromagnets whose field is linear in the currents they carry, plus a uniform background. Here now:
``CoilArray``, the contract every kind of array keeps, and ``CalibratedMultipole``, an array whose coils are a
calibrated multipole model; the field and gradient per unit current at a point (``Actuation``), and currents for a
wanted field and gradient within declared limits (``allocate``)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache, partial
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, cast, override

import jax
import numpy as np
from numpy.typing import NDArray

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.core import run_dtype
from mime.data import Asset, asset_bytes, asset_problem
from mime.gradients import GradientStatus
from mime.nodes.actuation import CalibratedMultipoleNode, Calibration, CalibrationError, read_calibration
from mime.registry import (
    DESIGN,
    NUMERICAL,
    PHYSICAL,
    BuildContext,
    Command,
    FieldProvider,
    Flag,
    SettingRole,
    convention,
    register_field_provider,
)
from mime.units import (
    CURRENT,
    FIELD_GRADIENT,
    LENGTH,
    MAGNETIC_FLUX_DENSITY,
    A,
    Dimension,
    Length,
    MagneticFluxDensity,
    Magnitude,
    Quantity,
    T,
    T_per_m,
    m,
    si,
)
from mime.vocabulary import FLUX_DENSITY, GRADIENT, Capability

if TYPE_CHECKING:
    from mime.build import ProviderBuilder, Slot

@cache
def _host_allocation(rows: int, coils: int) -> Any:
    """The exact allocation compiled for ``rows`` x ``coils``, run on the host in float64 by :func:`allocate`
    (``mime.control`` imported here: it imports the build, which imports this module)."""
    from mime.control import exact_allocation

    return cast(Any, jax).jit(partial(exact_allocation, owner="allocation.active_set"))

#: The calibrations the calibrated multipole reads, by their packaged asset's name (``asset("navion_2")``).
CALIBRATIONS = ("cardiomag", "navion_2", "octomag")
#: The declared guard on precision: the largest share of the field that rounding in float32 may cost, ``eps32 / c``
#: over the declared region (``c`` the field cancellation). The 0.2 line's number: it admits Navion_2 (1.5e-5
#: predicted) and CardioMag (4.8e-7) and refuses OctoMag (1.7e-3, 1.2e-3 measured on its gradient, which would land on
#: a gradient pull's force whole).
FLOAT32_BUDGET = 1e-4
#: How far from symmetric and trace-free a wanted gradient may be, as a share of its largest entry: a field with no
#: current where it is read has a symmetric, trace-free gradient, and no currents make another.
_MAXWELL = 1e-9


@cache
def _read(name: str, sha256: str) -> Calibration:
    return read_calibration(asset_bytes(Asset(name, sha256)))


@cache
def _gradient_limit(name: str, sha256: str) -> tuple[float, ...]:
    return tuple(float(c) for c in _read(name, sha256).gradient_limit())


def calibration_of(held: Asset) -> Calibration:
    """The calibration a packaged asset holds, read once."""
    return _read(held.name, held.sha256)


def _unit(quaternion: Sequence[float]) -> tuple[float, float, float, float]:
    length = math.sqrt(sum(c * c for c in quaternion))
    w, x, y, z = (c / length for c in quaternion)
    return (w, x, y, z)


def _turn(quaternion: Sequence[float]) -> NDArray[np.float64]:
    w, x, y, z = _unit(quaternion)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _gradient5(gradient: NDArray[np.float64]) -> NDArray[np.float64]:
    """A symmetric, trace-free gradient's five independent entries, ``[Gxx, Gxy, Gxz, Gyy, Gyz]``, along its first two
    axes (mag_manip's packing)."""
    return np.stack([gradient[0, 0], gradient[0, 1], gradient[0, 2], gradient[1, 1], gradient[1, 2]])


@dataclass(frozen=True)
class Actuation:
    """What an array gives at one point of the world, float64: ``field`` ``(3, K)``, each coil's field per unit current
    (T/A); ``gradient`` ``(3, 3, K)``, each coil's ``dB_i/dx_j`` per unit current (T/(m A)); and ``background``, the
    field at zero current (T). The field at currents ``i`` is ``field @ i + background``, its gradient
    ``gradient @ i``."""

    point: tuple[float, float, float]
    field: NDArray[np.float64]
    gradient: NDArray[np.float64]
    background: NDArray[np.float64]

    @property
    def coils(self) -> int:
        return int(self.field.shape[1])

    def at(self, currents: Quantity) -> tuple[Quantity, Quantity]:
        """The field (T) and its gradient (T/m, nine entries, row by row) at ``currents`` (A, one per coil)."""
        held = np.asarray(si(currents, CURRENT, what="the currents"), dtype=np.float64).reshape(-1)
        if held.shape != (self.coils,):
            raise ArgumentRefused([Refusal("allocation.wanted", RefusalClass.USER,
                                           f"{held.size} currents for an array of {self.coils} coils",
                                           f"give one current per coil: A((...)), {self.coils} of them")])
        return (T(tuple(self.field @ held + self.background)),
                T_per_m(tuple((self.gradient @ held).reshape(-1))))

    def matrix(self) -> NDArray[np.float64]:
        """The actuation matrix ``(8, K)``: the field's three rows (T/A) over the gradient's five independent ones
        (T/(m A), ``[Gxx, Gxy, Gxz, Gyy, Gyz]``)."""
        return np.concatenate([self.field, _gradient5(self.gradient)])


@dataclass(frozen=True)
class Allocation:
    """Currents from :func:`allocate`, and what they make: ``currents`` (A); the ``field`` (T) and ``gradient`` (T/m,
    nine entries) they make; each wanted quantity's miss, ``field_miss`` (T, the norm of the difference) and
    ``gradient_miss`` (T/m, over the five independent entries), ``None`` where it was not asked; which coils sit at a
    declared limit (``at_limit``); and the rank of the rows asked against the coils (``rank``, ``coils``): below the
    rows asked, the wanted field and gradient cannot all be set apart."""

    currents: Quantity
    field: Quantity
    gradient: Quantity
    field_miss: float | None
    gradient_miss: float | None
    at_limit: tuple[bool, ...]
    rank: int
    rows: int

    def text(self) -> str:
        misses = [f"the field missed by {self.field_miss:.3g} T"] if self.field_miss is not None else []
        misses += [f"the gradient by {self.gradient_miss:.3g} T/m"] if self.gradient_miss is not None else []
        limited = [str(k) for k, held in enumerate(self.at_limit) if held]
        return (f"currents {tuple(round(c, 6) for c in cast(tuple[float, ...], self.currents.value))} A; "
                f"{', '.join(misses)}; coils at their limit: {', '.join(limited) or 'none'}; rank {self.rank} of "
                f"{self.rows} rows asked")


def _scale(given: Quantity | None, unit: Dimension, name: str, example: str) -> float:
    """A row scale in SI, one above zero; ``1`` where none is given (the unweighted rows)."""
    if given is None:
        return 1.0
    values = np.asarray(si(given, unit, what=name), dtype=np.float64).reshape(-1)
    held = float(values[0]) if values.size == 1 else float("nan")
    if not (math.isfinite(held) and held > 0.0):
        raise ArgumentRefused([Refusal("allocation.wanted", RefusalClass.USER,
                                       f"{name} {given} is not one value above zero",
                                       f"give {name}={example}: one quantity above zero, the size at which a miss in "
                                       f"its rows counts as much as one in the others'")])
    return held


def allocate(actuation: Actuation, *, limits: Quantity, field: Quantity | None = None,
             gradient: Quantity | None = None, field_scale: Quantity | None = None,
             gradient_scale: Quantity | None = None) -> Allocation:
    """The currents, each within its declared limit (``limits``: one current for every coil, or one per coil), that
    come nearest the wanted ``field`` (T, three entries) and ``gradient`` (T/m, nine entries row by row,
    ``dB_i/dx_j``), in least squares: the field's rows and the gradient's five independent ones, as mag_manip's
    backward model stacks them, each divided by its scale. No system ships limits: the hardware's are the user's to
    declare. A wanted gradient must be symmetric and trace-free, as every gradient of a field with no current where it
    is read is. ``field_scale`` (T) and ``gradient_scale`` (T/m) weigh the rows: a miss of one scale in either counts
    the same. With neither, the rows are unweighted, in T and T/m, so out of reach the gradient (a hundred times the
    field's scale for 10 mT against 1 T/m) is favoured over the field: give both scales, or ask for one quantity alone
    when it must win. With both quantities asked, give both scales or neither. A wanted value that is not finite is
    refused. It runs on the host, in float64 whatever the run's dtype, through :func:`mime.control.exact_allocation`
    (exact by active sets, at most :data:`mime.control.EXACT_COMMANDS` coils): no derivative passes through it; in a
    controller's law, call that function."""
    if field is None and gradient is None:
        raise ArgumentRefused([Refusal("allocation.wanted", RefusalClass.USER,
                                       "allocate() is asked for neither a field nor a gradient",
                                       "give field=, gradient= or both")])
    if field is not None and gradient is not None and (field_scale is None) != (gradient_scale is None):
        raise ArgumentRefused([Refusal("allocation.wanted", RefusalClass.USER,
                                       "allocate() is given one row scale of two: the field's and the gradient's rows "
                                       "are weighed against each other only through both",
                                       "give field_scale= and gradient_scale= both, or neither (the unweighted rows)")])
    bounds = np.asarray(si(limits, CURRENT, what="the limits"), dtype=np.float64).reshape(-1)
    if bounds.size == 1:
        bounds = np.full(actuation.coils, float(bounds[0]))
    if bounds.shape != (actuation.coils,) or not np.all(np.isfinite(bounds) & (bounds > 0.0)):
        raise ArgumentRefused([Refusal("allocation.wanted", RefusalClass.USER,
                                       f"limits {bounds.tolist()} A: an array of {actuation.coils} coils takes one "
                                       f"limit above zero, or one per coil",
                                       "declare the hardware's limits: limits=A(...), one for every coil or one per "
                                       "coil")])
    for name, wanted in (("field", field), ("gradient", gradient)):
        if wanted is not None and not np.all(np.isfinite(np.asarray(wanted.value, dtype=np.float64))):
            raise ArgumentRefused([Refusal("allocation.wanted", RefusalClass.USER,
                                           f"the wanted {name} is not finite", f"give finite numbers for {name}=")])
    rows: list[NDArray[np.float64]] = []
    targets: list[NDArray[np.float64]] = []
    scales: list[NDArray[np.float64]] = []
    wanted_field = wanted_gradient = None
    if field is not None:
        wanted_field = np.asarray(si(field, MAGNETIC_FLUX_DENSITY, what="the wanted field"), dtype=np.float64)
        if wanted_field.shape != (3,):
            raise ArgumentRefused([Refusal("allocation.wanted", RefusalClass.USER,
                                           f"the wanted field has {wanted_field.size} entries",
                                           "give three: mT((x, y, z))")])
        rows.append(actuation.field)
        targets.append(wanted_field - actuation.background)
        scales.append(np.full(3, _scale(field_scale, MAGNETIC_FLUX_DENSITY, "field_scale", "mT(10)")))
    if gradient is not None:
        flat = np.asarray(si(gradient, FIELD_GRADIENT, what="the wanted gradient"), dtype=np.float64)
        if flat.shape != (9,):
            raise ArgumentRefused([Refusal("allocation.wanted", RefusalClass.USER,
                                           f"the wanted gradient has {flat.size} entries",
                                           "give nine, row by row: mT_per_m((...))")])
        wanted_gradient = flat.reshape(3, 3)
        scale = float(np.max(np.abs(wanted_gradient))) or 1.0
        if np.max(np.abs(wanted_gradient - wanted_gradient.T)) > _MAXWELL * scale or \
                abs(np.trace(wanted_gradient)) > _MAXWELL * scale:
            raise ArgumentRefused([Refusal("allocation.wanted", RefusalClass.USER,
                                           "the wanted gradient is not symmetric and trace-free, and no currents "
                                           "make one that is not: a field with no current where it is read has a "
                                           "symmetric, trace-free gradient",
                                           "give gradient=mT_per_m((...)), dB_i/dx_j row by row, with G_ij = G_ji "
                                           "and G_xx + G_yy + G_zz = 0")])
        rows.append(_gradient5(actuation.gradient))
        targets.append(_gradient5(wanted_gradient))
        scales.append(np.full(5, _scale(gradient_scale, FIELD_GRADIENT, "gradient_scale", "T_per_m(1)")))
    matrix, target, weights = np.concatenate(rows), np.concatenate(targets), np.concatenate(scales)
    from mime.control import EXACT_COMMANDS

    if actuation.coils > EXACT_COMMANDS:
        raise ArgumentRefused([Refusal("allocation.exact-size", RefusalClass.USER,
                                       f"allocate() on an array of {actuation.coils} coils: its exact allocation "
                                       f"takes at most {EXACT_COMMANDS}",
                                       f"allocate {EXACT_COMMANDS} of the coils with mime.control.exact_allocation(...) "
                                       f"on their columns of actuation.matrix(), the others held; or wait for the "
                                       f"iterative path, which takes any number (a later 0.3.0 development release)")])
    with cast(Any, jax).enable_x64(True):
        solved = _host_allocation(*matrix.shape)(matrix, target, lower=-bounds, upper=bounds, scales=weights)
        currents: NDArray[np.float64] = np.asarray(solved.commands, dtype=np.float64)
    made_field = actuation.field @ currents + actuation.background
    made_gradient = actuation.gradient @ currents
    return Allocation(
        A(tuple(currents)), T(tuple(made_field)), T_per_m(tuple(made_gradient.reshape(-1))),
        None if wanted_field is None else float(np.linalg.norm(made_field - wanted_field)),
        None if wanted_gradient is None else float(np.linalg.norm(_gradient5(made_gradient - wanted_gradient))),
        tuple(bool(abs(c) >= b * (1.0 - 1e-9)) for c, b in zip(currents, bounds, strict=True)),
        int(np.linalg.matrix_rank(matrix)), int(matrix.shape[0]))


#: The background's text, which every array's validity says.
_BACKGROUND = ("background, a uniform field (T, world frame) added to the coil field, is zero unless declared: the "
               "calibrations are fitted to the coil field alone, so a nonzero default would break parity with them; "
               "Earth's field, some 50 uT, is 0.1 to 1 % of a 5 to 50 mT field and has no gradient, so it turns a "
               "fixed moment and never pulls one")


@dataclass(frozen=True, kw_only=True)
class CoilArray(FieldProvider):
    """The contract every kind of coil array keeps: a field linear in ``currents``, the command (A, one per coil) that
    reaches the node as ``delivered_currents``, the current each coil carries, so that a stage between a command and
    the coils can sit there with no call site changed; plus ``background``, a uniform field (T, world frame), zero
    unless declared. The array is placed by one exact transform, ``at`` and ``orientation`` (a unit quaternion ``(w, x,
    y, z)``), its own frame in the world. It publishes the flux density and its gradient per reader slot, at the
    reader's point (a body's magnet's centroid where the read names it), and gives the field and gradient per unit
    current at a point (:meth:`actuation`), for :func:`allocate`. Not registered: each kind is (``CalibratedMultipole``
    now; coils from their geometry later, on this contract)."""

    setting_roles: ClassVar[Mapping[str, SettingRole]] = {
        "at": convention(PHYSICAL, "an array's own frame is the world's unless it is placed"),
        "orientation": convention(PHYSICAL, "an array's own frame is not turned in the world unless it is turned"),
        "background": convention(PHYSICAL, "a calibration is fitted to its coils' field alone, so the field it gives "
                                           "has no background unless one is declared")}
    differentiability = GradientStatus.EXACT
    flags = frozenset({Capability.PROVISION_EVALUATED, Capability.BODY_POINTS})
    publishes = {FLUX_DENSITY.name: "flux_density", GRADIENT.name: "gradient"}
    commands = {"currents": Command("delivered_currents", CURRENT, (-1,))}
    at: Annotated[Quantity, Length] = m((0.0, 0.0, 0.0))
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    background: Annotated[Quantity, MagneticFluxDensity] = T((0.0, 0.0, 0.0))

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found: list[str] = []
        turn = cast(object, self.orientation)
        if not (isinstance(turn, tuple) and len(cast(tuple[object, ...], turn)) == 4 and
                all(isinstance(c, (int, float)) for c in cast(tuple[object, ...], turn))) or \
                abs(math.sqrt(sum(float(c) ** 2 for c in cast(tuple[float, ...], turn))) - 1.0) > 1e-9:
            found.append(f"orientation {turn!r} is not a unit quaternion (w, x, y, z): it turns the array's own frame "
                         f"into the world's, so write four numbers of length one")
        for name in ("at", "background"):
            value = settings.get(name)
            if not (isinstance(value, tuple) and len(value) == 3):
                found.append(f"{name} {value!r} is not three numbers")
        return found

    def placement(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """``at`` (m), the turn of ``orientation`` as a matrix, and ``background`` (T), float64."""
        return (np.asarray(si(self.at, LENGTH, what="at"), dtype=np.float64), _turn(self.orientation),
                np.asarray(si(self.background, MAGNETIC_FLUX_DENSITY, what="background"), dtype=np.float64))

    def actuation(self, point: Quantity) -> Actuation:
        """Each coil's field and gradient per unit current at the world ``point`` (a length, three entries), float64,
        and the background there."""
        raise NotImplementedError


@register_field_provider
@dataclass(frozen=True, kw_only=True)
class CalibratedMultipole(CoilArray):
    """An array whose coils are a calibrated multipole model of order one (MPEM): each coil's field per unit current is
    the sum of a few fitted point sources, ``b = sum_j beta_j (3 (d_j . r) r / r^5 - d_j / r^3)``, from
    ``calibration``, one of the calibrations MIME packages (ETH MSRL's published calibrations of Navion_2, OctoMag and
    CardioMag, evaluated by MIME: ``asset("navion_2")``, ``asset("octomag")``, ``asset("cardiomag")``), its coefficients and sources the node's leaves, read from the file. The calibration's own
    frame is placed by ``at`` and ``orientation``.

    Each step it raises ``outside_region`` where a reader is outside the calibration's ``Workspace_Dimensions``, a
    declared region, not a fit domain, and ``near_source`` where a coil's gradient per unit current at a reader is more
    than 30 times its median over that region (a source's own term, near the source). ``source_clearance``, a length,
    replaces that rule with a distance from a fitted source, and zero removes the guard. In float32 it is refused where
    rounding would cost more than :data:`FLOAT32_BUDGET` of the field (OctoMag)."""

    name = "calibrated_multipole"
    version = "0.1.0"
    setting_roles = {
        "calibration": DESIGN,
        "source_clearance": convention(NUMERICAL, "none given: the guard is the gradient's, a coil's gradient per "
                                                        "unit current above a declared multiple of its median over "
                                                        "the declared region (the validity line states it), each "
                                                        "calibration's own; a length given replaces it with a "
                                                        "distance from a fitted source")}
    references = ("Petruska2017", "Charreyron2021", "Kummer2010", "Zughaibi2025", "Zughaibi2026", "Gervasoni2024", "Yu2019")
    builds = {"": (CalibratedMultipoleNode,)}
    validity = (
        "the field is linear in the delivered currents, each coil a calibrated multipole model of order one, ETH "
        "MSRL's published calibration evaluated by MIME: fitted point sources, basis functions and not hardware (Source_Position need not be a coil's, and a fit may place "
        "one far out)",
        "Workspace_Dimensions is a declared region, never fitted: a reader outside it raises outside_region, outside "
        "the declared region, which is no evidence of extrapolation, and inside it is no evidence of accuracy; the "
        "fit's own domain is not published",
        "the field is singular at a fitted source, where a source's own term takes the gradient to many times its "
        "interior scale: near_source is raised where a coil's gradient per unit current at a reader is more than 30 "
        "times its median over the declared region (the factor is MIME's declaration, set from a measurement over "
        "60 000 drawn points of each region: Navion_2 and "
        "CardioMag nowhere, their smooth gradient reaching 13.5 and 2.4 times the median; OctoMag at 19, a pocket "
        "about 1.2 mm across around its fitted source, which sits 20 um outside the region's -x face and whose own "
        "gradient there is 10^8 times the median; the region's nearest grid corner, 1 mm from it, reads 86 times); "
        "source_clearance, a length, replaces the rule with a distance from a fitted source, and zero removes the "
        "guard",
        "float32 is refused where rounding at its eps over the field's cancellation in the declared region exceeds "
        "the declared guard, 1e-4 of the field: OctoMag; Navion_2 and CardioMag build (each card's conditioning); on a "
        "GPU set JAX_DEFAULT_MATMUL_PRECISION=highest",
        "the coil field follows mag_manip's sign convention, which its oracle pins in software; ETH's measured "
        "Navion_2 data disagree with ETH MSRL's published calibration by a global sign, and which side is flipped is "
        "not known: gradient pull "
        "on a field-aligned body does not change with it, the torque on a fixed moment does, so establish the "
        "convention on the hardware before relying on a direction of rotation (MIME-ANO-153)",
        "agreement with mag_manip is a software oracle's: it cannot ground sign or frame against hardware",
        "the currents are ideal and unlimited here: allocate() takes the limits the user declares, and no calibration "
        "ships one, so a limit is MIME's declaration with its source and never a maker's specification; cited figures, none of them a rating of the rig a calibration was fitted on (which neither source "
        "states): OctoMag 16 A a coil, a published operating limit of ETH's rig (Zughaibi2025), and a 15 A amplifier "
        "saturation its 2010 force tables assume (Kummer2010); Navion_2 25 A a coil as one lab's operating restriction (Zughaibi2025, Sec. III-B: \"we conservatively "
        "restrict them to 25 A per coil for safety\"), the figure MIME declares; 45 A as what that lab says the hardware "
        "supports (Zughaibi2025, Sec. III-B, and its sequel Zughaibi2026, Sec. II-C, the same three-coil system and "
        "drivers); the drivers rated 50 A continuous and 100 A peak, with the system paper's workspace maps at 50 A in "
        "each coil (Gervasoni2024, Experimental Section and Fig. 3); no source states a coil rating, or that its unit is "
        "the one the calibration was fitted on; CardioMag 35 A a coil, with 15 kW for the whole system, as the maximum "
        "its eight electromagnets are given (Yu2019, Sec. III-A), nothing stating that its unit is the one the "
        "calibration was fitted on, and the power cap not modelled here (the coils' resistance is unknown)",
        _BACKGROUND,
        "CardioMag's file carries an Offset block of eight sources, current-independent, worth about 1.5 mT at the "
        "median over its region (3 to 30 % of a 5 to 50 mT field), which neither mag_manip nor this node evaluates: "
        "upstream loads the block only when Coil_List names it, and this file's does not (a Coil_List that names it is "
        "refused here); background is not it, the sources being position dependent; Navion_2's and OctoMag's files "
        "carry no Offset block",
        "not in this release: saturation of the cores (no shipped calibration has fitted it, all were fitted below "
        "core saturation, and it makes the field nonlinear in current: it waits for a calibration "
        "that fits it, or for the response and circuit stage); a calibration of order above one (none ships; the reader refuses "
        "one); a lab's own calibration file (the three packaged ones only); a coil's circuit (the currents delivered "
        "are ideal)",
    )
    raises = (Flag("outside_region", "a reader lay outside the calibration's declared region (Workspace_Dimensions), "
                                     "which is not a fit domain: outside the declared region, not extrapolation"),
              Flag("near_source", "a reader came nearer a fitted source than source_clearance, where the field's "
                                  "sources are singular"))
    calibration: Asset | None = None
    source_clearance: Annotated[Quantity | None, Length] = None

    def _held(self) -> Calibration:
        return calibration_of(cast(Asset, self.calibration))

    def calibration_problem(self) -> str | None:
        """Why the calibration cannot be read, or ``None``."""
        held = cast(object, self.calibration)  # a script is not type-checked
        if held is None:
            return ("the calibrated multipole reads its calibration: write calibration=asset('navion_2') (or "
                    "'octomag', 'cardiomag'), with asset from mime.data")
        if not isinstance(held, Asset):
            return (f"calibration={held!r} is not a packaged asset: a calibration is named by asset(...) from "
                    f"mime.data, one of {list(CALIBRATIONS)}")
        if held.name not in CALIBRATIONS:
            return f"calibration={held.name!r} is no calibration's; those packaged are {list(CALIBRATIONS)}"
        problem = asset_problem(held)
        if problem is not None:
            return f"calibration: {problem}"
        try:
            self._held()
        except CalibrationError as error:
            return f"calibration {held.name!r}: {error}"
        return None

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found = super().setting_problems(settings)
        problem = self.calibration_problem()
        if problem is not None:
            return [*found, problem]
        held = self._held()
        clearance = settings.get("source_clearance")
        if clearance is not None and not (isinstance(clearance, float) and clearance >= 0.0):
            found.append(f"source_clearance {clearance!r} m is not a length from zero on")
        if np.dtype(run_dtype()) == np.float32:
            floor = held.floor(float(np.finfo(np.float32).eps))
            if floor > FLOAT32_BUDGET:
                found.append(f"the {held.system} calibration is refused in float32: its fitted sources cancel to "
                             f"{held.cancellation():.2g} of their sum in its declared region, so float32's rounding "
                             f"costs {floor:.2g} of the field, above the guard of {FLOAT32_BUDGET:g}, so write "
                             f"JAX_ENABLE_X64=1 (or jax.config.update('jax_enable_x64', True) before the first build) "
                             f"to run in float64")
        return found

    @override
    def actuation(self, point: Quantity) -> Actuation:
        held = self._held()
        at, turn, background = self.placement()
        where = np.asarray(si(point, LENGTH, what="the point"), dtype=np.float64)
        if where.shape != (3,):
            raise ArgumentRefused([Refusal("allocation.wanted", RefusalClass.USER,
                                           f"the point has {where.size} entries", "give three: m((x, y, z))")])
        fields, gradients = held.unit_fields(turn.T @ (where - at))
        return Actuation((float(where[0]), float(where[1]), float(where[2])), turn @ fields,
                         np.einsum("ia,abk,jb->ijk", turn, gradients, turn), background)

    @override
    def build(self, ctx: BuildContext, b: ProviderBuilder, slots: Sequence[Slot]) -> None:
        held = self._held()
        clearance = ctx.settings.get("source_clearance")
        by_distance = clearance is not None
        at_body_points = [slot for slot in slots if slot.offset is not None]
        b.add_node(CalibratedMultipoleNode(
            ctx.node_name(), ctx.timestep, slots=[slot.name for slot in slots], sources=held.sources,
            source_positions=held.positions.ravel(), source_directions=held.directions.ravel(),
            coefficients=held.coefficients.ravel(), at=ctx.vector("at"), orientation=_unit(self.orientation),
            background=ctx.vector("background"), region=held.region.ravel(),
            gradient_limit=_gradient_limit(cast(Asset, self.calibration).name, cast(Asset, self.calibration).sha256),
            guard="distance" if by_distance else "gradient",
            source_clearance=float(cast(float, clearance)) if by_distance else 0.0,
            offset_slots=tuple(slot.name for slot in at_body_points),
            offsets=tuple(c for slot in at_body_points for c in slot.offset or ())))


__all__ = ["CALIBRATIONS", "FLOAT32_BUDGET", "Actuation", "Allocation", "CalibratedMultipole", "CoilArray", "allocate",
           "calibration_of"]
