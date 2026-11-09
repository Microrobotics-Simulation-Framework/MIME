"""The field of an electromagnet array as a calibrated multipole model (MPEM, Petruska et al. 2017), order one: each
coil's field per unit current is the sum of a few fitted point sources, ``b = sum_j beta_j (3 (d_j . r) r / r^5 - d_j /
r^3)``, ``r`` the point less the source's position; the field is linear in the delivered currents, and the gradient is
the closed form's own derivative. The calibration file's reader, the field-cancellation metric that predicts what
precision loses, and the node."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, cast, override

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from jax import Array, vmap
from jaxtyping import Float
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, as_run_array, jnp, node_input, quaternion, switch
from mime.nodes.actuation._chain import POSE_UNITS, flattened

#: Where a padded source sits, in the calibration's frame (m): far enough that no reader comes near it, and its
#: coefficient is zero, so it adds nothing to any field.
PAD_POSITION = (0.0, 0.0, 1.0e3)
#: The points per axis of the grid over the declared region the conditioning is judged on: the 0.2 line's, at which
#: ``eps / c`` was measured to predict the float32 error of the three shipped calibrations within about 1.4 times.
CONDITIONING_GRID = 3
#: The points per axis of the grid over the declared region that a coil's gradient scale is the median over.
SCALE_GRID = 9
#: How many times its scale (its median over the declared region) a coil's gradient per unit current may reach before
#: ``near_source`` is raised: a declared convention, set from a measurement. Over 60 000 drawn points of its own region
#: the smooth gradient of Navion_2 (the widest of the three shipped calibrations) reaches 13.5 times its median at the
#: most, CardioMag's 2.4 and OctoMag's 5.5 at the 99.9th percentile, so 30 flags no smooth part of any of them; the
#: gradient at OctoMag's fitted source, 20 um outside its region's face, is 10^8 times it.
GRADIENT_DEPARTURE = 30.0

_MODULE = "mime.nodes.actuation._multipole"
_NODE = f"{_MODULE}.CalibratedMultipoleNode"


class CalibrationError(ValueError):
    """A calibration file this reader does not take: why, in words."""


@dataclass(frozen=True)
class Calibration:
    """An order-one calibrated multipole model as its file holds it, in the calibration's own frame, SI: per coil ``k``
    and source ``j`` the source's fitted position (m), its direction (normalised on reading: the files do not hold
    unit vectors) and its coefficient ``beta`` (T m^3 / A), the coils' sources padded to one count; ``sources`` the
    real count per coil; ``region`` the declared ``Workspace_Dimensions``, ``[[x0, x1], [y0, y1], [z0, z1]]`` (m);
    ``offset_sources`` the count of sources in an ``Offset`` block the file carries and does not enable (not read).

    A source is a fitted basis function, not hardware: its position need not be a coil's, and a fit may place one
    far away with a large coefficient (OctoMag's sit up to 245 km out). The region is declared, never fitted: the
    upstream fitter never writes it, so being inside it is no evidence of accuracy and being outside it none of
    extrapolation."""

    system: str
    positions: NDArray[np.float64]
    directions: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    sources: tuple[int, ...]
    region: NDArray[np.float64]
    offset_sources: int = 0

    @property
    def coils(self) -> int:
        return len(self.sources)

    @cached_property
    def real(self) -> NDArray[np.bool_]:
        """``(K, S)``: which source slots are the file's, not padding."""
        width = int(self.positions.shape[1])
        return np.array([[j < count for j in range(width)] for count in self.sources], dtype=np.bool_)

    def nearest_approach(self) -> float:
        """The least distance (m) from a fitted source to the declared region."""
        low, high = self.region[:, 0], self.region[:, 1]
        held = self.positions[self.real]
        outside = np.maximum(0.0, np.maximum(low - held, held - high))
        return float(np.min(np.linalg.norm(outside, axis=-1)))

    def gradient_scale(self) -> NDArray[np.float64]:
        """Each coil's gradient scale, ``(K,)`` (T/(m A)): the median over the declared region (a ``SCALE_GRID`` grid)
        of the Frobenius norm of the coil's gradient per unit current. What the gradient is inside the region, as the
        fit left it; a reader's gradient many times it is a source's own term (``GRADIENT_DEPARTURE``)."""
        norms = np.array([np.linalg.norm(self.unit_fields(point)[1], axis=(0, 1)) for point in self.grid(SCALE_GRID)])
        scale: NDArray[np.float64] = np.median(norms, axis=0)
        return scale

    def gradient_limit(self) -> NDArray[np.float64]:
        """Each coil's gradient per unit current above which ``near_source`` is raised, ``(K,)`` (T/(m A))."""
        return GRADIENT_DEPARTURE * self.gradient_scale()

    def grid(self, count: int = CONDITIONING_GRID) -> NDArray[np.float64]:
        """``count`` points a side over the declared region, ``(count**3, 3)``."""
        axes: list[NDArray[np.float64]] = [np.linspace(float(self.region[i, 0]), float(self.region[i, 1]), count)
                                           for i in range(3)]
        found: NDArray[np.float64] = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        return found

    def cancellation(self, points: ArrayLike | None = None) -> float:
        """The worst field cancellation over ``points`` (the region's grid by default), ``|sum_j b_j| / sum_j |b_j|``
        per coil and point: how small each coil's field is against the sources it is summed from. Rounding in the sum
        costs ``eps / c`` of the field, which is what two implementations can agree to and what a lower precision
        loses. It is weighted by the field, ``1 / r^3``, not by the coefficients: a parametric ratio of the moments
        counts a source 245 km away as one 3 cm away and orders the shipped systems wrongly."""
        held = self.grid() if points is None else np.asarray(points, dtype=np.float64)
        worst = 1.0
        for k, count in enumerate(self.sources):
            offsets = held[:, None, :] - self.positions[k, :count][None]
            distance = np.linalg.norm(offsets, axis=-1, keepdims=True)
            unit = offsets / distance
            along = np.sum(unit * self.directions[k, :count][None], axis=-1, keepdims=True)
            each = self.coefficients[k, :count][None, :, None] / distance**3 * (
                3.0 * along * unit - self.directions[k, :count][None])
            ratio = np.linalg.norm(each.sum(axis=1), axis=-1) / np.linalg.norm(each, axis=-1).sum(axis=1)
            worst = min(worst, float(np.min(ratio)))
        return worst

    def unit_fields(self, point: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Each coil's field per unit current at ``point`` in the calibration's frame, ``(3, K)`` (T/A), and its
        gradient, ``(3, 3, K)`` (T/(m A)), in float64: the node's closed form, for the host's helpers."""
        offsets = np.asarray(point, dtype=np.float64)[None, None, :] - self.positions
        distance = np.linalg.norm(offsets, axis=-1)[..., None, None]
        vectors = (self.coefficients[..., None] * self.directions)
        along = np.sum(vectors * offsets, axis=-1)[..., None]
        fields = 3.0 * along * offsets / distance[..., 0] ** 5 - vectors / distance[..., 0] ** 3
        outer = offsets[..., :, None] * offsets[..., None, :]
        gradients = (3.0 / distance**5) * (along[..., None] * np.eye(3) + vectors[..., :, None] * offsets[..., None, :]
                                           + offsets[..., :, None] * vectors[..., None, :]
                                           - 5.0 * along[..., None] * outer / distance**2)
        return fields.sum(axis=1).T, np.moveaxis(gradients.sum(axis=1), 0, -1)

    def floor(self, eps: float) -> float:
        """The relative error rounding at ``eps`` puts on the field: ``eps / c`` over the region's grid."""
        return eps / self.cancellation()


def _vector(value: object, where: str) -> list[float]:
    if not isinstance(value, list) or len(cast(list[object], value)) != 3:
        raise CalibrationError(f"{where} is not three numbers")
    return [float(cast(float, c)) for c in cast(list[object], value)]


def read_calibration(data: bytes) -> Calibration:
    """The calibration in ``data``, a file in the upstream MPEM schema (``Coil_List``, each coil's ``Source_List``,
    each source's ``A_Coeff``, ``B_Coeff``, ``Source_Direction`` and ``Source_Position``, and ``Workspace_Dimensions``).
    Refused (:class:`CalibrationError`): interior terms (``A_Coeff``), which no shipped calibration holds; an order
    above one, which none holds either and whose fit this release does not validate; a zero direction; and a
    ``Coil_List`` that names the ``Offset`` block, current-independent sources that the upstream model (``mag_manip``)
    then evaluates and this release does not. An ``Offset`` block the list does not name is a disabled one, which
    upstream does not evaluate either (it loads a block as the offset only by that name, in ``Coil_List``): it is not
    read, and is counted in ``offset_sources`` so that what the file carries is said (CardioMag's carries eight)."""
    try:
        raw = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise CalibrationError(f"the file is not YAML: {error}") from error
    if not isinstance(raw, dict) or "Coil_List" not in raw or "Workspace_Dimensions" not in raw:
        raise CalibrationError("the file holds no Coil_List and Workspace_Dimensions: not an MPEM calibration")
    held = cast(dict[str, Any], raw)
    if any("Offset" in name for name in cast(list[str], held["Coil_List"])):
        raise CalibrationError("Coil_List names an Offset block, current-independent sources that mag_manip evaluates "
                               "and this release does not: remove it from the list to run without them")
    offset = cast(object, held.get("Offset"))
    offset_sources = (len(cast(list[str], cast(dict[str, Any], offset).get("Source_List", [])))
                      if isinstance(offset, dict) else 0)
    per_coil: list[list[tuple[list[float], list[float], float]]] = []
    for coil in cast(list[str], held["Coil_List"]):
        sources: list[tuple[list[float], list[float], float]] = []
        for name in cast(list[str], held[coil]["Source_List"]):
            source = cast(dict[str, Any], held[coil][name])
            where = f"{coil}/{name}"
            if source.get("A_Coeff") is not None:
                raise CalibrationError(f"{where} holds interior terms (A_Coeff), which this reader does not evaluate")
            coefficients = cast(list[float] | None, source.get("B_Coeff"))
            if not coefficients:
                raise CalibrationError(f"{where} holds no B_Coeff")
            if len(coefficients) != 1:
                raise CalibrationError(f"{where} is of order {len(coefficients)}: this release evaluates order one, "
                                       f"which every shipped calibration is, and validates no higher order's fit")
            direction = _vector(source["Source_Direction"], f"{where}'s Source_Direction")
            length = math.sqrt(sum(c * c for c in direction))
            if length == 0.0:
                raise CalibrationError(f"{where}'s Source_Direction is zero")
            sources.append((_vector(source["Source_Position"], f"{where}'s Source_Position"),
                            [c / length for c in direction], float(coefficients[0])))
        if not sources:
            raise CalibrationError(f"{coil} has no sources")
        per_coil.append(sources)
    width = max(len(sources) for sources in per_coil)
    positions = np.tile(np.asarray(PAD_POSITION), (len(per_coil), width, 1))
    directions = np.tile(np.asarray((0.0, 0.0, 1.0)), (len(per_coil), width, 1))
    coefficients = np.zeros((len(per_coil), width))
    for k, sources in enumerate(per_coil):
        for j, (position, direction, coefficient) in enumerate(sources):
            positions[k, j], directions[k, j], coefficients[k, j] = position, direction, coefficient
    region = np.asarray(held["Workspace_Dimensions"], dtype=np.float64)
    if region.shape != (3, 2) or not np.all(region[:, 0] < region[:, 1]):
        raise CalibrationError(f"Workspace_Dimensions is {region.tolist()}, not three ordered bounds")
    return Calibration(str(held.get("System_Name", "")), positions, directions, coefficients,
                       tuple(len(sources) for sources in per_coil), region, offset_sources)


def source_field(vector: Array, position: Array, point: Array) -> tuple[Array, Array]:
    """One order-one source's field per unit current at ``point`` and its gradient ``dB_i/dx_j``, ``vector`` its
    coefficient times its direction (T m^3 / A), in the calibration's frame. The field is singular at the source; a
    reader there is no valid state (the guard on the distance flags its approach)."""
    offset = point - position
    distance = switch.nonzero_norm(offset)
    along = jnp.dot(vector, offset)
    field = 3.0 * along * offset / distance**5 - vector / distance**3
    gradient = (3.0 / distance**5) * (along * jnp.eye(3, dtype=offset.dtype) + jnp.outer(vector, offset)
                                      + jnp.outer(offset, vector)
                                      - 5.0 * along * jnp.outer(offset, offset) / distance**2)
    return field, gradient


def unit_fields(point: Array, positions: Array, vectors: Array) -> tuple[Array, Array]:
    """Each coil's field per unit current at ``point`` in the calibration's frame, ``(3, K)`` (T/A), and its gradient,
    ``(3, 3, K)`` (T/(m A)): its sources' summed, ``positions`` and ``vectors`` ``(K, S, 3)``."""
    fields, gradients = vmap(vmap(source_field, in_axes=(0, 0, None)), in_axes=(0, 0, None))(vectors, positions, point)
    return jnp.sum(fields, axis=1).T, jnp.moveaxis(jnp.sum(gradients, axis=1), 0, -1)


def placed_unit_fields(point: Array, at: Array, orientation: Array, positions: Array,
                       vectors: Array) -> tuple[Array, Array]:
    """:func:`unit_fields` of the array placed in the world at ``at``, turned by the unit quaternion ``orientation``,
    read at the world ``point``: the exact transform, ``b = R b_c(R^T (x - t))`` and ``G = R G_c R^T``."""
    turn = quaternion.rotation_matrix(orientation)
    fields, gradients = unit_fields(turn.T @ (point - at), positions, vectors)
    return turn @ fields, jnp.einsum("ia,abk,jb->ijk", turn, gradients, turn)


EQUATIONS = (r"\mathbf{B} = R \sum_{k} i_{k} \sum_{j} \beta_{kj} \left( \frac{3 (\mathbf{d}_{kj} \cdot "
             r"\mathbf{r}_{kj}) \, \mathbf{r}_{kj}}{r_{kj}^{5}} - \frac{\mathbf{d}_{kj}}{r_{kj}^{3}} \right) + "
             r"\mathbf{B}_{0}, \quad G = R \, G_{c} \, R^{\top}, \quad \mathbf{r}_{kj} = R^{\top} "
             r"(\mathbf{x} + R(\mathbf{u}) \, \mathbf{o} - \mathbf{t}) - \mathbf{p}_{kj}, \quad \delta = \min_{kj} "
             r"r_{kj}")


class CalibratedMultipoleState(TypedDict, extra_items=ReadOnly[Float[Array, "..."]]):
    """The two flags, ``outside_region`` and ``near_source`` (1 at a step where any reader was outside the declared
    region, or where a coil's gradient per unit current at a reader was more than ``GRADIENT_DEPARTURE`` times its
    scale in the region, or, with a declared ``source_clearance``, a reader was nearer a fitted source than that; 0
    otherwise); and per reader slot ``flux_density_<slot>``,
    ``gradient_<slot>`` and ``source_distance_<slot>``, the reader's distance to the nearest fitted source (m; zero
    before the first step)."""

    outside_region: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    near_source: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape


class CalibratedMultipoleNode(MimeNode):
    """The field and its gradient, at each reader slot's point, of an electromagnet array whose coils are a
    calibrated multipole model of order one, linear in ``delivered_currents``, the current each coil carries, plus
    ``background``, a uniform field in the world frame: each
    coil's ``sources[k]`` fitted sources (``source_positions``, ``source_directions``, unit, and ``coefficients``,
    each a ``params`` leaf, flattened from ``(K, S, 3)`` and ``(K, S)``, padding included), in the calibration's frame,
    which ``at`` and ``orientation`` place in the world. A slot reads at the position of the pose it takes as
    ``points_<slot>``, or, in ``offset_slots``, at the point fixed in that pose at its row of ``offsets``.

    Each step it raises ``outside_region`` where a reader lies outside the declared ``region`` (``x0, x1, y0, y1, z0,
    z1`` in the calibration's frame), and ``near_source`` where a reader is near a fitted source: with ``guard``
    ``"gradient"``, where some coil's gradient per unit current there exceeds that coil's ``gradient_limit``; with
    ``"distance"``, where the reader is nearer a fitted source than ``source_clearance``. Both are reports no edge
    reads. A node built directly, not through the provider, skips the provider's float32 refusal (OctoMag's arrays in
    float32 are not guarded)."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-128",
        description="An electromagnet array's field and gradient at each reader's point, linear in its coils' "
                    "delivered currents, each coil a calibrated multipole model of order one (a sum of fitted point "
                    "sources), and a uniform background field.",
        governing_equations=EQUATIONS,
        assumptions=("The field is linear in the delivered currents: the cores are below saturation",
                     "Each source is of order one, a point source of the potential's dipole term, fitted, not hardware",
                     "Quasi-static: the delivered current is the field's, with no eddy currents and no inductance",
                     "The array is rigid and placed by one exact transform, its calibration's frame in the world"),
        limitations=("The field is singular at a fitted source; the node counts a reader nearer one than its guard",
                     "The declared region is not a fit domain: the node counts a reader outside it, which is no "
                     "evidence of extrapolation, and inside it no evidence of accuracy"),
        references=(Reference("Petruska2017", "the multipole electromagnet model"),
                    Reference("Charreyron2021", "modelling electromagnetic navigation systems with it")),
    )
    symbols = (
        Symbol(r"\mathbf{B}", SymbolKind.STATE, "flux_density_{slot}", "T", f"{_MODULE}.placed_unit_fields"),
        Symbol("G", SymbolKind.STATE, "gradient_{slot}", "T/m", f"{_MODULE}.placed_unit_fields"),
        Symbol("G_{c}", SymbolKind.STATE, "gradient_{slot}", "T/m", f"{_MODULE}.source_field"),
        Symbol("i_{k}", SymbolKind.INPUT, "delivered_currents", "A", f"{_NODE}.advance"),
        Symbol(r"\beta_{kj}", SymbolKind.PARAMETER, "coefficients", "T m^3/A", f"{_MODULE}.source_field"),
        Symbol(r"\mathbf{d}_{kj}", SymbolKind.PARAMETER, "source_directions", "1", f"{_MODULE}.source_field"),
        Symbol(r"\mathbf{p}_{kj}", SymbolKind.PARAMETER, "source_positions", "m", f"{_MODULE}.source_field"),
        Symbol(r"\mathbf{r}_{kj}", SymbolKind.PARAMETER, "source_positions", "m", f"{_MODULE}.source_field"),
        Symbol("r_{kj}", SymbolKind.PARAMETER, "source_positions", "m", f"{_MODULE}.source_field"),
        Symbol("R", SymbolKind.PARAMETER, "orientation", "1", f"{_MODULE}.placed_unit_fields"),
        Symbol(r"\mathbf{t}", SymbolKind.PARAMETER, "at", "m", f"{_MODULE}.placed_unit_fields"),
        Symbol(r"\mathbf{B}_{0}", SymbolKind.PARAMETER, "background", "T", f"{_NODE}.advance"),
        Symbol(r"\delta", SymbolKind.STATE, "source_distance_{slot}", "m", f"{_NODE}.advance"),
        Symbol(r"\mathbf{x}", SymbolKind.INPUT, "points_{slot}", POSE_UNITS, f"{_NODE}.reader_point"),
        Symbol(r"\mathbf{u}", SymbolKind.INPUT, "points_{slot}", POSE_UNITS, f"{_NODE}.reader_point"),
        Symbol(r"\mathbf{o}", SymbolKind.PARAMETER, "offsets", "m", f"{_NODE}.reader_point"),
    )

    def __init__(self, name: str, timestep: float, *, slots: Sequence[str], sources: Sequence[int],
                 source_positions: ArrayLike, source_directions: ArrayLike, coefficients: ArrayLike,
                 at: Sequence[float], orientation: Sequence[float], background: Sequence[float],
                 region: ArrayLike, gradient_limit: ArrayLike, guard: str = "gradient", source_clearance: float = 0.0,
                 offset_slots: Sequence[str] = (), offsets: Sequence[float] = ()) -> None:
        counts = tuple(int(c) for c in sources)
        width = max(counts)
        if guard not in ("gradient", "distance"):
            raise ValueError(f"guard {guard!r} is not 'gradient' or 'distance'")
        limits = flattened(gradient_limit)
        if len(limits) != len(counts):
            raise ValueError(f"{len(counts)} coils take {len(counts)} gradient limits, not {len(limits)}")
        held = [flattened(values) for values in (source_positions, source_directions, coefficients)]
        if len(held[2]) != len(counts) * width or len(held[0]) != 3 * len(held[2]) or len(held[1]) != len(held[0]):
            raise ValueError(f"{len(counts)} coils of up to {width} sources take {len(counts) * width} coefficients "
                             f"and three times as many position and direction numbers")
        at_body_points = {"offset_slots": tuple(offset_slots), "offsets": flattened(offsets)} if offset_slots else {}
        super().__init__(name, timestep, slots=tuple(slots), sources=counts,
                         source_positions=held[0], source_directions=held[1], coefficients=held[2], at=flattened(at),
                         orientation=flattened(orientation), background=flattened(background), region=flattened(region),
                         gradient_limit=limits, guard=guard, source_clearance=float(source_clearance),
                         **at_body_points)

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(),
                "source_positions": ParamSpec(units="m", description="each fitted source's position in the "
                                                                     "calibration's frame, padding included"),
                "source_directions": ParamSpec(units="1", description="each source's unit direction"),
                "coefficients": ParamSpec(units="T m^3/A", description="each source's coefficient, zero for padding"),
                "at": ParamSpec(units="m", description="where the calibration's origin sits in the world"),
                "orientation": ParamSpec(units="1", description="the calibration frame's turn in the world, a unit "
                                                                "quaternion (w, x, y, z), checked at build and "
                                                                "not normalised in the step: moved off the unit "
                                                                "sphere it gives a wrong field, unflagged"),
                "background": ParamSpec(units="T", description="a uniform field in the world frame, added to the "
                                                               "coils'"),
                "region": ParamSpec(trainable=False, units="m", description="the declared region the reader is "
                                                                            "counted outside of"),
                "gradient_limit": ParamSpec(trainable=False, units="T/(m A)",
                                            description="each coil's gradient per unit current above which "
                                                        "near_source is raised, in the 'gradient' guard"),
                "source_clearance": ParamSpec(trainable=False, units="m", description="the reader's least distance "
                                                                                      "from a fitted source, in the "
                                                                                      "'distance' guard"),
                "offsets": ParamSpec(units="m", description="each offset slot's point, in the frame of the pose it "
                                                            "reads; a node with no offset slot has none")}

    def _slots(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], self.constants()["slots"])

    def _offset_slots(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], self.constants().get("offset_slots", ()))

    def _sources(self) -> tuple[int, ...]:
        return cast(tuple[int, ...], self.constants()["sources"])

    @property
    def coils(self) -> int:
        return len(self._sources())

    def arrays(self, p: Mapping[str, Any], dtype: Any) -> tuple[Array, Array]:
        """The sources' positions and their coefficients times their directions, ``(K, S, 3)`` each, from the leaves."""
        counts = self._sources()
        shape = (len(counts), max(counts), 3)
        positions = jnp.reshape(jnp.asarray(p["source_positions"], dtype=dtype), shape)
        directions = jnp.reshape(jnp.asarray(p["source_directions"], dtype=dtype), shape)
        coefficients = jnp.reshape(jnp.asarray(p["coefficients"], dtype=dtype), shape[:2])
        return positions, coefficients[..., None] * directions

    def reader_point(self, p: Mapping[str, Any], slot: str, pose: Array) -> Array:
        """Where ``slot`` reads the field from the pose it takes: its position, or for an offset slot the point fixed
        in it, ``x + R(u) o``."""
        held = self._offset_slots()
        if slot not in held:
            return pose[:3]
        offset = jnp.reshape(jnp.asarray(p["offsets"], dtype=pose.dtype), (-1, 3))[held.index(slot)]
        return pose[:3] + quaternion.rotate(pose[3:], offset)

    @override
    def initial_state(self) -> NodeFields:
        zero = as_run_array(0.0)
        out: NodeFields = {"outside_region": zero, "near_source": zero}
        for slot in self._slots():
            out[f"flux_density_{slot}"] = as_run_array(jnp.zeros(3))
            out[f"gradient_{slot}"] = as_run_array(jnp.zeros((3, 3)))
            out[f"source_distance_{slot}"] = zero
        return out

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"delivered_currents": node_input("A", (self.coils,)),
                **{f"points_{slot}": node_input(POSE_UNITS, (7,)) for slot in self._slots()}}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        zero = as_run_array(0.0)
        dtype = zero.dtype
        positions, vectors = self.arrays(p, dtype)
        at = jnp.asarray(p["at"], dtype=dtype)
        orientation = jnp.asarray(p["orientation"], dtype=dtype)
        background = jnp.asarray(p["background"], dtype=dtype)
        currents = jnp.asarray(boundary_inputs["delivered_currents"], dtype=dtype)
        region = jnp.reshape(jnp.asarray(p["region"], dtype=dtype), (3, 2))
        counts = self._sources()
        real = np.arange(max(counts))[None, :] < np.asarray(counts)[:, None]
        by_gradient = self.constants()["guard"] == "gradient"
        limits = jnp.asarray(p["gradient_limit"], dtype=dtype)
        out: NodeFields = {}
        outside, near = jnp.asarray(False), jnp.asarray(False)
        for slot in self._slots():
            point = self.reader_point(p, slot, jnp.asarray(boundary_inputs[f"points_{slot}"], dtype=dtype))
            fields, gradients = placed_unit_fields(point, at, orientation, positions, vectors)
            out[f"flux_density_{slot}"] = fields @ currents + background
            out[f"gradient_{slot}"] = gradients @ currents
            local = quaternion.rotate_back(orientation, point - at)
            outside = outside | jnp.any(local < region[:, 0]) | jnp.any(local > region[:, 1])
            distances = switch.nonzero_norm(local - positions, axis=-1)
            nearest = jnp.min(switch.static_select(real, distances, jnp.inf))
            if by_gradient:
                near = near | jnp.any(jnp.sum(gradients**2, axis=(0, 1)) > limits**2)
            else:
                near = near | (nearest < jnp.asarray(p["source_clearance"], dtype=dtype))
            out[f"source_distance_{slot}"] = nearest
        out["outside_region"], out["near_source"] = outside.astype(dtype), near.astype(dtype)
        return out


__all__ = ["CONDITIONING_GRID", "EQUATIONS", "Calibration", "CalibrationError", "CalibratedMultipoleNode",
           "CalibratedMultipoleState", "placed_unit_fields", "read_calibration", "source_field", "unit_fields"]
