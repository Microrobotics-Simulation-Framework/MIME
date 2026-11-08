"""The prescribed flow: Poiseuille's and Womersley's analytic solutions in a straight circular tube driven by a body
force's waveform, the FVM's own oracles as a flow field kind (F0's item 9, 7 (c)), a CPU path with no grid.

**The solution.** A uniform body force ``g(t) = n (g0 + A cos(omega t + phi))`` along a straight rigid tube of radius
``R`` (its component along the tube's axis ``n``; a uniform force across an infinite tube is balanced by its pressure)
drives the fully developed axial flow ``u = g0 (R^2 - r^2) / (4 nu) + Re[(A / (i omega)) (1 - J0(b r) / J0(b R))
e^{i (omega t + phi)}]``, ``b = e^{-i pi / 4} sqrt(omega / nu)``, ``alpha = R sqrt(omega / nu)``: an exact solution of the
Navier-Stokes equations at any Reynolds number (convection vanishes), the periodic state of a steady and harmonic drive
(no start-up).

**The series** (round 4, item 9). JAX refuses Bessel functions of a complex argument, so ``J0`` of ``x e^{-i pi/4}`` is
summed in the program from its power series in ``q = x^2 / 4`` in nested form (only the term ratios ``i q / k^2`` appear,
so nothing under- or overflows in float32), with its first term removed (``S = J0 - 1``) and the profile written as
``(S(q_R) - S(q_r)) / (alpha^2 J0(b R))``, which divides by nothing that vanishes as ``alpha -> 0`` and loses nothing to
the cancellation of ``1 - J0(b r) / J0(b R)`` at small alpha. The term count is fixed by the dtype's validity line, so it
depends on no value. **Validity** (measured against SciPy, the profile's error over its peak): float64 under 2e-13 to
alpha 30 and 1e-10 to alpha 50; float32 7.5e-6 at alpha 20 and 3.2e-5 at 25. Past the dtype's line the node flags the
run, which is refused after it.

Viscosity, density, the tube's centre, axis and radius are leaves; the waveform is the law's, delivered each step. The
time is the node's own step count times dt."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, override

import numpy as np
from jax import Array
from jaxtyping import Float, Int
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.params import ParamSpec
from typing_extensions import Annotated, NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, Units, as_run_array, jnp, node_fields, node_input, \
    run_dtype
from mime.nodes.flow._waveform import WAVEFORM_SIZE
from mime.registry import Problem

EQUATIONS = (r"\operatorname{u}(r, t) = \frac{\rho g_0 (R^2 - r^2)}{4 \mu} + \mathrm{Re}\left[\frac{A}{2 \pi i f} "
             r"\left(1 - \frac{\operatorname{J}_{0}(\operatorname{b} r)}{\operatorname{J}_{0}(\operatorname{b} R)}"
             r"\right) e^{i (2 \pi f t + \varphi)}\right], \quad \operatorname{b} = e^{-i \pi / 4} "
             r"\sqrt{2 \pi f \rho / \mu}")

#: The largest Womersley number a dtype's series is good for, and its error there (the profile's over its peak).
VALIDITY: Mapping[str, tuple[float, float]] = {"float64": (50.0, 1e-10), "float32": (25.0, 4e-5)}

_STEP = "mime.nodes.flow._womersley.PrescribedFlowNode.advance"
_READ = "mime.nodes.flow._womersley.axial_velocity"


def series_terms(dtype: Any) -> int:
    """The series' fixed term count: enough for float64's rounding to the dtype's line (measured: the line plus 20)."""
    line, _ = VALIDITY[np.dtype(dtype).name]
    return int(math.ceil(line)) + 20


def _square_ratio(k: int) -> int:
    """``J0``'s term ratio past its first term, ``(k + 1)^2``."""
    return (k + 1) ** 2


def _flux_ratio(k: int) -> int:
    """The mean profile's term ratio, ``k (k + 2)``."""
    return k * (k + 2)


def _nested(x_im: Array, terms: int, ratio: Callable[[int], int]) -> tuple[Array, Array]:
    """``1 + (i x / d_1)(1 + (i x / d_2)(1 + ...))`` to ``terms`` levels, ``d_k = ratio(k)``, as (re, im)."""
    re = jnp.ones_like(x_im)
    im = jnp.zeros_like(x_im)
    for k in range(terms - 1, 0, -1):
        d = float(ratio(k))
        re, im = 1.0 - im * x_im / d, re * x_im / d
    return re, im


def _times_i(a: tuple[Array, Array]) -> tuple[Array, Array]:
    return -a[1], a[0]


def _divide(a: tuple[Array, Array], b: tuple[Array, Array]) -> tuple[Array, Array]:
    den = b[0] * b[0] + b[1] * b[1]
    return (a[0] * b[0] + a[1] * b[1]) / den, (a[1] * b[0] - a[0] * b[1]) / den


def profile(rho_squared: Array, alpha_squared: Any, terms: int) -> tuple[Array, Array]:
    """``D(rho) = (1 - J0(b r) / J0(b R)) / alpha^2`` as (re, im), at ``rho^2 = (r / R)^2``: the harmonic's profile
    over ``alpha^2``, so the axial velocity is ``(A R^2 / nu) Re[-i D e^{i theta}]``; ``i (1 - rho^2) / 4`` at ``alpha =
    0``. It takes ``alpha^2 = R^2 omega / nu`` and ``rho^2``, never a square root, so its derivatives are finite at a
    steady drive and on the axis."""
    q_wall = alpha_squared / 4.0 + 0.0 * rho_squared
    q = q_wall * rho_squared
    n_wall = _nested(q_wall, terms, _square_ratio)
    n_r = _nested(q, terms, _square_ratio)
    # S(q) = J0 - 1 = i q N(q); so (S(q_R) - S(q_r)) / alpha^2 = (i / 4) (N(q_R) - rho^2 N(q_r)).
    difference = _times_i((0.25 * (n_wall[0] - rho_squared * n_r[0]), 0.25 * (n_wall[1] - rho_squared * n_r[1])))
    s_wall = _times_i((q_wall * n_wall[0], q_wall * n_wall[1]))
    return _divide(difference, (1.0 + s_wall[0], s_wall[1]))


def mean_profile(alpha_squared: Any, terms: int) -> tuple[Array, Array]:
    """The profile's mean over the cross-section, ``(1 - 2 J1(b R) / (b R J0(b R))) / alpha^2`` as (re, im): ``i / 8``
    at ``alpha = 0``. ``J0 - 2 J1 / z = sum_{k >= 1} (i q)^k k / ((k + 1) (k!)^2)``, nested with the ratios ``1 / (k (k
    + 2))``."""
    q_wall = jnp.asarray(alpha_squared) / 4.0
    n_flux = _nested(q_wall, terms, _flux_ratio)
    n_wall = _nested(q_wall, terms, _square_ratio)
    top = _times_i((n_flux[0] / 8.0, n_flux[1] / 8.0))
    s_wall = _times_i((q_wall * n_wall[0], q_wall * n_wall[1]))
    return _divide(top, (1.0 + s_wall[0], s_wall[1]))


def _harmonic(shape: tuple[Array, Array], amplitude: Any, theta: Any, scale: Any) -> Array:
    """``scale A Re[-i D e^{i theta}] = scale A (D_im cos theta + D_re sin theta)``."""
    return scale * amplitude * (shape[1] * jnp.cos(theta) + shape[0] * jnp.sin(theta))


def axial_velocity(rho_squared: Array, time: Any, waveform: Array, axis: Array, radius: Any, nu: Any,
                   terms: int) -> Array:
    """The axial velocity (m/s) at ``rho^2 = (r / R)^2`` (inside the lumen; zero at and beyond the wall) at ``time``."""
    w = jnp.asarray(waveform)
    driven = jnp.sum(w[:3] * axis)
    steady, amplitude, frequency, phase = driven * w[3], driven * w[4], w[5], w[6]
    omega = 2.0 * jnp.pi * frequency
    scale = radius * radius / nu
    inside = jnp.clip(1.0 - rho_squared, 0.0, None)
    shape = profile(jnp.clip(rho_squared, 0.0, 1.0), scale * omega, terms)
    harmonic = _harmonic(shape, amplitude, omega * time + phase, scale)
    return jnp.where(rho_squared < 1.0, 0.25 * scale * steady * inside + harmonic, 0.0)


def mean_velocity(time: Any, waveform: Array, axis: Array, radius: Any, nu: Any, terms: int) -> Array:
    """The cross-section's mean axial velocity (m/s); the flux is ``pi R^2`` times it."""
    w = jnp.asarray(waveform)
    driven = jnp.sum(w[:3] * axis)
    omega = 2.0 * jnp.pi * w[5]
    scale = radius * radius / nu
    return scale * driven * w[3] / 8.0 + _harmonic(mean_profile(scale * omega, terms), driven * w[4],
                                                   omega * time + w[6], scale)


class PrescribedFlowState(TypedDict, closed=True):
    steps: ReadOnly[Annotated[Int[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    #: The waveform it last read, which a reading at points after the step responds to, by coefficient.
    drive_along: ReadOnly[Annotated[Float[Array, "3"], Units("1")]]  # noqa: F722 - a jaxtyping shape
    drive_steady: ReadOnly[Annotated[Float[Array, ""], Units("m/s^2")]]  # noqa: F722 - a jaxtyping shape
    drive_amplitude: ReadOnly[Annotated[Float[Array, ""], Units("m/s^2")]]  # noqa: F722 - a jaxtyping shape
    drive_frequency: ReadOnly[Annotated[Float[Array, ""], Units("Hz")]]  # noqa: F722 - a jaxtyping shape
    drive_phase: ReadOnly[Annotated[Float[Array, ""], Units("rad")]]  # noqa: F722 - a jaxtyping shape
    centreline: ReadOnly[Annotated[Float[Array, ""], Units("m/s")]]  # noqa: F722 - a jaxtyping shape
    lumen_flux: ReadOnly[Annotated[Float[Array, ""], Units("m^3/s")]]  # noqa: F722 - a jaxtyping shape
    womersley_number: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    outside_validity: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape


def held_waveform(state: Mapping[str, Any]) -> Array:
    """The waveform's coefficients (``mime.nodes.flow._waveform``) a state holds by coefficient, as one vector."""
    return jnp.concatenate([jnp.asarray(state["drive_along"]).reshape(3)] + [
        jnp.asarray(state[f"drive_{name}"]).reshape(1) for name in ("steady", "amplitude", "frequency", "phase")])


class PrescribedFlowInputs(TypedDict, closed=True):
    waveform: NotRequired[ReadOnly[Annotated[Float[Array, "7"], Units("1")]]]  # noqa: F722 - a jaxtyping shape


class PrescribedFlowNode(MimeNode):
    """A medium's prescribed flow in a straight circular tube: Poiseuille's mean term and Womersley's harmonic,
    responding to the drive's waveform (the module states the solution)."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-125",
        description="A medium's flow prescribed analytically in a straight circular tube: Poiseuille's steady term and "
                    "Womersley's harmonic, the response to the body force's waveform.",
        governing_equations=EQUATIONS,
        assumptions=(
            "A straight rigid circular tube, infinite along its axis, its flow fully developed (axial, a function of "
            "the radius and time alone): an exact solution of the Navier-Stokes equations at any Reynolds number",
            "The periodic state of a steady and harmonic drive: no start-up transient",
            "Incompressible Newtonian fluid, its viscosity and density uniform; no-slip at the wall",
            "Only the drive's component along the tube's axis drives flow",
            "Nothing it carries is changed by what reads it: it takes no deliveries (outside)",
            "The time of a step is its count times the timestep, never a sum of timesteps"),
        limitations=(
            "Straight circular tubes only; one harmonic",
            "The Womersley number within its dtype's validity line (VALIDITY), the run refused past it",
        ),
        references=(Reference("Womersley1955"),),
    )
    symbols = (
        Symbol("g_0", SymbolKind.STATE, "drive_steady", "m/s^2", _STEP),
        Symbol("A", SymbolKind.STATE, "drive_amplitude", "m/s^2", _STEP),
        Symbol("f", SymbolKind.STATE, "drive_frequency", "Hz", _STEP),
        Symbol(r"\varphi", SymbolKind.STATE, "drive_phase", "rad", _STEP),
        Symbol("R", SymbolKind.PARAMETER, "tube_radius", "m", _READ),
        Symbol(r"\mu", SymbolKind.PARAMETER, "viscosity", "Pa s", _READ),
        Symbol(r"\rho", SymbolKind.PARAMETER, "density", "kg/m^3", _READ),
    )

    def __init__(self, name: str, timestep: float, *, viscosity: float, density: float,
                 tube_centre: Sequence[float], tube_axis: Sequence[float], tube_radius: float,
                 terms: int | None = None) -> None:
        super().__init__(name, timestep, viscosity=float(viscosity), density=float(density),
                         tube_centre=tuple(float(c) for c in tube_centre),
                         tube_axis=tuple(float(c) for c in tube_axis), tube_radius=float(tube_radius),
                         terms=int(terms) if terms is not None else series_terms(run_dtype()))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(),
                "viscosity": ParamSpec(units="Pa s", bounds=(0.0, None)),
                "density": ParamSpec(units="kg/m^3", bounds=(0.0, None)),
                "tube_centre": ParamSpec(trainable=False, units="m", description=(
                    "read only where the flow is read at points, after the step: no state the step writes depends on "
                    "it")),
                "tube_axis": ParamSpec(units="1"), "tube_radius": ParamSpec(units="m", bounds=(0.0, None))}

    @override
    def boundary_input_spec(self) -> dict[str, Any]:
        return {"waveform": node_input("1", (WAVEFORM_SIZE,),
                                       description="the drive's waveform: along (3), steady and amplitude (m/s^2), "
                                                   "frequency (Hz), phase (rad)")}

    @staticmethod
    def kinematic(p: Mapping[str, Any]) -> Array:
        return as_run_array(p["viscosity"]) / as_run_array(p["density"])

    @staticmethod
    def unit_axis(p: Mapping[str, Any]) -> Array:
        axis = as_run_array(p["tube_axis"])
        return axis / jnp.sqrt(jnp.sum(axis * axis))

    def velocity_at(self, p: Mapping[str, Any], state: Mapping[str, Array], points: Array) -> Array:
        """The velocity (m/s, shape ``(n, 3)``) at ``points`` (m, ``(n, 3)``), at the state's time: the reading the
        field kind provides (``field.flow.velocity``)."""
        terms = int(self.constants()["terms"])
        axis = self.unit_axis(p)
        offset = jnp.asarray(points) - as_run_array(p["tube_centre"])
        across = offset - jnp.sum(offset * axis, axis=-1, keepdims=True) * axis
        radius = as_run_array(p["tube_radius"])
        rho_squared = jnp.sum(across * across, axis=-1) / (radius * radius)
        time = jnp.asarray(state["steps"]).astype(axis.dtype) * self.delta_t
        u = axial_velocity(rho_squared, time, held_waveform(state), axis, as_run_array(p["tube_radius"]),
                           self.kinematic(p), terms)
        return u[:, None] * axis[None, :]

    def womersley_number(self, p: Mapping[str, Any], waveform: Array) -> Array:
        """``R sqrt(|omega| / nu)``, zero at a steady drive with a zero derivative there (the root is taken only of a
        positive number); a negative frequency is the same harmonic with its phase reversed."""
        radius = as_run_array(p["tube_radius"])
        squared = radius * radius * 2.0 * jnp.pi * jnp.abs(jnp.asarray(waveform)[5]) / self.kinematic(p)
        driven = squared > 0.0
        return jnp.where(driven, jnp.sqrt(jnp.where(driven, squared, 1.0)), 0.0)

    @override
    def initial_state(self) -> NodeFields:
        z = as_run_array(0.0)
        return node_fields(PrescribedFlowState(steps=jnp.asarray(0, dtype=jnp.int32), drive_along=as_run_array(
            np.zeros(3)), drive_steady=z, drive_amplitude=z, drive_frequency=z, drive_phase=z, centreline=z,
            lumen_flux=z, womersley_number=z, outside_validity=z))

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        dtype = run_dtype()
        w = jnp.asarray(boundary_inputs.get("waveform", jnp.zeros(WAVEFORM_SIZE, dtype=dtype)), dtype=dtype)
        steps = jnp.asarray(state["steps"]) + 1
        time = steps.astype(dtype) * dt
        terms = int(self.constants()["terms"])
        axis = self.unit_axis(p)
        radius, nu = as_run_array(p["tube_radius"]), self.kinematic(p)
        centre = axial_velocity(jnp.zeros((), dtype=dtype), time, w, axis, radius, nu, terms)
        flux = jnp.pi * radius * radius * mean_velocity(time, w, axis, radius, nu, terms)
        alpha = self.womersley_number(p, w)
        line, _ = VALIDITY[np.dtype(dtype).name]
        outside = jnp.maximum(jnp.asarray(state["outside_validity"]), (~(alpha <= line)).astype(dtype))
        return node_fields(PrescribedFlowState(steps=steps, drive_along=w[:3], drive_steady=w[3], drive_amplitude=w[4],
                                               drive_frequency=w[5], drive_phase=w[6], centreline=centre,
                                               lumen_flux=flux, womersley_number=alpha.astype(dtype),
                                               outside_validity=outside))

    def stop_problem(self, state: Mapping[str, Any]) -> Problem | None:
        """A run whose Womersley number left the dtype's validity line is refused after it."""
        if float(np.asarray(state.get("outside_validity", 0.0))) == 0.0:
            return None
        dtype = np.dtype(np.asarray(state["womersley_number"]).dtype).name
        line, error = VALIDITY[dtype]
        return Problem(
            "flow.prescribed-alpha",
            f"the prescribed flow's Womersley number reached {float(np.asarray(state['womersley_number'])):.4g}, past "
            f"the {dtype} series' line of {line:g} (its profile within {error:g} of its peak there)",
            "run in float64 (its line is 50), or lower the drive's frequency or the tube's radius; past 50, use FVM()")


__all__ = ["VALIDITY", "PrescribedFlowInputs", "held_waveform", "PrescribedFlowNode", "PrescribedFlowState", "axial_velocity", "mean_profile",
           "mean_velocity", "profile", "series_terms"]
