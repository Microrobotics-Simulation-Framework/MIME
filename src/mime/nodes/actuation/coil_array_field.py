"""CoilArrayFieldNode — electromagnetic navigation system (eMNS) field source.

This node models the magnetic field ``B`` and its spatial gradient ``∇B``
produced at an arbitrary target point by an array of *current-controlled
electromagnets* — an electromagnetic navigation system (eMNS) such as the
ETH/Nanoflex **Navion**, **OctoMag**, or **CardioMag**.

It complements the other two field sources in this package:

* ``ExternalMagneticFieldNode`` — spatially uniform rotating field
  (Helmholtz idealisation); ``∇B ≡ 0``, torque only.
* ``PermanentMagnetNode`` — a *positioned permanent magnet*, whose pose is the
  control input.
* ``CoilArrayFieldNode`` (this node) — a *fixed* array of ``K`` electromagnets
  whose **currents** are the control input.

The distinction matters: a permanent magnet's field is steered by moving it,
whereas an eMNS steers the field electronically. Gradient-based actuation
(``F = ∇(m·B)``, "gradient pulling") is an eMNS's native force mode, and the
current vector is what a controller allocates.

Physics
-------
Magnetostatics is linear in the coil currents, so the field superposes::

    B(p)  = Σ_k i_k · b_k(p)  = A_b(p) · i
    ∇B(p) = Σ_k i_k · ∇b_k(p)

where ``i ∈ R^K`` is the current vector [A] and ``b_k(p)`` is coil ``k``'s
field at **unit current** [T/A]. Stacking the ``b_k`` columnwise gives the
*actuation matrix* ``A_b(p) ∈ R^{3×K}`` — the object an eMNS controller
inverts to allocate currents. See :func:`field_actuation_matrix`.

Each coil's unit-current field is represented by the **Multipole Electromagnet
Model** (MPEM) of Petruska, Edelmann & Nelson (2017): a small set of fitted,
axially-symmetric multipole *sources*, each a magnetic scalar potential
expanded in ordinary Legendre polynomials about its own centre::

    ψ_j(p) = Σ_l  B_{j,l} · P_l(u_j) / r_j^{l+1}
    b_k(p) = −∇ Σ_{j ∈ coil k} ψ_j(p)

with ``r_j = p − pos_j``, ``u_j = d̂_j · r̂_j``. Because ``b = −∇ψ`` with ψ
harmonic, the field is **curl-free and divergence-free by construction** — the
gradient is automatically symmetric and trace-free, with no constraint to
impose numerically.

Differentiating the potential analytically gives the closed form actually
evaluated here::

    b(p) = Σ_l  B_l / r^{l+2} · [ (l+1)·P_l(u)·r̂ + P_l'(u)·(u·r̂ − d̂) ]

Note the ``l = 1`` term reduces to exactly the point-dipole field
``B_1/r³ · (3(d̂·r̂)r̂ − d̂)`` — and **every calibration shipped by ETH is order
1 only**, so in practice each coil is a sum of 3–8 fitted point dipoles. The
general-order path is retained so higher-order fits remain usable.

The spatial gradient ``∇B`` is obtained with ``jax.jacrev`` of the same
function that produces ``B`` — the single-code-path convention used by
``PermanentMagnetNode`` — so the gradient can never drift out of sync with the
field it is supposed to differentiate.

Sources are *fitted basis functions*, not hardware: ``Source_Position`` need
not coincide with a physical coil centre, and Navion coils contain
near-cancelling dipole pairs whose individual moments are ~100× the net.

Saturation
----------
The linear model assumes the iron cores are below saturation. Above it, the
S-MPEM correction of Charreyron et al. (2021) rescales each coil's current::

    h_k(i_k) = (√π / 2β_k) · erf(β_k · i_k)      B(p) = Σ_k h_k(i_k) · b_k(p)

``h_k'(0) = 1``, so the saturated model matches the linear one at low current.
Enable with ``saturation_beta``; omit for the linear model.

Reference (validity envelope): MPEM is an *exterior* expansion — it diverges as
``r → 0`` at a source. Sources sit inside the iron cores, well outside the
workspace, so this is not a concern for in-workspace queries; but the fit is
only validated inside ``Workspace_Dimensions``.

Algorithm ID: MIME-NODE-103
Stability: EXPERIMENTAL
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.node import BoundaryInputSpec
from maddening.core.compliance.metadata import (
    NodeMeta, StabilityLevel, ValidatedRegime, Reference,
)
from maddening.core.compliance.stability import stability

from mime.core.node import MimeNode
from mime.core.metadata import (
    MimeNodeMeta, NodeRole, ActuationMeta, ActuationPrinciple,
)


# The vendored ETH calibrations live *inside* the package (unlike the wall
# tables under the repo-root data/, which is gitignored) so they ship in the
# wheel and the node works from an installed distribution, not just a source
# checkout. parents[2] == src/mime.
_CALIBRATION_DIR = Path(__file__).resolve().parents[2] / "data" / "emns_calibrations"

# Bundled calibrations, by short name. See the README alongside the YAMLs for
# provenance and licence (Apache-2.0, ETH Zurich MSRL).
BUNDLED_CALIBRATIONS = {
    "navion_2":  "navion_2.yaml",    # 3-coil planar (Gervasoni 2024 / Navion)
    "octomag":   "octomag.yaml",     # 8-coil (Landers 2025 used this)
    "cardiomag": "cardiomag.yaml",   # 8-coil, no-offset variant
}

# Padding sentinel for ragged coils (see _pad_sources). Far enough that
# 1/r^(l+2) cannot overflow float32; the zero coefficient zeroes the term anyway.
_PAD_POSITION_M = 1.0e3

# Single-precision conditioning. A multipole least-squares fit may legitimately
# place sources far outside the hardware with huge coefficients whose fields
# nearly cancel inside the workspace: the answer is right, but computing it means
# subtracting large numbers, which costs significant digits.
#
# Measured float32 error against an x64 reference, over the workspace box, for
# the three shipped models (tests/verification/test_coil_array_field.py):
#
#   model      dist_ratio  field_cancel  predicted  meas. FIELD  meas. GRAD
#   navion_2      0.91       8.0e-3       1.5e-5      1.1e-5      1.1e-5
#   cardiomag     7.4        2.5e-1       4.8e-7      3.6e-7      5.0e-7
#   octomag       7.1e6      6.9e-5       1.7e-3      5.2e-4      1.2e-3
#
# `predicted` = eps_f32 / field_cancellation_ratio. It tracks the measured error
# to within ~1.4x across a 3600x spread in conditioning — good enough to key a
# decision on. The budget below is the largest relative error we accept in single
# precision. 1e-4 admits navion_2 and cardiomag (both comfortably better) and
# rejects octomag, whose GRADIENT error (1.2e-3) is ~0.1% and would land directly
# on a gradient-pull force.
#
# Note the gradient errors: taking grad B by autodiff of the closed form amplifies
# the field error by only 0.95x / 1.4x / 2.4x — a small constant, NOT the ~eps/dx
# blow-up a finite-difference gradient would suffer. There is no step size here to
# divide by; AD differentiates the expression, not samples of it.
_EPS_FLOAT32 = float(np.finfo(np.float32).eps)   # ~1.19e-7
_FLOAT32_REL_ERR_BUDGET = 1.0e-4


@dataclass(frozen=True)
class MpemCalibration:
    """Calibrated MPEM parameters for one eMNS.

    Zero-padded to a rectangular ``(K, S_max, L_max)`` layout so the whole array
    can be ``vmap``-ed. Padded sources carry ``b_coeff = 0`` (contributing
    nothing), a unit ``source_direction`` (a zero direction would produce
    ``0/0 = NaN`` on normalisation), and a distant ``source_position``.

    Attributes
    ----------
    system_name : str
        ``System_Name`` from the calibration file.
    source_position : (K, S_max, 3) float64
        Fitted multipole source centres [m]. **Not** physical coil centres.
    source_direction : (K, S_max, 3) float64
        Source symmetry axes, normalised on load (upstream does not guarantee
        unit length).
    b_coeff : (K, S_max, L_max) float64
        Exterior multipole coefficients [T·m³/A for l=1]; ``b_coeff[..., i]``
        has order ``l = i + 1``.
    workspace : (3, 2) float64
        Fit validity box ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`` [m].
    sources_per_coil : tuple[int, ...]
        True (unpadded) source count per coil, for reporting.
    """

    system_name: str
    source_position: np.ndarray
    source_direction: np.ndarray
    b_coeff: np.ndarray
    workspace: np.ndarray
    sources_per_coil: tuple[int, ...]

    @property
    def n_coils(self) -> int:
        return int(self.source_position.shape[0])

    @property
    def max_order(self) -> int:
        """Highest multipole order L present (1 = pure dipole)."""
        return int(self.b_coeff.shape[2])

    def in_workspace(self, p) -> bool:
        """Is ``p`` inside the fit validity box? (Advisory — bounds inclusive.)"""
        p = np.asarray(p, dtype=float)
        return bool(np.all(p >= self.workspace[:, 0]) and np.all(p <= self.workspace[:, 1]))

    @property
    def _real_mask(self) -> np.ndarray:
        """(K, S_max) bool — which source slots are real rather than padding."""
        mask = np.zeros(self.source_position.shape[:2], dtype=bool)
        for k, n_s in enumerate(self.sources_per_coil):
            mask[k, :n_s] = True
        return mask

    @property
    def workspace_diagonal_m(self) -> float:
        return float(np.linalg.norm(self.workspace[:, 1] - self.workspace[:, 0]))

    @property
    def max_source_distance_m(self) -> float:
        """Furthest fitted source from the workspace centre [m].

        A least-squares multipole fit is under-determined: it can place sources
        arbitrarily far away with arbitrarily large coefficients, so long as
        their fields cancel to the right answer *inside the workspace*. Such a
        fit is perfectly valid but numerically vicious — see
        :meth:`conditioning`.
        """
        centre = self.workspace.mean(axis=1)
        d = np.linalg.norm(self.source_position - centre, axis=-1)
        return float(d[self._real_mask].max())

    def field_cancellation_ratio(self, n_per_axis: int = 3) -> float:
        """Worst-case in-workspace field cancellation, ``|Σ_j b_j| / Σ_j |b_j|``.

        For each coil, at each point of an ``n_per_axis³`` grid spanning the
        workspace box, compare the coil's net unit-current field to the sum of
        its individual source-field magnitudes. Returns the worst (smallest)
        ratio over all coils and points. 1.0 = the sources reinforce; → 0 = the
        physical field is a small residual of large opposing contributions, so
        computing it costs significant digits.

        This is the metric that actually predicts single-precision loss::

            relative error  ≈  machine_epsilon / field_cancellation_ratio

        Measured on the shipped models (float32 ε ≈ 1.2e-7):

            model      ratio     predicted   measured
            navion_2   8.0e-3    1.5e-5      1.1e-5
            cardiomag  2.5e-1    4.8e-7      3.6e-7
            octomag    6.9e-5    1.7e-3      5.2e-4

        It is evaluated in *field* space, weighted by 1/r³, which is what makes
        it meaningful. A purely parametric moment-space version
        (``|Σ_j m_j| / Σ_j |m_j|``) is **not** a usable substitute: it ignores
        where the sources sit relative to the workspace, so it counts a source
        245 km away with an enormous moment the same as one 3 cm away — and it
        ranks the shipped models in the *wrong order* (it makes the Navion look
        worse than the OctoMag, which is the opposite of the truth).

        Current-independent by construction: each coil's column of ``A_b`` is
        summed independently, so the cancellation that matters for the actuation
        matrix is the one *within* a coil.
        """
        pos = jnp.asarray(self.source_position)
        d = jnp.asarray(self.source_direction)
        b = jnp.asarray(self.b_coeff)

        axes = [np.linspace(self.workspace[i, 0], self.workspace[i, 1], n_per_axis)
                for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

        worst = 1.0
        for k, n_s in enumerate(self.sources_per_coil):
            for p in grid:
                pj = jnp.asarray(p)
                per_src = jax.vmap(_source_field, in_axes=(None, 0, 0, 0))(
                    pj, pos[k, :n_s], d[k, :n_s], b[k, :n_s],
                )
                sum_abs = float(jnp.sum(jnp.linalg.norm(per_src, axis=-1)))
                if sum_abs <= 0.0:
                    continue
                net = float(jnp.linalg.norm(jnp.sum(per_src, axis=0)))
                worst = min(worst, net / sum_abs)
        return worst

    def conditioning(self) -> dict:
        """Numerical-conditioning diagnostics for this fit.

        Returns ``{'max_source_distance_m', 'workspace_diagonal_m',
        'distance_ratio', 'field_cancellation_ratio',
        'predicted_float32_rel_err', 'needs_float64'}``.

        ``needs_float64`` keys on the *predicted* single-precision error,
        ``ε_f32 / field_cancellation_ratio`` — the only quantity here shown to
        track the measured error (see :meth:`field_cancellation_ratio`).
        ``distance_ratio`` (furthest source, in workspace diagonals) is reported
        as an explanatory diagnostic: it is what *causes* a bad cancellation
        ratio, but it is a proxy, not the thing itself.
        """
        dist = self.max_source_distance_m
        diag = self.workspace_diagonal_m
        cancel = self.field_cancellation_ratio()
        predicted = float(_EPS_FLOAT32 / max(cancel, 1e-30))
        return {
            "max_source_distance_m": dist,
            "workspace_diagonal_m": diag,
            "distance_ratio": dist / max(diag, 1e-30),
            "field_cancellation_ratio": cancel,
            "predicted_float32_rel_err": predicted,
            "needs_float64": bool(predicted > _FLOAT32_REL_ERR_BUDGET),
        }


def _pad_sources(per_coil: list[list[dict]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack ragged per-coil source lists into rectangular padded arrays."""
    n_coils = len(per_coil)
    s_max = max(len(s) for s in per_coil)
    l_max = max((len(src["b"]) for coil in per_coil for src in coil), default=1)

    pos = np.zeros((n_coils, s_max, 3), dtype=float)
    direction = np.zeros((n_coils, s_max, 3), dtype=float)
    coeff = np.zeros((n_coils, s_max, l_max), dtype=float)

    # Inert defaults for padded slots.
    pos[..., 2] = _PAD_POSITION_M
    direction[..., 2] = 1.0

    for k, coil in enumerate(per_coil):
        for s, src in enumerate(coil):
            pos[k, s] = src["pos"]
            d = np.asarray(src["dir"], dtype=float)
            norm = np.linalg.norm(d)
            if norm == 0.0:
                raise ValueError(
                    f"Coil {k} source {s} has a zero Source_Direction; "
                    f"cannot normalise"
                )
            direction[k, s] = d / norm
            coeff[k, s, : len(src["b"])] = src["b"]
    return pos, direction, coeff


def load_mpem_calibration(
    name_or_path: Union[str, Path],
) -> MpemCalibration:
    """Load a calibrated MPEM model from an ETH-format YAML.

    Parameters
    ----------
    name_or_path : str | Path
        A bundled calibration name (see :data:`BUNDLED_CALIBRATIONS` —
        ``"navion_2"``, ``"octomag"``, ``"cardiomag"``) or a path to a
        calibration YAML in the upstream schema.

    Returns
    -------
    MpemCalibration

    Raises
    ------
    ValueError
        If a source declares interior (``A_Coeff``) terms, which are not
        implemented — no shipped ETH calibration uses them.

    Notes
    -----
    An ``Offset:`` block (current-independent sources modelling remanent /
    ambient field) is **ignored**, matching the upstream ``ForwardModelMPEM``
    default. Only coils named in ``Coil_List`` are read, so zero current gives
    exactly zero field.
    """
    import yaml  # local import: keep node import cheap

    if str(name_or_path) in BUNDLED_CALIBRATIONS:
        path = _CALIBRATION_DIR / BUNDLED_CALIBRATIONS[str(name_or_path)]
        if not path.exists():
            raise FileNotFoundError(
                f"Bundled calibration {name_or_path!r} not found at {path}. "
                f"This should ship with the package — the installation may be "
                f"incomplete. Pass an explicit path to a calibration YAML "
                f"instead."
            )
    else:
        path = Path(name_or_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Calibration file not found: {path}. Bundled names are "
                f"{sorted(BUNDLED_CALIBRATIONS)}."
            )

    with open(path) as fh:
        raw = yaml.safe_load(fh)

    coil_names = raw["Coil_List"]
    per_coil: list[list[dict]] = []
    for coil_name in coil_names:
        coil = raw[coil_name]
        sources = []
        for src_name in coil["Source_List"]:
            src = coil[src_name]
            if src.get("A_Coeff") is not None:
                raise ValueError(
                    f"{path.name}: {coil_name}/{src_name} declares A_Coeff "
                    f"(interior/solid-harmonic terms). Only exterior B_Coeff "
                    f"terms are implemented; no shipped ETH calibration uses "
                    f"A_Coeff."
                )
            b = src["B_Coeff"]
            if b is None:
                raise ValueError(
                    f"{path.name}: {coil_name}/{src_name} has neither A_Coeff "
                    f"nor B_Coeff"
                )
            sources.append({
                "pos": src["Source_Position"],
                "dir": src["Source_Direction"],
                "b": list(b),
            })
        per_coil.append(sources)

    pos, direction, coeff = _pad_sources(per_coil)
    workspace = np.asarray(raw["Workspace_Dimensions"], dtype=float)

    return MpemCalibration(
        system_name=str(raw.get("System_Name", path.stem)),
        source_position=pos,
        source_direction=direction,
        b_coeff=coeff,
        workspace=workspace,
        sources_per_coil=tuple(len(s) for s in per_coil),
    )


# ---------------------------------------------------------------------------
# MPEM field evaluation — JAX-traceable
# ---------------------------------------------------------------------------

def _legendre_P_dP(u: jnp.ndarray, L: int):
    """``P_l(u)`` and ``P_l'(u)`` for ``l = 1..L`` via stable recurrences.

    ``P_l   = ((2l−1)·u·P_{l−1} − (l−1)·P_{l−2}) / l``
    ``P_l'  = P_{l−2}' + (2l−1)·P_{l−1}``

    The derivative recurrence is used in preference to the closed form
    ``P_l'(u) = l(u·P_l − P_{l−1})/(u² − 1)``, which is singular at ``u = ±1``
    (i.e. on a source's symmetry axis — a perfectly ordinary query point).
    """
    Ps = [jnp.ones_like(u), u]
    dPs = [jnp.zeros_like(u), jnp.ones_like(u)]
    for l in range(2, L + 1):
        Ps.append(((2 * l - 1) * u * Ps[l - 1] - (l - 1) * Ps[l - 2]) / l)
        dPs.append(dPs[l - 2] + (2 * l - 1) * Ps[l - 1])
    return Ps[1:], dPs[1:]  # drop l = 0


def _source_field(
    p: jnp.ndarray, pos: jnp.ndarray, d_hat: jnp.ndarray, b: jnp.ndarray,
) -> jnp.ndarray:
    """Unit-current field of ONE axially-symmetric multipole source [T/A].

    ``b(p) = Σ_l B_l / r^{l+2} · [ (l+1)·P_l(u)·r̂ + P_l'(u)·(u·r̂ − d̂) ]``

    Derived as ``−∇ψ`` from ``ψ = Σ_l B_l P_l(u)/r^{l+1}``. ``d_hat`` is assumed
    already normalised (:func:`load_mpem_calibration` does this).
    """
    r = p - pos
    rn = jnp.maximum(jnp.linalg.norm(r), 1e-30)
    rh = r / rn
    u = jnp.dot(d_hat, rh)
    # Guard the Legendre argument: |d̂·r̂| can exceed 1 by an ulp, which makes
    # the recurrences (and jacrev through them) produce NaN.
    u = jnp.clip(u, -1.0, 1.0)

    L = b.shape[0]
    Ps, dPs = _legendre_P_dP(u, L)
    out = jnp.zeros(3, dtype=rh.dtype)
    for i in range(L):
        l = i + 1
        out = out + b[i] / rn ** (l + 2) * (
            (l + 1) * Ps[i] * rh + dPs[i] * (u * rh - d_hat)
        )
    return out


def _coil_field(p, pos, d_hat, b) -> jnp.ndarray:
    """Unit-current field of one coil = Σ over its sources. pos/d: (S,3) b: (S,L)."""
    return jnp.sum(
        jax.vmap(_source_field, in_axes=(None, 0, 0, 0))(p, pos, d_hat, b), axis=0,
    )


def _actuation_matrix(p, pos, d_hat, b) -> jnp.ndarray:
    """``A_b(p) ∈ R^{3×K}`` — columns are per-coil unit-current fields [T/A]."""
    return jax.vmap(_coil_field, in_axes=(None, 0, 0, 0))(p, pos, d_hat, b).T


def _b_total_world(
    target_world: jnp.ndarray,
    currents: jnp.ndarray,
    pos: jnp.ndarray,
    d_hat: jnp.ndarray,
    b: jnp.ndarray,
    background_field_world_t: jnp.ndarray,
) -> jnp.ndarray:
    """Total field at target = Σ_k i_k·b_k(target) + background."""
    return _actuation_matrix(target_world, pos, d_hat, b) @ currents \
        + background_field_world_t


def h_erf(currents: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """S-MPEM saturation, ``h_k(i_k) = (√π / 2β_k)·erf(β_k·i_k)``.

    ``h'(0) = 1`` exactly, so the saturated model reduces to the linear model at
    low current (Charreyron et al. 2021, Eq. 32).
    """
    return jnp.sqrt(jnp.pi) / (2.0 * beta) * jax.scipy.special.erf(beta * currents)


def field_actuation_matrix(p, calibration: MpemCalibration) -> jnp.ndarray:
    """Public helper: ``A_b(p) ∈ R^{3×K}`` [T/A] for an eMNS calibration.

    The object a controller inverts to allocate currents for a desired field::

        i = pinv(A_b(p)) @ B_desired

    Note this is *unconstrained* least squares — it respects no current limit.
    Clamp and re-check the achieved field afterwards.

    Caution for under-actuated arrays: on a 3-coil Navion ``A_b`` is 3×3 and
    invertible, so the field solution is unique and ``∇B`` is then **entirely
    determined** — field and gradient cannot be allocated independently.
    """
    return _actuation_matrix(
        jnp.asarray(p),
        jnp.asarray(calibration.source_position),
        jnp.asarray(calibration.source_direction),
        jnp.asarray(calibration.b_coeff),
    )


def gradient5(grad_b: jnp.ndarray) -> jnp.ndarray:
    """Pack a 3×3 gradient into the upstream 5-vector convention.

    ``[∂Bx/∂x, ∂Bx/∂y, ∂Bx/∂z, ∂By/∂y, ∂By/∂z]`` — the ordering used by
    ``mag_manip`` and Charreyron et al. In free space ``∇B`` is symmetric
    (curl-free) and trace-free (``∇·B = 0``), leaving 5 independent components;
    ``∂Bz/∂z = −(∂Bx/∂x + ∂By/∂y)`` recovers the sixth.
    """
    return jnp.array([
        grad_b[0, 0], grad_b[0, 1], grad_b[0, 2], grad_b[1, 1], grad_b[1, 2],
    ])


@stability(StabilityLevel.EXPERIMENTAL)
class CoilArrayFieldNode(MimeNode):
    """Current-controlled electromagnet array (eMNS) field source.

    Produces the magnetic field ``B`` and its spatial gradient ``∇B`` at a
    commanded target world position, given the coil current vector. The field
    model is a calibrated MPEM (see module docstring); ``∇B`` is computed by
    ``jax.jacrev`` of the same function that produces ``B``.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep [s].
    calibration : MpemCalibration
        Calibrated coil model, from :func:`load_mpem_calibration`. Passed as a
        constructed object rather than a path, mirroring the wall-table
        convention in ``StokesletFluidNode``.
    background_field_world_t : tuple[3], optional
        Uniform background field added to the coil field [T]. Defaults to
        **zero**, unlike ``PermanentMagnetNode`` (which defaults to a real
        Earth field): an eMNS runs at 5–50 mT, i.e. 100–1000× Earth's ~50 µT,
        and the ETH calibrations are fitted to the coil field alone — so a
        nonzero default would break parity with the upstream model. Set it
        explicitly if you care about the ~0.1–1 % Earth contribution.
    saturation_beta : tuple[K] | None, optional
        Per-coil S-MPEM saturation parameter ``β_k`` [1/A]. When given, the
        current is passed through :func:`h_erf` before superposition. ``None``
        (default) selects the linear model — correct below core saturation,
        which is where all the shipped calibrations were fitted.
    current_limit_a : float | None, optional
        Documented per-coil current limit [A], e.g. 35 for the Navion, 20 for
        CardioMag. **Advisory only — currents are NOT clamped**: silently
        clamping would misreport the field a controller actually commanded.
        Recorded in metadata and used by :meth:`within_current_limits`.
    allow_float32 : bool, optional
        Bypass the single-precision conditioning check. The constructor
        normally **raises** when the calibration is too ill-conditioned for the
        active float32 dtype (see :meth:`MpemCalibration.conditioning`), because
        the resulting error is silent. Set True only once you have established
        that the error is acceptable for your use.

    Boundary Inputs (commandable)
    -----------------------------
    coil_currents : (K,)
        Per-coil current [A]. Default zeros (= zero field). The control input.
    target_position_world : (3,)
        Point at which to evaluate ``B`` and ``∇B`` [m]. Default zero.

    State Fields (observable)
    -------------------------
    field_vector : (3,)
        Last-computed ``B`` at the target [T].
    field_gradient : (3,3)
        Last-computed ``∂B_i/∂x_j`` at the target [T/m].

    Notes
    -----
    Curl- and divergence-freeness are structural, not enforced: ``B = −∇ψ``
    with ψ harmonic makes ``∇B`` symmetric and trace-free automatically. This
    is the main reason to prefer MPEM over a data-driven field fit, whose
    gradient (and hence force) can violate Maxwell even when the field looks
    accurate.
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-103",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "Electromagnetic navigation system (eMNS) coil-array field source. "
            "Calibrated multipole electromagnet model (MPEM); field linear in "
            "coil currents; spatial gradient via jax.jacrev."
        ),
        governing_equations=(
            r"ψ_j(p) = Σ_l B_{j,l} P_l(u_j) / r_j^{l+1}; "
            r"b_k = −∇ Σ_{j∈k} ψ_j "
            r"= Σ_l B_l/r^{l+2} [(l+1)P_l(u) r̂ + P_l'(u)(u r̂ − d̂)]; "
            r"B(p) = Σ_k i_k b_k(p) = A_b(p) i; "
            r"∇B = ∂B/∂p; "
            r"h_k(i_k) = (√π/2β_k) erf(β_k i_k)  [S-MPEM saturation]"
        ),
        discretization=(
            "Analytical. Legendre P_l and P_l' by stable upward recurrence "
            "(the closed-form P_l' is singular at u=±1, i.e. on a source axis). "
            "∇B by jax.jacrev of the field function (automatic "
            "differentiation, machine precision). No spatial mesh."
        ),
        assumptions=(
            "Field is linear in coil current (cores below saturation) unless saturation_beta is supplied",
            "Coils are static — geometry fixed at calibration time; no coil motion",
            "Quasi-static: no eddy currents, no coil inductance/current dynamics, no field propagation delay",
            "Multipole sources are exterior-only (B_Coeff); interior A_Coeff terms are rejected at load",
            "Calibration is valid only inside the fitted Workspace_Dimensions box",
            "No shielding, eddy currents, or permeability contrast from tissue or apparatus",
        ),
        limitations=(
            "MPEM is an exterior expansion: it diverges as r→0 at a source. Sources lie inside the iron cores, outside the workspace, so in-workspace queries are safe — but do not query at a source",
            "Extrapolation outside Workspace_Dimensions is unvalidated; no error is raised (the check is not JAX-traceable)",
            "Linear model ignores core cross-magnetisation and saturation unless saturation_beta is given; Gervasoni 2024 reports this is a small effect for the Navion",
            "Currents are NOT clamped to current_limit_a — an unconstrained allocation can command a physically impossible field",
            "A DC Offset block (remanent/ambient field), if present in the calibration file, is ignored — matching upstream ForwardModelMPEM",
            "No coil temperature model; real duty cycles are thermally limited",
        ),
        validated_regimes=(
            ValidatedRegime("|B|", 0.0, 0.1, "T",
                            "eMNS operating band; Navion 5-50 mT, OctoMag ~10 mT"),
            ValidatedRegime("|∇B|", 0.0, 1.0, "T/m",
                            "Navion ~0.25 T/m single, ~1 T/m double (Landers 2025)"),
            ValidatedRegime("coil_current", -35.0, 35.0, "A",
                            "Navion +/-35 A; CardioMag +/-20 A; OctoMag few A"),
            ValidatedRegime("multipole_order", 1.0, 1.0, "-",
                            "All shipped ETH calibrations are order 1 (pure dipole); higher orders implemented but unvalidated"),
        ),
        references=(
            Reference("Petruska2017",
                      "Multipole electromagnet model — the MPEM formulation"),
            Reference("Charreyron2021",
                      "Modeling electromagnetic navigation systems; MPEM/S-MPEM benchmark"),
            Reference("Gervasoni2024",
                      "Navion: human-scale clinical eMNS; 3-coil planar, 5-50 mT, 250 mT/m"),
            Reference("Boehler2023",
                      "On the workspace of eMNS; actuation-matrix feasibility"),
            Reference("Landers2025",
                      "Clinically ready magnetic microrobots; OctoMag gradient pulling"),
        ),
        hazard_hints=(
            "Currents are NOT clamped. current_limit_a is advisory metadata. An "
            "unconstrained pseudoinverse allocation routinely returns currents far "
            "beyond hardware limits (upstream's own L2 backward model returns >60 A "
            "on a +/-20 A CardioMag). Always clamp and re-evaluate the achieved field.",
            "Outside Workspace_Dimensions the calibration extrapolates silently — no "
            "warning is emitted, because the check cannot be JAX-traced. Use "
            "MpemCalibration.in_workspace() at setup time.",
            "Under-actuated arrays cannot allocate field and gradient independently. "
            "On a 3-coil Navion rank(A_b) = 3: choosing B fixes ∇B entirely, so there "
            "is no null space to steer the gradient with.",
            "A single-sided planar array pulls a field-aligned moment TOWARD the coil "
            "plane in the overwhelming majority of the workspace, so treat 'push away' "
            "as unavailable when planning a trajectory — bidirectional gradient force "
            "is why Landers 2025 used a DOUBLE Navion. But this is an empirical "
            "tendency, NOT a theorem: measured over 2000 random (position, current) "
            "samples on the calibrated Navion_2, the outward direction occurs in ~0.2 % "
            "of samples, at ordinary field strengths (12-22 mT), concentrated within "
            "~15 mm of the coil-facing workspace boundary (y ≳ 0.12 m of a 0.134 m box) "
            "where the fit is least constrained. Earnshaw forbids a field MAXIMUM (and "
            "hence a stable trap), which is not the same as making ∇|B| point at the "
            "coils everywhere. Do not rely on a fixed force sign as an invariant.",
            "An ill-conditioned calibration loses precision SILENTLY in float32 — no "
            "error, no NaN, just wrong forces. A multipole least-squares fit may place "
            "sources far outside the hardware with huge coefficients whose fields "
            "nearly cancel inside the workspace. The shipped OctoMag fit does exactly "
            "this (sources ~245 km from a 2 cm workspace, coefficients ~1e9, worst "
            "in-workspace field cancellation 6.5e-4): measured float32 error is ~5e-4 "
            "on the field and ~1e-3 on the GRADIENT, i.e. ~0.1% straight into any "
            "gradient-pull force. The constructor RAISES on this combination rather "
            "than warning; use x64. On GPU also set "
            "JAX_DEFAULT_MATMUL_PRECISION=highest — TF32 would compound it.",
            "Source_Position values are fitted basis-function locations, NOT physical "
            "coil centres. Navion coils contain near-cancelling dipole pairs whose "
            "individual moments are ~100x the net; do not interpret them as hardware "
            "or use them for collision geometry.",
        ),
        implementation_map={
            "ψ_j = Σ_l B_{j,l} P_l(u_j)/r_j^{l+1}; b = −∇ψ":
                "mime.nodes.actuation.coil_array_field._source_field",
            "P_l, P_l' by stable recurrence":
                "mime.nodes.actuation.coil_array_field._legendre_P_dP",
            "b_k = Σ_{j∈k} b_j (per-coil source sum)":
                "mime.nodes.actuation.coil_array_field._coil_field",
            "A_b(p) ∈ R^{3×K} (actuation matrix)":
                "mime.nodes.actuation.coil_array_field._actuation_matrix",
            "B = Σ_k i_k b_k + B_background":
                "mime.nodes.actuation.coil_array_field._b_total_world",
            "h_k(i_k) = (√π/2β_k) erf(β_k i_k) (S-MPEM)":
                "mime.nodes.actuation.coil_array_field.h_erf",
            "∇B = jax.jacrev(B_total, target)":
                "mime.nodes.actuation.coil_array_field.CoilArrayFieldNode.update",
            "G5 packing [∂Bx/∂x,∂Bx/∂y,∂Bx/∂z,∂By/∂y,∂By/∂z]":
                "mime.nodes.actuation.coil_array_field.gradient5",
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.EXTERNAL_APPARATUS,
        actuation=ActuationMeta(
            principle=ActuationPrinciple.GRADIENT_MAGNETIC_FIELD,
            is_onboard=False,
            max_field_strength_mt=50.0,
            max_gradient_t_per_m=1.0,
            commandable_fields=("coil_currents",),
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        calibration: MpemCalibration,
        background_field_world_t: tuple[float, float, float] = (0.0, 0.0, 0.0),
        saturation_beta: Optional[tuple[float, ...]] = None,
        current_limit_a: Optional[float] = None,
        allow_float32: bool = False,
        **kwargs,
    ):
        if not isinstance(calibration, MpemCalibration):
            raise TypeError(
                f"calibration must be an MpemCalibration (from "
                f"load_mpem_calibration()), got {type(calibration).__name__}"
            )
        n_coils = calibration.n_coils
        if saturation_beta is not None:
            saturation_beta = tuple(float(b) for b in saturation_beta)
            if len(saturation_beta) != n_coils:
                raise ValueError(
                    f"saturation_beta has {len(saturation_beta)} entries but "
                    f"the calibration has {n_coils} coils"
                )
            if any(b <= 0.0 for b in saturation_beta):
                raise ValueError("saturation_beta entries must be positive")

        super().__init__(
            name, timestep,
            system_name=calibration.system_name,
            n_coils=n_coils,
            background_field_world_t=tuple(
                float(c) for c in background_field_world_t),
            saturation_beta=saturation_beta,
            current_limit_a=(
                None if current_limit_a is None else float(current_limit_a)),
            **kwargs,
        )
        # Heavy arrays live outside params (params feed graph.json serialisation);
        # the calibration is identified there by system_name.
        self._cal = calibration
        self._pos = jnp.asarray(calibration.source_position)
        self._dir = jnp.asarray(calibration.source_direction)
        self._b = jnp.asarray(calibration.b_coeff)

        # An ill-conditioned fit loses precision in float32 SILENTLY rather than
        # failing, so refuse to construct one. Construction is the right place:
        # it is the first moment both the calibration and the active dtype are
        # known, and it fires once rather than every step. A warning is too easy
        # to lose in a long run.
        cond = calibration.conditioning()
        self._conditioning = cond
        if cond["needs_float64"] and self._pos.dtype != jnp.float64 \
                and not allow_float32:
            raise RuntimeError(
                f"CoilArrayFieldNode {name!r}: the {calibration.system_name!r} "
                f"calibration is too ill-conditioned for float32.\n"
                f"  worst in-workspace field cancellation : "
                f"{cond['field_cancellation_ratio']:.3g}\n"
                f"  predicted float32 relative error      : "
                f"{cond['predicted_float32_rel_err']:.3g}  "
                f"(budget {_FLOAT32_REL_ERR_BUDGET:.0e})\n"
                f"  furthest fitted source               : "
                f"{cond['max_source_distance_m']:.3g} m from a "
                f"{cond['workspace_diagonal_m']:.3g} m workspace\n"
                f"The field is a small residual of large opposing source "
                f"contributions, so single precision loses accuracy with no "
                f"error and no NaN — the failure this exception exists to "
                f"prevent. Measured on this fit: ~5e-4 relative on the field "
                f"and ~1e-3 on the gradient, i.e. ~0.1% straight into any "
                f"gradient-pull force.\n"
                f"Fix: enable x64 — JAX_ENABLE_X64=1, or "
                f"jax.config.update('jax_enable_x64', True) before constructing "
                f"nodes. On GPU also set JAX_DEFAULT_MATMUL_PRECISION=highest.\n"
                f"Override with allow_float32=True only if you have established "
                f"that this error is acceptable for your use."
            )

    @property
    def calibration(self) -> MpemCalibration:
        return self._cal

    @property
    def n_coils(self) -> int:
        return self._cal.n_coils

    def within_current_limits(self, currents) -> bool:
        """Are all commanded currents inside ``current_limit_a``? Advisory."""
        limit = self.params["current_limit_a"]
        if limit is None:
            return True
        return bool(np.all(np.abs(np.asarray(currents)) <= limit))

    def initial_state(self) -> dict:
        return {
            "field_vector":   jnp.zeros(3),
            "field_gradient": jnp.zeros((3, 3)),
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "coil_currents": BoundaryInputSpec(
                shape=(self.n_coils,), default=jnp.zeros(self.n_coils),
                description="Per-coil current [A]; the eMNS control input",
            ),
            "target_position_world": BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description="Target point at which to evaluate B and ∇B [m]",
            ),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        currents = jnp.asarray(
            boundary_inputs.get("coil_currents", jnp.zeros(self.n_coils)))
        target = jnp.asarray(
            boundary_inputs.get("target_position_world", jnp.zeros(3)))

        beta = self.params["saturation_beta"]
        if beta is not None:
            currents = h_erf(currents, jnp.asarray(beta))

        background = jnp.asarray(self.params["background_field_world_t"])

        # Single code path for B and ∇B: the gradient is jax.jacrev of the very
        # function that produces the field, so the two cannot disagree.
        def _field_fn(t):
            return _b_total_world(
                t, currents, self._pos, self._dir, self._b, background,
            )

        B = _field_fn(target)
        grad_B = jax.jacrev(_field_fn)(target)

        return {
            "field_vector":   B,
            "field_gradient": grad_B,
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "field_vector":   state["field_vector"],
            "field_gradient": state["field_gradient"],
        }
