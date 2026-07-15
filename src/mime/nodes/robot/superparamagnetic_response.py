"""SuperparamagneticResponseNode — force on a field-aligned moment from a measured m(|B|) curve.

This node models a **superparamagnetic** body — a bead, capsule, or particle
loaded with single-domain magnetic nanoparticles — actuated by **gradient
pulling** in the field of an electromagnetic navigation system (eMNS). It is the
body-side counterpart of ``CoilArrayFieldNode`` (MIME-NODE-103): that node emits
``(field_vector, field_gradient)``; this one turns them into
``(magnetic_force, magnetic_torque)`` for a ``RigidBodyNode``.

It completes a family of three magnetic-response nodes, which differ in *where
the moment comes from*:

* ``MagneticResponseNode`` (MIME-NODE-002) — soft magnet at the **demagnetization
  ceiling**: ``chi_diag = [1/n_axi, 1/n_rad, 1/n_rad]``, i.e. the ``chi -> inf``
  limit of ``chi_eff = chi/(1 + N*chi)``. Anisotropic, so it produces torque.
* ``PermanentMagnetResponseNode`` (MIME-NODE-008) — **body-fixed** moment,
  ``m_body = n * m_single * axis_hat``, rotated to the lab frame by the body
  quaternion. Produces torque; that is its whole point.
* ``SuperparamagneticResponseNode`` (this node) — **field-aligned** moment whose
  magnitude comes from a *measured magnetization curve*: ``m = m(|B|) * B_hat``.
  Isotropic and orientation-free, so the torque vanishes identically and the
  force is the entire physics.

Physics
-------
A superparamagnetic body has no easy axis and no remanence: its nanoparticle
moments are thermally free to follow the applied field, so the net moment is
**parallel to B by construction**::

    m(B) = m(|B|) * B_hat            [A.m^2]

    T = m x B  ==  0                 (structurally — m is parallel to B)
    F = (m . grad) B = grad_B @ m    (F_i = m_j * dB_i/dx_j)

The vanishing torque is a *consequence* of field alignment, not an assumption
bolted on afterwards. See "Torque is identically zero" below for what that costs.

**The force expression is exact for an induced moment, but the energy is not.**
For a *constant* moment, ``F = grad(m.B)``. For an *induced* moment the stored
energy carries a factor of one half (``U = -1/2 m.B``), and it is a standard
error to propagate that half into the force. It does not belong there: the force
on each elementary dipole is ``(m.grad)B`` with the *instantaneous* moment
regardless of how that moment arose, so::

    F = grad_B @ m        <-- what this node computes; correct for both cases

For a curl-free field (``grad_B`` symmetric, which every physical field and
``CoilArrayFieldNode`` guarantee structurally) this is equivalent to::

    F = m(|B|) * grad|B|         with   grad|B| = grad_B^T @ B_hat

which shows the force is **conservative**, with potential
``Phi(|B|) = integral_0^|B| m(b) db`` — reducing to the familiar
``chi V grad(|B|^2) / (2 mu_0)`` in the linear regime, half and all. The
equivalence of the two forms is pinned by MIME-VER-146; it is a genuine check of
the whole chain, because it holds only if the upstream gradient is
Maxwell-consistent.

DO NOT APPLY A DEMAGNETIZATION CORRECTION ON THE MEASURED PATH
--------------------------------------------------------------
**This is the single most likely way to misuse this node.**

A vibrating-sample magnetometer applies a known field to the *real object* and
measures the moment that results. The object's own self-demagnetization is
therefore **already baked into the measured curve** — ``m_measured(|B_applied|)``
*is* ``chi_eff``, not ``chi``. The field ``CoilArrayFieldNode`` emits at the bead
is likewise the *applied* field. So:

* the measured curve consumes the **applied** field, and
* correcting it again for demagnetization **double-counts** and *under-predicts*
  the moment, hence the force.

Both paths in this node consume the applied field, for different reasons:

===================  ================================  ==========================
path                 constitutive law                  where demag lives
===================  ================================  ==========================
measured (primary)   ``m = m_curve(|B_applied|)``      already inside the curve
first-principles     ``m = chi/(1+N chi) * H_app * V`` the ``1/(1+N chi)`` factor
===================  ================================  ==========================

The error to guard against is **stacking** them: running a measured curve through
a demagnetization correction, or feeding an intrinsic ``chi`` an internal field
that has already had the demag term subtracted. Pick one row and stop.

Volume scaling is exact for spheres
-----------------------------------
A sphere's demagnetization factor is ``N = 1/3`` **independent of its size**. For
the same material and shape, then::

    m_new(|B|) = (V_new / V_ref) * m_ref(|B|)

is *exact* — no re-derivation and no re-measurement per size. Every curve here
therefore carries the ``reference_volume_m3`` it was measured at, and the node
rescales by ``volume_m3 / reference_volume_m3``. This is what makes a diameter
sweep free, and it is a first-class feature rather than a convenience: the
Landers capsule's size is *ambiguous in the source* (see below), so being able to
sweep it exactly is a requirement, not a nicety.

The exactness is a property of the **sphere**. For a non-spherical body ``N``
depends on aspect ratio but not on scale, so the scaling still holds under pure
*isotropic* rescaling — but it does **not** license changing the shape.

THE SMALL-BODY ASSUMPTION
-------------------------
The field at the bead **centre** is taken as representative of the field across
the **whole bead**. This is the condition under which a VSM curve transfers into
a MIME experiment at all, and it is not free:

* a VSM applies a **uniform** field, whereas
* here the field has a **gradient by construction** — that is the entire point of
  gradient pulling.

The relevant measure is the fractional variation of ``|B|`` across the body::

    delta|B| / |B|  ~  |grad|B|| * d / |B|

For the Landers capsule (``d = 1.69 mm``, ``|B| = 30 mT``) the number depends on
which gradient the eMNS is actually running, and the spread is a factor of 4:

===================  =============  ==============================  ==========
``|grad B|``         ``delta|B|``   ``delta|B|/|B|``                regime
===================  =============  ==============================  ==========
0.25 T/m (single)    0.42 mT        **1.4 %**                       comfortable
0.5 T/m              0.845 mT       **2.8 %**                       good
1.0 T/m (double)     1.69 mT        **5.6 %**                       at the edge
===================  =============  ==============================  ==========

So the commonly-quoted "good to ~3 %" is the **0.5 T/m** figure. At the double-
gradient setting — which is precisely the mode Landers uses to pull *against
flow*, i.e. the anchor's most demanding configuration — the assumption is only
good to ~6 %, and a 2 mm body there reaches 6.7 %. Quote the number for the
gradient you are running, not the headline one.

It is **linear in ``d`` and in ``|grad B|``, and inverse in ``|B|``**, so it
degrades exactly where gradient pulling gets interesting: **larger bodies**,
**tighter gradients**, and **weak-field regions**. The last is the sharpest — the
ratio diverges as ``|B| -> 0`` regardless of body size, so a field null is where
this model breaks first, not a big bead. :func:`small_body_field_variation_ratio`
computes the ratio so a caller can check their own regime; the node exposes it as
a method against its own diameter. **It is not checked at runtime** — the test is
not JAX-traceable, and a per-step host callback is not worth it. Call it yourself.

Torque is identically zero — read this before wiring
-----------------------------------------------------
``m x B == 0`` structurally, so this node emits **exact zeros** for
``magnetic_torque`` rather than the ~1e-20 N.m of floating-point noise that
evaluating the cross product would produce. Injecting roundoff into a body
integrator would be strictly worse than injecting nothing.

The zero survives a stronger test than the point-dipole argument. One might worry
that the *distributed* force over a body in a **gradient** exerts a net torque
about the centre of mass even when the local ``m x B`` vanishes. It does not, for
a sphere with uniform field-aligned magnetization::

    tau = integral r x (M.grad)B(r) dV
        = (V R^2 / 5) * eps_ijk * M_l * d_l d_j B_k     (the constant part
                                                         integrates to zero)
        = 0

because ``d_j B_k`` is **symmetric** in ``(j, k)`` for a curl-free field, while
``eps_ijk`` is **antisymmetric** in ``(j, k)`` — the contraction vanishes
identically. So the exact zero is a consequence of Maxwell plus sphericity, not
merely of the point-dipole idealisation. (It does assume uniform ``M`` across the
body, which is the small-body assumption again.)

The consequence is that **this node cannot reorient a body**. If it is a body's
only magnetic effect, that body's orientation is *unactuated* — it will tumble
under drag and contact alone. That is correct for an ideal superparamagnetic
sphere and is exactly why gradient pulling, not field steering, is the actuation
mode. A real capsule has some shape anisotropy and hence a small real aligning
torque; this node models it as zero. If you need it, you need an anisotropic
response node, not a parameter on this one.

Because the response is isotropic and field-aligned, **body orientation is
physically irrelevant** and this node does **not** declare an ``orientation``
boundary input. See the class docstring for why that is a deliberate interface
decision rather than an omission.

The anchor: Landers et al. (2025)
---------------------------------
A gelatin microcapsule loaded with zinc-substituted iron-oxide nanocubes (18.4 nm
edge) plus radiopaque tantalum, gradient-pulled against flow by an eMNS.

Two numbers in the source **do not reconcile, and the paper never addresses it**:

* the main text reports a mean diameter of **1.69 mm**, whereas
* the paper's own CFD models the capsule as **1.4 mm** at 3187.74 kg/m^3.

The volume ratio is ``(1.4/1.69)^3 = 0.57`` — a **43 % swing in the moment, and
therefore in the force**. This node takes no position on which is right; exact
volume scaling means both are one constructor argument apart, and they should be
swept and compared rather than silently picked. :data:`LANDERS_CAPSULE_DIAMETER_M`
and :data:`LANDERS_CFD_DIAMETER_M` are both provided, deliberately without a
default preference.

The magnetics were measured by VSM (EZ9, Microsense) **on the capsules
themselves** — not on the bulk material, and not back-computed. That is what
makes the measured path legitimate and the demag warning above binding.

Why the two existing response nodes do not fit
----------------------------------------------
``MagneticResponseNode`` *is* a soft-magnetic node and *does* handle
demagnetization — but it hardwires the susceptibility to the demagnetization
**ceiling**, ``chi_diag = [1/n_axi, 1/n_rad, 1/n_rad]``. For a sphere
(``n = 1/3``) that pins ``chi_eff = 3`` and predicts::

    m = 3 * B/mu_0 * V = 1.81e-4 A.m^2   at 30 mT, d = 1.69 mm

against the **measured 7.2e-5** — **2.5x too high**. The susceptibility cannot be
lowered without violating ``n_axi + 2 n_rad = 1``, and any attempt to do so by
detuning the ``n`` factors invents spurious anisotropy and hence spurious torque.
``PermanentMagnetResponseNode`` assumes a body-fixed moment, which is simply the
wrong physics for a thermally-free superparamagnet.

For the record, the two paths here *agree*: an intrinsic ``chi = 1.98`` gives
``chi_eff = chi/(1 + chi/3) = 1.19`` and reproduces the measured 7.2e-5 exactly —
a physically ordinary value for a dense iron-oxide loading, and comfortably below
the ``chi_eff = 3`` ceiling. The measurement is consistent with a real material;
it is MIME-NODE-002's *ceiling* assumption that is wrong here, not the data.
Pinned by MIME-VER-148.

Algorithm ID: MIME-NODE-009
Stability: EXPERIMENTAL
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

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
    MimeNodeMeta, NodeRole, BiocompatibilityMeta, BiocompatibilityClass,
)


MU_0 = 4.0e-7 * jnp.pi          # Permeability of free space [T.m/A]

#: 1 memu (milli-emu) in SI. VSM data is conventionally quoted in emu;
#: 1 emu = 1e-3 A.m^2, so 1 memu = 1e-6 A.m^2. Landers 2025 quotes memu.
MEMU_TO_AM2 = 1.0e-6

# ── The Landers et al. (2025) capsule ───────────────────────────────────────
#
# NOTE the unresolved size ambiguity: the main text says 1.69 mm; the paper's own
# CFD says 1.4 mm. Both are exported, neither is blessed. See the module
# docstring.

#: Mean capsule diameter from the Science main text [m].
LANDERS_CAPSULE_DIAMETER_M = 1.69e-3

#: Capsule diameter used by the paper's *own* CFD model [m]. Irreconcilable with
#: the above; the volume ratio is 0.57, i.e. a 43 % swing in the force.
LANDERS_CFD_DIAMETER_M = 1.4e-3

#: Capsule density used by the paper's own CFD model [kg/m^3].
LANDERS_CFD_DENSITY_KG_M3 = 3187.74

#: VSM moment at 30 mT [A.m^2] = 72 memu. Measured on the capsules themselves.
LANDERS_M_AT_30MT_AM2 = 72.0 * MEMU_TO_AM2

#: The field at which the above was measured [T].
LANDERS_B_MEASURED_T = 30.0e-3

#: Saturation moment [A.m^2] ~ 225 memu.
LANDERS_M_SAT_AM2 = 225.0 * MEMU_TO_AM2

#: Quoted low-field slope, 2.4 memu/mT [A.m^2 / T].
#:
#: **This is a chord, not a tangent.** 2.4 memu/mT x 30 mT = 72.0 memu *exactly*,
#: so this number and the 30 mT measurement are not independent data — one is
#: derived from the other. That matters: it is why no Langevin can fit all three
#: quoted numbers at once (see :meth:`LangevinCurve.landers_capsule`).
LANDERS_CHORD_SLOPE_AM2_PER_T = 2.4 * MEMU_TO_AM2 / 1.0e-3

# Langevin series/direct crossover. Below this the direct form coth(x) - 1/x
# catastrophically cancels (both terms ~1/x, difference ~x/3, so the relative
# error goes as ~3*eps/x^2); above it the 4-term Maclaurin series truncates at
# ~x^8/15592. At x = 0.1 the two errors are ~6.6e-14 (f64 direct) and ~6.4e-13
# (series) — both far below the ~5 % uncertainty on any real VSM curve.
_LANGEVIN_SERIES_CUTOFF = 0.1


def sphere_volume_m3(diameter_m: float) -> float:
    """Volume of a sphere of the given diameter [m^3]."""
    return float(np.pi / 6.0 * diameter_m ** 3)


def sphere_equivalent_diameter_m(volume_m3: float) -> float:
    """Diameter of the sphere with the given volume [m], ``d = (6V/pi)^(1/3)``.

    Exact for a sphere; for any other shape it is the equivalent-sphere
    diameter, which is the right length scale for the small-body check.
    """
    return float((6.0 * volume_m3 / np.pi) ** (1.0 / 3.0))


def _safe_norm(v: jnp.ndarray) -> jnp.ndarray:
    """``|v|``, returning 0 with a 0 gradient at ``v = 0`` instead of NaN.

    ``jnp.linalg.norm`` is not differentiable at the origin (``d|v|/dv = v/|v|``
    is 0/0), and a plain ``jnp.where`` guard does **not** help: reverse-mode AD
    evaluates *both* branches and NaN propagates through the untaken one. Hence
    the double-where — the inner guard keeps the untaken branch finite.

    The gradient of ``|B|`` at ``B = 0`` is genuinely undefined (the norm has a
    cone point there); returning 0 is a *choice* that keeps the model traceable
    through a field null rather than a statement about the physics.
    """
    sq = jnp.sum(v * v)
    safe_sq = jnp.where(sq > 0.0, sq, 1.0)
    return jnp.where(sq > 0.0, jnp.sqrt(safe_sq), 0.0)


def _langevin_np(x: float) -> float:
    """Host-side float64 Langevin, for curve *fitting* at construction time.

    Deliberately NOT the JAX path: a fit is host code that runs once, and its
    result (``b0``) must be reproducible regardless of whether the caller happens
    to have ``JAX_ENABLE_X64`` set. Routing the bisection through ``jnp`` would
    silently make the fitted parameter float32-accurate in a default-dtype
    session and float64-accurate under x64 — the same curve object differing by
    ~1e-7 relative depending on ambient config. Same series/cutoff logic as
    :func:`langevin`; the two agree to float64 precision (MIME-VER-145).
    """
    ax = abs(x)
    if ax < _LANGEVIN_SERIES_CUTOFF:
        x2 = x * x
        return x * (1.0 / 3.0 + x2 * (-1.0 / 45.0
                    + x2 * (2.0 / 945.0 + x2 * (-1.0 / 4725.0))))
    return 1.0 / np.tanh(x) - 1.0 / x


def langevin(x: jnp.ndarray) -> jnp.ndarray:
    """Langevin function ``L(x) = coth(x) - 1/x``, exact and NaN-free at ``x=0``.

    ``L(0) = 0`` analytically, but the direct form evaluates ``inf - inf`` there
    and cancels catastrophically nearby. For ``|x| < 0.1`` this uses the Maclaurin
    series::

        L(x) = x/3 - x^3/45 + 2 x^5/945 - x^7/4725 + O(x^9)

    guarded by the double-where idiom so that reverse-mode AD does not propagate
    a NaN from the untaken branch. ``L'(0) = 1/3`` exactly, which is what makes
    the low-field susceptibility come out right.
    """
    ax = jnp.abs(x)
    small = ax < _LANGEVIN_SERIES_CUTOFF

    # Double-where: keep BOTH branches finite for AD, then select.
    safe_x = jnp.where(small, 1.0, x)
    direct = 1.0 / jnp.tanh(safe_x) - 1.0 / safe_x

    xs = jnp.where(small, x, 0.0)
    x2 = xs * xs
    series = xs * (1.0 / 3.0 + x2 * (-1.0 / 45.0
                   + x2 * (2.0 / 945.0 + x2 * (-1.0 / 4725.0))))

    return jnp.where(small, series, direct)


def small_body_field_variation_ratio(
    field_vector: jnp.ndarray,
    field_gradient: jnp.ndarray,
    diameter_m: float,
) -> jnp.ndarray:
    """``delta|B| / |B|`` across a body of diameter ``d`` — the small-body check.

    The node evaluates the magnetization curve at the field at the body *centre*
    and applies it to the whole body. That is legitimate only while ``|B|`` is
    nearly constant across the body — a VSM curve is measured in a *uniform*
    field, whereas gradient pulling supplies a gradient by construction. This
    quantifies the discrepancy::

        delta|B|/|B|  =  |grad|B|| * d / |B|        grad|B| = grad_B^T @ B_hat

    Note it is the gradient of the **magnitude** that matters, not the norm of
    ``grad_B``: the curve is a function of ``|B|``, so only the part of the
    gradient that changes ``|B|`` breaks the assumption. A pure rotation of ``B``
    across the body leaves the moment's magnitude untouched.

    Returns ``inf`` where ``|B| = 0``. That is the honest value, not a guard: the
    ratio *diverges* as ``|B| -> 0`` for any body size (0.845 at 1 mT, 84.5 at
    10 uT for the Landers capsule at 0.5 T/m), and a field null is where this
    model breaks first and worst. Returning 0 there — the most reassuring value
    a caller could read — would report "transfers cleanly" at exactly the voxel
    that must be flagged, which matters because an eMNS null can be *exactly*
    zero by symmetry in a computed field, and because sweeping this diagnostic
    across a workspace to find the breakdown is its intended use. (At ``B = 0``
    *and* ``grad_B = 0`` the ratio is genuinely indeterminate rather than
    infinite, but a body in an identically-zero field feels no force at all, so
    the diagnostic is moot there; ``inf`` remains the conservative answer.)

    This is a diagnostic, not a physics path — ``update()`` never calls it, so
    the ``inf`` cannot reach the force.

    Reference values — the Landers capsule (``d = 1.69 mm``, ``|B| = 30 mT``):
    **0.014** at 0.25 T/m (Navion single-gradient), **0.028** at 0.5 T/m, and
    **0.056** at 1 T/m (double-gradient, the pull-against-flow mode).

    Rule of thumb: below ~0.05 the measured curve transfers cleanly; by ~0.2 the
    centre field is no longer representative and the moment is a fiction. Note
    the anchor device *itself* crosses 0.05 in double-gradient mode — this is a
    real constraint that bites at the operating point, not a formality. This is
    a *diagnostic for the caller* — the node does not check it at runtime (the
    check is not JAX-traceable).
    """
    B = jnp.asarray(field_vector)
    grad_B = jnp.asarray(field_gradient)
    b_mag = _safe_norm(B)
    safe_mag = jnp.where(b_mag > 0.0, b_mag, 1.0)
    b_hat = B / safe_mag
    grad_b_mag = grad_B.T @ b_hat
    ratio = _safe_norm(grad_b_mag) * diameter_m / safe_mag
    # inf, not 0, at a field null: the ratio diverges there and this diagnostic
    # exists to say so. The double-where above keeps the computation NaN-free;
    # only the reported value changes.
    return jnp.where(b_mag > 0.0, ratio, jnp.inf)


# ---------------------------------------------------------------------------
# Magnetization curves — |B| [T] -> moment [A.m^2] at a reference volume
# ---------------------------------------------------------------------------

class MagnetizationCurve(ABC):
    """A measured or modelled ``m(|B|)`` for a body of ``reference_volume_m3``.

    Curves are **volume-referenced**: each carries the volume of the object the
    curve describes, so the node can rescale exactly (``N = 1/3`` for a sphere is
    scale-invariant — see the module docstring). Implementations must be pure
    JAX, traceable, differentiable, and finite at ``|B| = 0``.

    Every curve consumes the **APPLIED** field. None of them applies a
    demagnetization correction to a measured input; see the module docstring.
    """

    #: Volume of the object this curve was measured/derived for [m^3].
    reference_volume_m3: float

    #: Short tag recorded in node params for graph.json provenance.
    kind: str = "abstract"

    @abstractmethod
    def moment_am2(self, b_mag: jnp.ndarray) -> jnp.ndarray:
        """Moment magnitude [A.m^2] at applied field magnitude ``b_mag`` [T]."""

    @abstractmethod
    def describe(self) -> str:
        """One-line human-readable summary, recorded in node params."""

    @property
    @abstractmethod
    def m_sat_am2(self) -> float:
        """Saturation moment [A.m^2] at ``reference_volume_m3``."""

    def initial_susceptibility_am2_per_t(self) -> float:
        """``dm/d|B|`` as ``|B| -> 0`` [A.m^2/T], by AD of the curve itself.

        The *tangent* at the origin — not a chord. Reported because the two are
        routinely confused in the literature (Landers' quoted "2.4 memu/mT" is a
        chord; see :data:`LANDERS_CHORD_SLOPE_AM2_PER_T`).
        """
        return float(jax.grad(lambda b: self.moment_am2(b))(1e-9))


class LangevinCurve(MagnetizationCurve):
    """Textbook superparamagnetic curve, ``m = m_sat * L(|B| / b0)``.

    ``L(x) = coth(x) - 1/x`` is the classical (Langevin) magnetization of an
    ensemble of thermally-free, non-interacting single-domain moments — the
    defining physics of a superparamagnet, and the honest default for iron-oxide
    nanocubes. Two parameters, both physically meaningful:

    * ``m_sat_am2`` — every moment aligned, the ``|B| -> inf`` asymptote;
    * ``b0_t`` — the knee, ``b0 = k_B T / mu_particle``; the initial
      susceptibility is ``m_sat / (3 b0)``.

    Chosen over the alternatives because it is **smooth, monotone, saturating and
    exactly linear at low field with the right 1/3 coefficient** — all of which a
    linear-plus-clamp model (:class:`LinearSaturationCurve`) gets wrong precisely
    at the knee, which is where the Landers capsule *operates* (``x ~ 1.03`` at
    30 mT). A kink in ``m(|B|)`` becomes a discontinuity in the force, and this
    node is meant to be differentiated.

    Its honest limitation: a single Langevin describes a **monodisperse,
    non-interacting** ensemble. A real capsule is polydisperse and dipolar-coupled,
    so its true knee is *broader* than any single Langevin. That is not a
    hypothetical — it is measurable in the Landers numbers themselves, and it is
    why :meth:`landers_capsule` cannot fit all three. When a real VSM sweep is
    available, prefer :class:`InterpolatedCurve` and let the data have the last
    word.

    Parameters
    ----------
    m_sat_am2 : float
        Saturation moment [A.m^2] at ``reference_volume_m3``.
    b0_t : float
        Langevin field scale [T]; the knee.
    reference_volume_m3 : float
        Volume of the object the curve describes [m^3].
    """

    kind = "langevin"

    def __init__(self, m_sat_am2: float, b0_t: float, reference_volume_m3: float):
        if m_sat_am2 <= 0.0:
            raise ValueError(f"m_sat_am2 must be positive, got {m_sat_am2!r}")
        if b0_t <= 0.0:
            raise ValueError(
                f"b0_t must be positive, got {b0_t!r}. It is the Langevin field "
                f"scale k_B T / mu_particle; b0 -> 0 is the fully-saturated "
                f"limit and b0 -> inf the linear one, but neither is reachable "
                f"as a value."
            )
        if reference_volume_m3 <= 0.0:
            raise ValueError(
                f"reference_volume_m3 must be positive, got "
                f"{reference_volume_m3!r}"
            )
        self._m_sat = float(m_sat_am2)
        self._b0 = float(b0_t)
        self.reference_volume_m3 = float(reference_volume_m3)

    @property
    def m_sat_am2(self) -> float:
        return self._m_sat

    @property
    def b0_t(self) -> float:
        return self._b0

    def moment_am2(self, b_mag: jnp.ndarray) -> jnp.ndarray:
        return self._m_sat * langevin(jnp.asarray(b_mag) / self._b0)

    def describe(self) -> str:
        return (
            f"langevin(m_sat={self._m_sat:.6g} A.m^2, b0={self._b0:.6g} T) "
            f"@ V={self.reference_volume_m3:.6g} m^3"
        )

    # ── Fitting ─────────────────────────────────────────────────────────────

    @classmethod
    def from_saturation_and_slope(
        cls, m_sat_am2: float, slope_am2_per_t: float, reference_volume_m3: float,
    ) -> "LangevinCurve":
        """Fit to saturation and the **initial (tangent) susceptibility**.

        ``dm/dB|_0 = m_sat / (3 b0)``  =>  ``b0 = m_sat / (3 * slope)``. Closed
        form, no iteration.
        """
        if slope_am2_per_t <= 0.0:
            raise ValueError(
                f"slope_am2_per_t must be positive, got {slope_am2_per_t!r}"
            )
        return cls(m_sat_am2, m_sat_am2 / (3.0 * slope_am2_per_t),
                   reference_volume_m3)

    @classmethod
    def from_saturation_and_point(
        cls, m_sat_am2: float, b_t: float, m_at_b_am2: float,
        reference_volume_m3: float,
    ) -> "LangevinCurve":
        """Fit to saturation and one measured ``(B, m)`` point.

        Solves ``m_sat * L(b/b0) = m_at_b`` for ``b0`` by bisection on the
        strictly monotone ``L``. Raises if the point is unreachable.

        ``m/m_sat`` must lie in ``(0, 1)``, and — less obviously — the point must
        satisfy ``m_at_b < m_sat * b / (3 b0)`` for some ``b0``, which reduces to
        the requirement below.
        """
        if not 0.0 < m_at_b_am2 < m_sat_am2:
            raise ValueError(
                f"m_at_b_am2 ({m_at_b_am2!r}) must lie strictly between 0 and "
                f"m_sat_am2 ({m_sat_am2!r}); L(x) is in (0, 1) for x in "
                f"(0, inf)."
            )
        if b_t <= 0.0:
            raise ValueError(f"b_t must be positive, got {b_t!r}")

        target = m_at_b_am2 / m_sat_am2
        # L is strictly increasing in x = b/b0, so decreasing in b0. Bracket b0.
        # Host float64 throughout (see _langevin_np): the fitted b0 must not
        # depend on the ambient JAX default dtype.
        lo, hi = b_t * 1e-6, b_t * 1e6
        f = lambda b0: _langevin_np(b_t / b0) - target
        if f(hi) > 0.0:
            raise ValueError(
                f"m_at_b_am2 / m_sat_am2 = {target:.6g} is too small to reach "
                f"with any positive b0 at b = {b_t!r} T."
            )
        if f(lo) < 0.0:
            raise ValueError(
                f"m_at_b_am2 / m_sat_am2 = {target:.6g} is too large to reach "
                f"with any positive b0 at b = {b_t!r} T."
            )
        for _ in range(200):                     # bisection: ~1e-60 in 200 steps
            mid = 0.5 * (lo + hi)
            if f(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return cls(m_sat_am2, 0.5 * (lo + hi), reference_volume_m3)

    @classmethod
    def landers_capsule(cls) -> "LangevinCurve":
        """The Landers et al. (2025) capsule, pinned to 72 memu at 30 mT.

        THE THREE QUOTED NUMBERS DO NOT FIT ONE LANGEVIN. This is a real tension
        in the source data, reported rather than hidden.

        Landers quotes three things: ``m(30 mT) = 72 memu``, ``m_sat ~ 225 memu``,
        and a low-field slope of ``2.4 memu/mT``. A Langevin has **two** free
        parameters, so three constraints over-determine it. They are also mutually
        inconsistent, and provably so::

            72 memu  ==  2.4 memu/mT x 30 mT   EXACTLY

        i.e. the quoted slope is the **chord** through the 30 mT point, not the
        tangent at the origin. But ``L`` is strictly **concave** on ``x > 0``, so
        for any finite ``b0``::

            m(30 mT)  <  slope_tangent x 30 mT  =  72 memu

        with 72 memu attained only in the ``b0 -> inf`` (pure-linear,
        ``m_sat -> inf``) limit. **No Langevin with a finite saturation can hit
        both 2.4 memu/mT and 72 memu at 30 mT.** The two candidate fits:

        =========================  ========  ===========  ===========  ==========
        fit                        b0 [mT]   m(30 mT)     tangent      m_sat
        =========================  ========  ===========  ===========  ==========
        pin (m_sat, chord slope)   31.25     67.93 memu   2.4          225
                                             **-5.7 %**   (exact)      (exact)
        pin (m_sat, 30 mT point)   29.26     72.0 memu    2.563        225
                                             (exact)      **+6.8 %**   (exact)
        =========================  ========  ===========  ===========  ==========

        **This factory returns the second**, because:

        1. 30 mT *is the operating point* — Landers' eMNS runs there, so the error
           belongs anywhere but at the field the device actually sees;
        2. 72 memu at 30 mT is a **direct VSM reading**, whereas 2.4 memu/mT is
           a derived chord (it reproduces 72/30 to five digits — it is the same
           datum, not a second one);
        3. the ~6 % residual is real physics, not fit error: a single Langevin is
           monodisperse and non-interacting, while the capsule is a dense,
           polydisperse, dipolar-coupled loading whose true knee is broader. The
           tension is the model class being honest about its own idealisation.

        Use :meth:`from_saturation_and_slope` to get the other fit if you want to
        bound the sensitivity — a ~6 % spread on the moment is a ~6 % spread on
        the force, well inside the 43 % the 1.69-vs-1.4 mm ambiguity already
        costs. Both are pinned by MIME-VER-145.
        """
        return cls.from_saturation_and_point(
            m_sat_am2=LANDERS_M_SAT_AM2,
            b_t=LANDERS_B_MEASURED_T,
            m_at_b_am2=LANDERS_M_AT_30MT_AM2,
            reference_volume_m3=sphere_volume_m3(LANDERS_CAPSULE_DIAMETER_M),
        )


class LinearSaturationCurve(MagnetizationCurve):
    """Linear-then-clamped curve, ``m = min(slope * |B|, m_sat)``.

    The crude alternative to :class:`LangevinCurve`, provided so the modelling
    choice can be *compared* rather than asserted, and for callers whose datasheet
    gives only a slope and a saturation.

    **Not recommended near the knee.** It has a kink at ``|B| = m_sat/slope``,
    where ``dm/d|B|`` jumps from ``slope`` to 0 — a discontinuity in the force and
    a zero-measure but real hazard for AD and for any implicit integrator. For the
    Landers capsule the kink sits at 93.75 mT while the device operates at 30 mT,
    so this curve would predict the *linear* 72 memu there and disagree with
    :class:`LangevinCurve` by ~6 % — the very tension documented in
    :meth:`LangevinCurve.landers_capsule`, now silently rounded off. Prefer
    Langevin, or real data.

    Parameters
    ----------
    slope_am2_per_t : float
        Low-field slope ``dm/d|B|`` [A.m^2/T].
    m_sat_am2 : float
        Saturation moment [A.m^2].
    reference_volume_m3 : float
        Volume of the object the curve describes [m^3].
    """

    kind = "linear_saturation"

    def __init__(self, slope_am2_per_t: float, m_sat_am2: float,
                 reference_volume_m3: float):
        if slope_am2_per_t <= 0.0:
            raise ValueError(
                f"slope_am2_per_t must be positive, got {slope_am2_per_t!r}")
        if m_sat_am2 <= 0.0:
            raise ValueError(f"m_sat_am2 must be positive, got {m_sat_am2!r}")
        if reference_volume_m3 <= 0.0:
            raise ValueError(
                f"reference_volume_m3 must be positive, got "
                f"{reference_volume_m3!r}")
        self._slope = float(slope_am2_per_t)
        self._m_sat = float(m_sat_am2)
        self.reference_volume_m3 = float(reference_volume_m3)

    @property
    def m_sat_am2(self) -> float:
        return self._m_sat

    @property
    def knee_t(self) -> float:
        """Field at which the curve kinks into saturation [T]."""
        return self._m_sat / self._slope

    def moment_am2(self, b_mag: jnp.ndarray) -> jnp.ndarray:
        return jnp.minimum(self._slope * jnp.asarray(b_mag), self._m_sat)

    def describe(self) -> str:
        return (
            f"linear_saturation(slope={self._slope:.6g} A.m^2/T, "
            f"m_sat={self._m_sat:.6g} A.m^2, knee={self.knee_t:.6g} T) "
            f"@ V={self.reference_volume_m3:.6g} m^3"
        )


class InterpolatedCurve(MagnetizationCurve):
    """Piecewise-linear interpolation of a **real measured** VSM sweep.

    The preferred curve once an actual ``m(|B|)`` sweep exists: it makes no
    functional-form assumption at all, and so cannot manufacture the ~6 % tension
    that fitting a 2-parameter Langevin to a polydisperse, interacting ensemble
    does (see :meth:`LangevinCurve.landers_capsule`). This is the path for
    "validate against future data".

    **The table must start at the origin, ``(|B|, m) = (0, 0)``**, and this is
    enforced. ``jnp.interp`` clamps *below* the first knot as well as above it,
    so a table starting at, say, 5 mT would return its first moment all the way
    down to zero field — a **remanent moment**, which is exactly what a
    superparamagnet does not have and what this node exists to reject. It would
    also make the force jump-discontinuous at a field null, because
    :meth:`SuperparamagneticResponseNode.update` zeroes the moment at ``|B| = 0``
    exactly. If a real sweep starts away from zero, prepend the origin
    *explicitly* in caller code so that the low-field extrapolation is visible
    where it is made, rather than silently supplied here.

    Beyond the table's last point the moment is **clamped to the last value**
    (``jnp.interp`` end behaviour), which is the physically right extrapolation
    for a saturating curve *provided the table reaches saturation*. If it does
    not, the clamp silently under-predicts — so pass a table that saturates, or
    stay inside it. The two ends are treated differently on purpose: clamping
    high is conservative and physical, whereas clamping low would invent force
    out of nothing.

    Piecewise-linear means the *derivative* is piecewise-constant and
    discontinuous at the knots. The moment is continuous, the force is
    continuous (it depends on ``m``, not ``dm/dB``), but ``d(force)/d(position)``
    is not — relevant only if you differentiate *through* the force.

    Parameters
    ----------
    b_table_t : array (n,)
        Applied field magnitudes [T]. Must be strictly increasing and must start
        at **exactly 0.0** — see the note on the low-end clamp above.
    m_table_am2 : array (n,)
        Measured moments [A.m^2] at ``reference_volume_m3``. Must start at
        **exactly 0.0**: a superparamagnet has no remanence.
    reference_volume_m3 : float
        Volume of the object the curve was measured on [m^3].
    """

    kind = "interpolated"

    def __init__(self, b_table_t, m_table_am2, reference_volume_m3: float):
        b = np.asarray(b_table_t, dtype=float)
        m = np.asarray(m_table_am2, dtype=float)
        if b.ndim != 1 or m.ndim != 1:
            raise ValueError(
                f"b_table_t and m_table_am2 must be 1-D, got shapes "
                f"{b.shape} and {m.shape}")
        if b.shape != m.shape:
            raise ValueError(
                f"b_table_t {b.shape} and m_table_am2 {m.shape} must have the "
                f"same length")
        if b.size < 2:
            raise ValueError("need at least 2 table points to interpolate")
        if np.any(np.diff(b) <= 0.0):
            raise ValueError(
                "b_table_t must be strictly increasing (jnp.interp assumes a "
                "sorted table and returns silent nonsense otherwise)")
        if b[0] < 0.0:
            raise ValueError(
                f"b_table_t must be non-negative — the curve is a function of "
                f"|B|, not B. Got a minimum of {b[0]!r}. A VSM sweep through "
                f"negative field should be folded to |B| before use.")
        # The table MUST be anchored at the origin. jnp.interp clamps below the
        # first knot, so an off-origin table would return m_table[0] all the way
        # down to |B| = 0 — a remanent moment on a node whose defining premise is
        # that it has none, and a force discontinuity at every field null. This
        # is the invariant update() relies on when it takes b_hat = 0 at B = 0.
        if b[0] != 0.0:
            raise ValueError(
                f"b_table_t must start at exactly 0.0 T, got {b[0]!r}. "
                f"jnp.interp CLAMPS below the first knot, so this table would "
                f"report m = {m[0]!r} A.m^2 at every field from 0 up to "
                f"{b[0]!r} T — a remanent moment, which a superparamagnet does "
                f"not have (that is what this node models). It would also make "
                f"the force jump-discontinuous at a field null, since update() "
                f"zeroes the moment at |B| = 0 exactly. If your VSM sweep starts "
                f"away from zero, prepend the origin explicitly — "
                f"b = np.concatenate([[0.0], b]), m = np.concatenate([[0.0], m]) "
                f"— which extrapolates the first segment linearly to the origin. "
                f"Do it in your own code so the extrapolation is visible there.")
        if m[0] != 0.0:
            raise ValueError(
                f"m_table_am2 must start at exactly 0.0 A.m^2 (at |B| = 0), got "
                f"{m[0]!r}. A superparamagnet has NO REMANENCE: m(0) = 0 by "
                f"definition. A non-zero moment at zero field is either a "
                f"ferromagnetic sample — wrong node, see "
                f"PermanentMagnetResponseNode — or a VSM baseline offset that "
                f"must be subtracted before use.")
        if np.any(m < 0.0):
            raise ValueError(
                f"m_table_am2 must be non-negative — it is the moment MAGNITUDE "
                f"at |B|, and m = m(|B|) * B_hat carries the direction. Got a "
                f"minimum of {m.min()!r}. A sweep folded from negative field "
                f"should have its moment folded too.")
        if reference_volume_m3 <= 0.0:
            raise ValueError(
                f"reference_volume_m3 must be positive, got "
                f"{reference_volume_m3!r}")
        self._b = jnp.asarray(b)
        self._m = jnp.asarray(m)
        self._b_np = b
        self._m_np = m
        self.reference_volume_m3 = float(reference_volume_m3)

    @property
    def m_sat_am2(self) -> float:
        """The last tabulated moment — saturation *only if the table reaches it*."""
        return float(self._m_np[-1])

    def moment_am2(self, b_mag: jnp.ndarray) -> jnp.ndarray:
        return jnp.interp(jnp.asarray(b_mag), self._b, self._m)

    def describe(self) -> str:
        return (
            f"interpolated({self._b_np.size} points, "
            f"B in [{self._b_np[0]:.6g}, {self._b_np[-1]:.6g}] T, "
            f"m_max={self._m_np[-1]:.6g} A.m^2) "
            f"@ V={self.reference_volume_m3:.6g} m^3"
        )


class DemagLimitedLinearCurve(MagnetizationCurve):
    """FIRST-PRINCIPLES path — ``chi_eff = chi / (1 + N chi)``. **Gated.**

    ============================================================================
    THIS IS THE SECONDARY PATH. PREFER A MEASURED CURVE.
    ============================================================================

    For callers who have an **intrinsic material susceptibility** ``chi`` (the
    bulk constitutive ``M = chi H_internal``) rather than a measured curve. It
    derives the moment from first principles::

        chi_eff = chi / (1 + N chi)                  demagnetization correction
        m       = chi_eff * (|B_applied| / mu_0) * V     clamped to m_sat

    Every parameter is **required** — there are no defaults, deliberately. You
    cannot reach this path without naming this class and stating an intrinsic
    ``chi``, a demagnetization factor ``N``, and a saturation, each of which is a
    conscious modelling commitment. That is the gate.

    **It still consumes the APPLIED field.** The ``1/(1 + N chi)`` factor *is* the
    demagnetization correction: it converts the intrinsic ``chi`` into the
    effective susceptibility seen by the applied field. Do not also subtract a
    demagnetizing field from the input — that is the double-count this node's
    module docstring exists to prevent, approached from the other side.

    Why you probably want a measured curve instead: this path assumes a **linear**
    material with a **hard clamp** at saturation. A superparamagnet is
    intrinsically Langevin-shaped, so a linear model is only right well below the
    knee, and its clamp introduces the same kink as
    :class:`LinearSaturationCurve`.

    Consistency with the measured path, for the Landers capsule: an intrinsic
    ``chi = 1.98`` gives ``chi_eff = 1.19`` and reproduces the measured 7.2e-5
    A.m^2 at 30 mT. That is an ordinary value for a dense iron-oxide loading and
    sits comfortably below the ``chi -> inf`` ceiling of ``1/N = 3`` — which is
    exactly the ceiling ``MagneticResponseNode`` (MIME-NODE-002) hardwires, and
    why it overshoots by 2.5x. Pinned by MIME-VER-148.

    Parameters
    ----------
    intrinsic_susceptibility : float
        The material's ``chi`` [-], dimensionless SI. **Not** ``chi_eff``: if you
        have ``chi_eff`` already (e.g. from a measurement), you have a measured
        curve — use :class:`LinearSaturationCurve` with
        ``slope = chi_eff * V / mu_0``.
    demag_factor : float
        ``N`` [-], in ``(0, 1)``. **1/3 for a sphere**, size-independent. Along
        the principal axes of an ellipsoid the three factors sum to 1
        (Osborn 1945).
    m_sat_am2 : float
        Saturation moment [A.m^2] at ``reference_volume_m3``.
    reference_volume_m3 : float
        Volume of the magnetic body [m^3].
    """

    kind = "demag_limited_linear"

    def __init__(
        self,
        intrinsic_susceptibility: float,
        demag_factor: float,
        m_sat_am2: float,
        reference_volume_m3: float,
    ):
        chi = float(intrinsic_susceptibility)
        n = float(demag_factor)
        if chi <= 0.0:
            raise ValueError(
                f"intrinsic_susceptibility must be positive, got {chi!r}. This "
                f"is the INTRINSIC chi (M = chi H_internal), not chi_eff.")
        if not 0.0 < n < 1.0:
            raise ValueError(
                f"demag_factor must lie in (0, 1), got {n!r}. It is 1/3 for a "
                f"sphere; along the principal axes of an ellipsoid the three "
                f"factors sum to 1.")
        if m_sat_am2 <= 0.0:
            raise ValueError(f"m_sat_am2 must be positive, got {m_sat_am2!r}")
        if reference_volume_m3 <= 0.0:
            raise ValueError(
                f"reference_volume_m3 must be positive, got "
                f"{reference_volume_m3!r}")
        self._chi = chi
        self._n = n
        self._m_sat = float(m_sat_am2)
        self.reference_volume_m3 = float(reference_volume_m3)

    @property
    def m_sat_am2(self) -> float:
        return self._m_sat

    @property
    def intrinsic_susceptibility(self) -> float:
        return self._chi

    @property
    def demag_factor(self) -> float:
        return self._n

    @property
    def effective_susceptibility(self) -> float:
        """``chi_eff = chi / (1 + N chi)``, the demagnetization-corrected value.

        Bounded above by ``1/N`` (= 3 for a sphere) as ``chi -> inf``. That
        ceiling is what ``MagneticResponseNode`` assumes unconditionally.
        """
        return self._chi / (1.0 + self._n * self._chi)

    @property
    def knee_t(self) -> float:
        """Applied field at which the clamp engages [T]."""
        return (self._m_sat * MU_0
                / (self.effective_susceptibility * self.reference_volume_m3))

    def moment_am2(self, b_mag: jnp.ndarray) -> jnp.ndarray:
        linear = (self.effective_susceptibility * jnp.asarray(b_mag) / MU_0
                  * self.reference_volume_m3)
        return jnp.minimum(linear, self._m_sat)

    def describe(self) -> str:
        return (
            f"demag_limited_linear(chi={self._chi:.6g}, N={self._n:.6g}, "
            f"chi_eff={self.effective_susceptibility:.6g}, "
            f"m_sat={self._m_sat:.6g} A.m^2) "
            f"@ V={self.reference_volume_m3:.6g} m^3"
        )


@stability(StabilityLevel.EXPERIMENTAL)
class SuperparamagneticResponseNode(MimeNode):
    """Superparamagnetic response: gradient force on a field-aligned moment.

    Consumes ``(field_vector, field_gradient)`` — typically from
    ``CoilArrayFieldNode`` — and emits ``(magnetic_force, magnetic_torque)`` for a
    ``RigidBodyNode`` on additive edges::

        m = m_curve(|B|) * (V / V_ref) * B_hat
        F = grad_B @ m
        T = 0                                    (structurally: m is || to B)

    See the module docstring for the physics, the demagnetization
    double-counting warning, the small-body assumption, and the Landers anchor.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep [s].
    curve : MagnetizationCurve
        The constitutive law, ``|B_applied| -> moment``, carrying the reference
        volume it applies to. :class:`LangevinCurve` (or
        :class:`InterpolatedCurve`, given real data) is the primary path;
        :class:`DemagLimitedLinearCurve` is the gated first-principles path.
        Keyword-only and required — there is no default body, because there is no
        such thing as a default magnetic material.
    volume_m3 : float
        Volume of *this* body [m^3]. The curve is rescaled exactly by
        ``volume_m3 / curve.reference_volume_m3`` (valid because a sphere's
        ``N = 1/3`` is size-independent). Defaults to the curve's own reference
        volume, i.e. no rescaling. Use :func:`sphere_volume_m3` to convert from a
        diameter.

    Boundary Inputs
    ---------------
    field_vector : (3,)
        **Applied** B field at the body centre [T], lab frame. NOT demagnetization-
        corrected — see the module docstring.
    field_gradient : (3,3)
        Spatial gradient ``grad_B[i][j] = dB_i/dx_j`` [T/m] (``jax.jacrev``
        convention), lab frame.

    Notes
    -----
    **``magnetization`` is a MOMENT in A.m^2, not a magnetization in A/m.** This
    matches ``PermanentMagnetResponseNode`` (MIME-NODE-008), which is the
    majority convention in the family — but ``MagneticResponseNode``
    (MIME-NODE-002) emits an output *of the same name* in **A/m**, multiplying by
    volume inside its own force/torque. For the Landers capsule the two differ by
    ``1/V = 4.0e8``. The inconsistency is pre-existing in MIME-NODE-002 and is
    not this node's to fix, but anything reading ``magnetization`` generically
    across the response-node family — the effects layer above all — must **not**
    treat the field as interchangeable between them. MIME-VER-148 compensates
    explicitly (``m_legacy = norm(legacy["magnetization"]) * V``). Tracked as
    MIME-ANO-107.

    **There is deliberately no ``orientation`` input.** The two sibling response
    nodes take one; this node does not, because a field-aligned isotropic moment
    is *structurally* orientation-independent — ``m = m(|B|) B_hat`` contains no
    body-frame quantity, so there is no frame to rotate into and nothing for a
    quaternion to do. Declaring the input and ignoring it would advertise a
    dependence that does not exist, and this repo has already been bitten by a
    dead boundary input.

    This is safe against the existing wiring: ``_MagneticEffect.build()``
    unconditionally adds a ``body.orientation -> response.orientation`` edge, and
    ``GraphManager.validate()`` skips edges whose target field is not in the
    target's ``boundary_input_spec()`` (``if e.target_field not in bi_spec:
    continue``). So the hardwired edge is a silent no-op rather than an error —
    verified by ``test_orientation_edge_is_a_harmless_no_op``. An anisotropic bead
    *would* need the input back, but it would need a different constitutive law
    too, and so a different node.
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-009",
        algorithm_version="1.0.1",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "Superparamagnetic gradient-pull response: field-aligned moment from "
            "a measured magnetization curve m(|B|); force F = grad_B @ m; torque "
            "identically zero."
        ),
        governing_equations=(
            r"m = m_curve(|B|) * (V/V_ref) * B_hat; "
            r"T = m x B = 0 (structural — m || B); "
            r"F = (m . grad)B = grad_B @ m; "
            r"m_curve = m_sat * L(|B|/b0), L(x) = coth(x) - 1/x  [Langevin]; "
            r"chi_eff = chi / (1 + N chi)  [gated first-principles path]"
        ),
        discretization=(
            "Analytical — no discretisation. The Langevin function uses a 4-term "
            "Maclaurin series below |x| = 0.1 (where coth(x) - 1/x cancels "
            "catastrophically) and the direct form above, selected with the "
            "double-where idiom so reverse-mode AD cannot propagate a NaN from "
            "the untaken branch. The field magnitude uses a double-where safe "
            "norm so |B| = 0 yields 0 rather than NaN."
        ),
        assumptions=(
            "Moment is field-aligned: m || B, no easy axis, no remanence, no "
            "hysteresis. True for a superparamagnet above its blocking "
            "temperature; false for a ferromagnetic or shape-anisotropic body",
            "SMALL BODY: the field at the body CENTRE represents the field across "
            "the whole body. This is what lets a VSM curve (measured in a UNIFORM "
            "field) transfer to a gradient-pulling context at all. Quantified by "
            "delta|B|/|B| ~ |grad|B||*d/|B|; for the Landers capsule (d = 1.69 mm, "
            "|B| = 30 mT) that is 1.4% at 0.25 T/m (Navion single-gradient), 2.8% "
            "at 0.5 T/m, and 5.6% at 1 T/m (double-gradient — the mode used to "
            "pull against flow, so the anchor's most demanding configuration sits "
            "at the EDGE of this assumption, not comfortably inside it)",
            "MEASURED PATH: the input field is the APPLIED field and the curve "
            "already contains the body's self-demagnetization (a VSM measures the "
            "real object). No demag correction is applied and none must be — "
            "correcting again double-counts",
            "Volume scaling m ~ V is exact for spheres (N = 1/3 is "
            "size-independent) at fixed material and shape",
            "Quasi-static: the moment follows |B| instantaneously. No Neel or "
            "Brownian relaxation lag, so this is wrong above ~1/tau_N",
            "The body does not perturb the applied field (dilute limit); no "
            "bead-bead dipolar interaction",
        ),
        limitations=(
            "TORQUE IS IDENTICALLY ZERO. This node cannot reorient a body. If it "
            "is a body's only magnetic effect, that body's orientation is "
            "unactuated. A real capsule has some shape anisotropy and a small "
            "real aligning torque; this models it as exactly zero",
            "Small-body assumption is NOT checked at runtime (the check is not "
            "JAX-traceable). It degrades linearly in body size and in |grad B|, "
            "and diverges as |B| -> 0 — a field null breaks it for ANY body size. "
            "Call small_body_field_variation_ratio() yourself",
            "No relaxation dynamics: the Landers nanocubes are 18.4 nm, whose "
            "Neel time is nanoseconds against millisecond timesteps, so the "
            "quasi-static limit is safe THERE — but it is not a property of the "
            "node, and larger or blocked particles violate it",
            "A single Langevin is monodisperse and non-interacting; a real "
            "polydisperse, dipolar-coupled loading has a broader knee. This shows "
            "up as a ~6% inconsistency in the Landers numbers themselves — see "
            "LangevinCurve.landers_capsule. Prefer InterpolatedCurve with real data",
            "No temperature dependence: b0 = k_B T / mu_particle is fixed at "
            "construction, so a curve measured at room temperature is used "
            "unchanged at 37 C (~5% error in b0 via T alone)",
            "The gradient is consumed as given and is NOT symmetrised or checked "
            "for Maxwell consistency. F = grad_B @ m is right for a physical "
            "(curl-free) field; a non-symmetric grad_B silently yields a "
            "non-conservative force",
        ),
        validated_regimes=(
            # NOTE: these four are NOT independently attainable. small_body_ratio
            # is DERIVED from the other three (~ |grad B| * d / |B|), so it
            # couples them: its ceiling is what the (|grad B|, diameter) corners
            # are allowed to reach at the 30 mT operating point, not a free
            # dimension. The ceiling below is set to admit the anchor device at
            # the anchor's own maximum gradient — see the note on it.
            ValidatedRegime("|B|", 0.0, 0.1, "T",
                            "eMNS operating band; Landers/OctoMag ~30 mT. VERIFIED "
                            "at the single measured point 30 mT (MIME-VER-144) "
                            "plus the analytic high-field asymptote; the rest of "
                            "the band is the curve's declared domain of use, not "
                            "a set of validated data points — there is only one "
                            "VSM datum in the source"),
            ValidatedRegime("|grad B|", 0.0, 1.0, "T/m",
                            "Navion ~0.25 T/m single, ~1 T/m double "
                            "(Landers 2025). The double-gradient mode is the one "
                            "used to pull against flow, and it is what forces the "
                            "small_body_ratio ceiling up to 0.07"),
            ValidatedRegime("body_diameter", 0.0, 2.0e-3, "m",
                            "Landers capsule 1.4-1.69 mm. Upper bound is the "
                            "small-body assumption, not the algebra: at 1 T/m "
                            "and 30 mT a 2 mm body already varies 6.7% in |B|"),
            ValidatedRegime("small_body_ratio", 0.0, 0.07, "-",
                            "delta|B|/|B| across the body — DERIVED, not "
                            "independent: it is ~|grad B|*d/|B| and so is pinned "
                            "by the three regimes above. The ceiling is 0.07 "
                            "because that is what the other three corners "
                            "actually reach: 2 mm at 1 T/m and 30 mT gives 0.067, "
                            "and the 1.69 mm anchor at its own double-gradient "
                            "setting gives 0.056. It is 0.014 at 0.25 T/m and "
                            "0.028 at 0.5 T/m — the anchor is comfortable in "
                            "single-gradient mode and AT THE EDGE in double. "
                            "Beyond ~0.2 the centre field is not representative "
                            "and the moment is a fiction"),
        ),
        references=(
            Reference("Landers2025",
                      "Clinically ready magnetic microrobots for targeted "
                      "therapies — the anchor: gelatin/zinc-ferrite capsule, VSM "
                      "72 memu at 30 mT, gradient-pulled against flow"),
            Reference("Bean1959",
                      "Superparamagnetism — the Langevin form for an ensemble of "
                      "thermally-free single-domain moments"),
            Reference("Osborn1945",
                      "Demagnetizing factors of the general ellipsoid — N = 1/3 "
                      "for a sphere, size-independent, which is what makes the "
                      "volume scaling exact"),
            Reference("Abbott2020",
                      "Magnetic methods in robotics — F = (m.grad)B for an "
                      "induced moment, and the one-half that belongs in the "
                      "energy but not the force"),
        ),
        hazard_hints=(
            "DO NOT APPLY A DEMAGNETIZATION CORRECTION TO THE FIELD ON THE "
            "MEASURED PATH. A VSM measures the real object in a known APPLIED "
            "field, so the object's self-demagnetization is already inside the "
            "measured curve: m_measured(|B_applied|) IS chi_eff. The field "
            "CoilArrayFieldNode emits is the applied field. Correcting again "
            "double-counts and UNDER-predicts the moment and hence the force. "
            "This is the single most likely error in this node.",
            "TORQUE IS EXACTLY ZERO BY CONSTRUCTION, not by accident and not by "
            "numerical smallness. A field-aligned moment has m x B = 0 "
            "identically. This node therefore cannot rotate a body: if it is the "
            "only magnetic effect on a body, that body's orientation is "
            "unactuated and will drift under drag and contact alone. Do not "
            "reach for this node to model field steering — use "
            "PermanentMagnetResponseNode or MagneticResponseNode, whose "
            "anisotropy is the point.",
            "THE SMALL-BODY ASSUMPTION IS NOT CHECKED AT RUNTIME. The curve is "
            "evaluated at the body-centre field and applied to the whole body, "
            "which is only valid while delta|B|/|B| ~ |grad|B||*d/|B| is small. "
            "For the Landers capsule at 30 mT that is 1.4% at 0.25 T/m but 5.6% "
            "at the 1 T/m double-gradient setting it uses to pull against flow — "
            "quote the number for YOUR gradient, not the 2.8% headline. It "
            "degrades LINEARLY in body diameter and in |grad B| and DIVERGES as "
            "|B| -> 0 — a field null breaks it for any body size, however small. "
            "Nothing will warn you: the moment will simply be wrong. Call "
            "small_body_field_variation_ratio() for your own regime; it returns "
            "inf at a null.",
            "The three Landers numbers (72 memu at 30 mT, 2.4 memu/mT, 225 memu "
            "saturation) are MUTUALLY INCONSISTENT with any single Langevin — "
            "72 = 2.4 x 30 exactly, so the quoted slope is a chord, and a "
            "concave L cannot reach its own chord at finite saturation. "
            "landers_capsule() pins the 30 mT operating point and accepts a "
            "+6.8% error in the tangent slope. If you need the other end of that "
            "trade, use from_saturation_and_slope() and expect ~-5.7% on the "
            "moment at 30 mT.",
            "THE LANDERS CAPSULE SIZE IS AMBIGUOUS IN THE SOURCE and the paper "
            "never reconciles it: the main text says 1.69 mm, its own CFD says "
            "1.4 mm. That is a 0.57x volume ratio — a 43% swing in the moment "
            "and therefore in the force, which dwarfs every other error here. "
            "Sweep both (LANDERS_CAPSULE_DIAMETER_M, LANDERS_CFD_DIAMETER_M); do "
            "not silently pick one.",
            "Volume scaling is exact for a SPHERE because N = 1/3 does not "
            "depend on size. It is NOT a licence to change SHAPE: rescaling a "
            "curve measured on a sphere onto a rod or a disc is wrong, because N "
            "depends on aspect ratio. Only isotropic rescaling is free.",
            "grad_B is consumed as given: it is NOT symmetrised, and Maxwell "
            "consistency is NOT enforced. A physical field is curl-free and so "
            "has a symmetric gradient (CoilArrayFieldNode guarantees this "
            "structurally); a hand-built or data-fitted gradient may not, and "
            "will then produce a non-conservative force that quietly does net "
            "work around a closed loop.",
        ),
        implementation_map={
            "m = m_curve(|B|) * (V/V_ref) * B_hat": (
                "mime.nodes.robot.superparamagnetic_response."
                "SuperparamagneticResponseNode.update"
            ),
            "F = (m . grad)B = grad_B @ m": (
                "mime.nodes.robot.superparamagnetic_response."
                "SuperparamagneticResponseNode.update"
            ),
            "T = m x B = 0 (structural)": (
                "mime.nodes.robot.superparamagnetic_response."
                "SuperparamagneticResponseNode.update"
            ),
            "L(x) = coth(x) - 1/x (series below |x|=0.1)":
                "mime.nodes.robot.superparamagnetic_response.langevin",
            "m_curve = m_sat * L(|B|/b0)": (
                "mime.nodes.robot.superparamagnetic_response."
                "LangevinCurve.moment_am2"
            ),
            "b0 from (m_sat, measured point) by bisection": (
                "mime.nodes.robot.superparamagnetic_response."
                "LangevinCurve.from_saturation_and_point"
            ),
            "chi_eff = chi / (1 + N chi) [gated]": (
                "mime.nodes.robot.superparamagnetic_response."
                "DemagLimitedLinearCurve.effective_susceptibility"
            ),
            "delta|B|/|B| = |grad_B^T B_hat| d / |B|": (
                "mime.nodes.robot.superparamagnetic_response."
                "small_body_field_variation_ratio"
            ),
            "|v| with 0 gradient at v=0 (double-where)":
                "mime.nodes.robot.superparamagnetic_response._safe_norm",
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.ROBOT_BODY,
        biocompatibility=BiocompatibilityMeta(
            materials=(
                "gelatin",
                "zinc-substituted iron oxide nanocubes (18.4 nm)",
                "tantalum",
            ),
            iso_10993_class=BiocompatibilityClass.NOT_ASSESSED,
            biocompatibility_hazard_hints=(
                "Materials describe the Landers et al. (2025) capsule — gelatin "
                "matrix, zinc-substituted iron-oxide nanocubes, radiopaque "
                "tantalum. NOT assessed here: this is a technical descriptor, "
                "not a safety claim, and the manufacturer must perform their own "
                "ISO 10993 evaluation.",
                "The node's physics is material-agnostic — it takes whatever "
                "magnetization curve it is given. These materials describe the "
                "DEFAULT anchor device only, and are wrong for any other body. "
                "Update them alongside the curve.",
                "Zinc-substituted iron-oxide nanoparticles are not the same "
                "material as the magnetite whose biocompatibility literature is "
                "often cited by analogy; zinc leaching is a distinct question "
                "and is not addressed by a magnetite assessment.",
            ),
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        *,
        curve: MagnetizationCurve,
        volume_m3: Optional[float] = None,
        **kwargs,
    ):
        if not isinstance(curve, MagnetizationCurve):
            raise TypeError(
                f"curve must be a MagnetizationCurve, got "
                f"{type(curve).__name__}. Use LangevinCurve.landers_capsule() "
                f"for the Landers anchor, InterpolatedCurve for a measured VSM "
                f"sweep, or DemagLimitedLinearCurve for the gated "
                f"first-principles path."
            )
        if volume_m3 is None:
            volume_m3 = curve.reference_volume_m3
        volume_m3 = float(volume_m3)
        if volume_m3 <= 0.0:
            raise ValueError(f"volume_m3 must be positive, got {volume_m3!r}")

        super().__init__(
            name, timestep,
            volume_m3=volume_m3,
            reference_volume_m3=float(curve.reference_volume_m3),
            volume_scale=volume_m3 / float(curve.reference_volume_m3),
            curve_kind=curve.kind,
            curve_description=curve.describe(),
            **kwargs,
        )
        # The curve holds jnp arrays (InterpolatedCurve) and is a Python object;
        # it lives outside params, which feed graph.json serialisation. Its
        # identity is recorded there by curve_kind / curve_description — the same
        # split CoilArrayFieldNode uses for its calibration.
        self._curve = curve

    @property
    def curve(self) -> MagnetizationCurve:
        return self._curve

    @property
    def volume_scale(self) -> float:
        """``V / V_ref`` — the exact sphere volume-scaling factor."""
        return float(self.params["volume_scale"])

    @property
    def diameter_m(self) -> float:
        """Equivalent-sphere diameter of this body [m]."""
        return sphere_equivalent_diameter_m(self.params["volume_m3"])

    def moment_am2(self, b_mag) -> jnp.ndarray:
        """Moment magnitude [A.m^2] of *this* body at applied field ``b_mag`` [T].

        The curve rescaled to this body's volume. Public because the moment is
        the quantity to compare against a VSM datasheet.
        """
        return self._curve.moment_am2(jnp.asarray(b_mag)) * self.volume_scale

    def m_sat_am2(self) -> float:
        """This body's saturation moment [A.m^2]."""
        return self._curve.m_sat_am2 * self.volume_scale

    def initial_susceptibility_am2_per_t(self) -> float:
        """``dm/d|B|`` as ``|B| -> 0`` [A.m^2/T] for *this* body.

        The curve's tangent rescaled to ``volume_m3``. Present so that the node's
        public surface is uniform in what it scales: :meth:`moment_am2` and
        :meth:`m_sat_am2` are volume-scaled, so this must be too. Reaching
        through to ``node.curve.initial_susceptibility_am2_per_t()`` instead
        returns the slope at the curve's *reference* volume, which is a different
        body whenever ``volume_m3`` was overridden — the exact trap the 1.4 mm
        Landers arm walks into.
        """
        return self._curve.initial_susceptibility_am2_per_t() * self.volume_scale

    def small_body_ratio(self, field_vector, field_gradient) -> jnp.ndarray:
        """``delta|B|/|B|`` across *this* body — see the module docstring.

        Advisory: **not** called during ``update()``. Below ~0.05 the measured
        curve transfers. The Landers capsule sits at 0.014 in single-gradient
        mode, 0.028 at 0.5 T/m, and 0.056 at the 1 T/m double-gradient setting —
        i.e. the anchor crosses the rule-of-thumb ceiling in the very mode it
        uses to pull against flow. Returns ``inf`` at a field null.
        """
        return small_body_field_variation_ratio(
            field_vector, field_gradient, self.diameter_m,
        )

    def initial_state(self) -> dict:
        return {
            "magnetization": jnp.zeros(3),     # A.m^2, lab frame (a MOMENT)
            "magnetic_torque": jnp.zeros(3),   # N.m, lab frame — always zero
            "magnetic_force": jnp.zeros(3),    # N, lab frame
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        # NOTE: no `orientation`. A field-aligned isotropic moment has no
        # body-frame quantity to rotate, so the input would be dead. See the
        # class docstring.
        return {
            "field_vector": BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description="APPLIED B field at the body centre [T] (NOT "
                            "demagnetization-corrected)",
            ),
            "field_gradient": BoundaryInputSpec(
                shape=(3, 3), default=jnp.zeros((3, 3)),
                description="Spatial gradient dB_i/dx_j [T/m]",
            ),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        B = jnp.asarray(boundary_inputs.get("field_vector", jnp.zeros(3)))
        grad_B = jnp.asarray(
            boundary_inputs.get("field_gradient", jnp.zeros((3, 3))))

        # |B| with a finite (zero) gradient at B = 0 rather than NaN.
        b_mag = _safe_norm(B)
        safe_mag = jnp.where(b_mag > 0.0, b_mag, 1.0)
        b_hat = B / safe_mag

        # The curve consumes the APPLIED field. No demagnetization correction:
        # a VSM measured the real object, so chi_eff is already in the curve.
        # Volume scaling is exact — a sphere's N = 1/3 is size-independent.
        m_mag = self._curve.moment_am2(b_mag) * self.params["volume_scale"]

        # Field-aligned moment. Every curve is ANCHORED AT THE ORIGIN, m(0) = 0:
        # Langevin analytically (L(0) = 0), LinearSaturationCurve and
        # DemagLimitedLinearCurve because they are linear through it, and
        # InterpolatedCurve because __init__ REQUIRES the table to start at
        # (0, 0) — precisely so that jnp.interp's low-end clamp cannot invent a
        # remanent moment here. That invariant is enforced, not assumed
        # (test_every_curve_is_zero_at_zero_field), and it is what makes the
        # b_hat = 0 taken by the guard above at B = 0 the correct LIMIT rather
        # than a discontinuity papered over. A curve with m(0) != 0 would make
        # the force jump across a field null; the constructors reject one.
        m_vec = m_mag * b_hat

        # F_i = m_j dB_i/dx_j = grad_B @ m. Correct for an INDUCED moment: the
        # one-half of U = -1/2 m.B belongs to the energy, not the force.
        force = grad_B @ m_vec

        # T = m x B == 0 structurally (m || B). Emitted as exact zeros rather
        # than the ~1e-20 N.m of roundoff that evaluating the cross product
        # would produce — injecting numerical noise into the body integrator
        # would be strictly worse than injecting nothing. The moment really is
        # parallel: MIME-VER-147 asserts cross(magnetization, B) is at roundoff.
        torque = jnp.zeros(3, dtype=force.dtype)

        return {
            "magnetization": m_vec,
            "magnetic_torque": torque,
            "magnetic_force": force,
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "magnetic_torque": state["magnetic_torque"],
            "magnetic_force": state["magnetic_force"],
        }
