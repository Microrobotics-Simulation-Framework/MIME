"""A0f's statics at A (register section 5, review 1's fixes): the swim speed U(s) every 0.25 cm over the paper's
measured window (+-2 cm about the magnet), from static readings at the floored pose that include the fixed magnet's
force and torque at each position; the time-weighted slope 2L / int ds/U(s) per direction and its two-direction mean
against A; the stall rule (U(s) <= 0 anywhere: the stall and where are the result); and the planted incline's control.
Local CPU. No measured speed enters. Predictions: ``predictions_a0f_stage1.json``, committed before any run.

**The resistance.** The BEM's 6 x 6 resistance ``R(phi)`` at the floored pose at the eight phases of the half turn
(``a0f_solves.floored``, on Modal) serves every position, since only the field changes with ``s``. Between the phases
it is the trigonometric interpolant of the eight (the screw's two starts make ``R`` periodic in ``pi``); every reading's
``R`` is checked positive definite on a dense grid of phases that covers every phase a reading uses.

**The field.** The magnet is a point dipole (MIME's provider, ``mime.nodes.actuation._point_dipole.point_dipole``) of
18.89 A m^2, 150 mm above the tube's axis over ``s = 0``, its moment turning at 10 Hz in the plane across the tube. The
body's moment, 1.68e-3 A m^2 at its centre, lies across the body's axis at the body's phase ``phi``. The force is
``G^T m`` (``Magnetic.Force``'s law) and the torque ``m x B`` (``Magnetic.Torque``'s).

**The static reading with the force and torque.** The pose is held (on the floor, its axis along the tube); the
translation along the tube ``U`` and the turning about it ``Omega = dphi/dt`` are free; the body receives
``-R [U; Omega]`` and the magnet's wrench, so the force along the tube and the torque about it balance:

    [R_xx  R_xO] [U    ]   [F_x (+ W sin(incline))]
    [R_Ox  R_OO] [Omega] = [T_x                    ]

with ``F_x`` and ``T_x`` at the body's phase. That is an equation for ``phi(t)`` (the lock, its lag behind the field
set by the torque); its periodic solution, after the transient, gives ``U(t)`` and its mean over a turn of the magnet
is ``U(s)``. The torque across the axis is reacted by the held pose and reported. A variant with **zero lag** (the
moment along the field's direction across the tube, ``Omega`` the field's own rate, the force row alone; C1's immediate
check and review 1's convention) is reported beside it. The incline enters as an axial force ``W sin(theta)``.

**A** is the same reading in script A's uniform field (1.2 mT, no gradient): the eight phases' mean axial speed.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python scripts/sim2real/a0f_statics.py --resistance FLOORED.json --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import bem  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
import script_a as A  # noqa: E402

FREQUENCY_HZ = 10.0
OMEGA = 2.0 * math.pi * FREQUENCY_HZ
#: The actuator (read, the paper pp. 29201-29202): de Jongh's 35 x 20 mm N45 magnet, 15 cm above the vessel's centre,
#: its axis along the vessel, turning about it.
ACTUATOR_MOMENT = 18.89
ACTUATOR_HEIGHT_M = 0.150
BODY_MOMENT = float(A.MOMENT.value[1])
GRAVITY = 9.81
WATER_DENSITY = float(A.WATER.value)
WEIGHT_N = (float(A.MASS.value) - WATER_DENSITY * float(A.VOLUME.value)) * GRAVITY
#: The spin's rotational inertia (the reference's I_eff), for the size of what a static reading leaves out.
SPIN_INERTIA = float(A.INERTIA.value[0])
#: FL-9's length, for the pitching torque's scale W L / 2 (review 1's figure).
BODY_LENGTH_M = 7.47e-3
UNIFORM_FIELD_T = 1.2e-3
#: The register's frozen window and static grid (section 2): +-2 cm, every 0.25 cm.
HALF_WINDOW_M = 0.02
GRID_M = 0.0025
STATIC_GRID = tuple(round(-HALF_WINDOW_M + GRID_M * k, 6) for k in range(17))
#: A dense grid for the stall's position and the quadrature's diagnostic (not the reading of record).
DENSE_STEP_M = 1e-4
INCLINE_DEG = 0.05
TURN_POINTS = 2048
TURNS = 2
VARIANTS = ("torque", "zero_lag")


# ---- the resistance between the eight phases ------------------------------------------------------------------------
class Interpolant:
    """``R(phi)``, periodic in ``pi``, from its values at ``phi_k = pi k / 8``: the trigonometric interpolant in
    ``2 phi`` (the real part of the inverse DFT, which reproduces the eight exactly)."""

    def __init__(self, matrices: np.ndarray) -> None:
        self.samples = np.asarray(matrices, dtype=np.float64)
        self.n = self.samples.shape[0]
        self.coefficients = np.fft.fft(self.samples, axis=0) / self.n
        self.harmonics = np.fft.fftfreq(self.n, d=1.0 / self.n)

    def __call__(self, phi: np.ndarray, rows: Any = slice(None), columns: Any = slice(None)) -> np.ndarray:
        theta = 2.0 * np.asarray(phi, dtype=np.float64)
        c = self.coefficients[:, rows][:, :, columns] if not isinstance(rows, slice) or not isinstance(
            columns, slice) else self.coefficients[:, rows, columns]
        phase = np.exp(1j * np.multiply.outer(theta, self.harmonics))
        return np.real(np.tensordot(phase, c, axes=(-1, 0)))

    def phase_mean(self) -> np.ndarray:
        return self.samples.mean(axis=0)


def positive_definite_everywhere(interp: Interpolant, points: int = 2048) -> dict[str, Any]:
    """The interpolant's smallest non-dimensional eigenvalue over a dense grid of the half turn: every reading's
    ``R`` is the interpolant at its phase modulo ``pi``."""
    phi = np.linspace(0.0, math.pi, points, endpoint=False)
    least = np.asarray([bem.smallest_eigenvalue(r) for r in interp(phi)])
    return {"points": points, "least": float(least.min()), "at_phase": float(phi[int(np.argmin(least))]),
            "positive_definite": bool(least.min() > 0.0)}


# ---- the field -------------------------------------------------------------------------------------------------------
def actuator_moment(t: np.ndarray, sense: int) -> np.ndarray:
    """The magnet's moment at times ``t``; ``sense`` +1 turns the field at the body positively about +x (the body swims
    +x), -1 the other way. Below a dipole the field turns against the dipole's own sense, hence the minus sign (checked
    on the field's advance in ``field_table``)."""
    return ACTUATOR_MOMENT * np.stack([np.zeros_like(t), np.cos(OMEGA * t), -sense * np.sin(OMEGA * t)], axis=-1)


def dipole_at(s: float, t: np.ndarray, sense: int, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """``B`` (T) and ``G_ij = dB_i/dx_j`` (T/m) at the body's centre ``(s, 0, -offset)``, from MIME's provider."""
    import jax
    import jax.numpy as jnp

    from mime.nodes.actuation._point_dipole import point_dipole

    centre = jnp.asarray([0.0, 0.0, ACTUATOR_HEIGHT_M])
    point = jnp.asarray([s, 0.0, -offset])
    field, gradient = jax.vmap(lambda m: point_dipole(m, centre, point))(jnp.asarray(actuator_moment(t, sense)))
    return np.asarray(field), np.asarray(gradient)


def uniform_at(t: np.ndarray, sense: int) -> tuple[np.ndarray, np.ndarray]:
    """Script A's uniform field: 1.2 mT turning about +x at 10 Hz, no gradient."""
    field = UNIFORM_FIELD_T * np.stack([np.zeros_like(t), np.cos(OMEGA * t), sense * np.sin(OMEGA * t)], axis=-1)
    return field, np.zeros(t.shape + (3, 3))


def body_moment(phi: np.ndarray) -> np.ndarray:
    return BODY_MOMENT * np.stack([np.zeros_like(phi), np.cos(phi), np.sin(phi)], axis=-1)


def wrench(phi: np.ndarray, field: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The magnet's force ``G^T m`` and torque ``m x B`` on the body at phase ``phi``."""
    m = body_moment(phi)
    return np.einsum("...ij,...i->...j", gradient, m), np.cross(m, field)


def field_table(s_values: np.ndarray, sense: int, offset: float, points: int, uniform: bool
                ) -> tuple[np.ndarray, np.ndarray]:
    """The field and its gradient at every position over one turn at ``2 * points`` half steps, ``(n_s, 2 points,
    ...)``; the field's direction across the tube must advance by ``sense * 2 pi`` over the turn."""
    t = np.arange(2 * points) / (2 * points * FREQUENCY_HZ)
    if uniform:
        field, gradient = uniform_at(t, sense)
        fields = np.broadcast_to(field, (len(s_values),) + field.shape).copy()
        gradients = np.broadcast_to(gradient, (len(s_values),) + gradient.shape).copy()
    else:
        pairs = [dipole_at(float(s), t, sense, offset) for s in s_values]
        fields = np.stack([p[0] for p in pairs])
        gradients = np.stack([p[1] for p in pairs])
    raw = np.arctan2(fields[..., 2], fields[..., 1])
    # the turn closed on its first sample: the unwrapped angle's last minus its first is the whole turn's advance
    closed = np.unwrap(np.concatenate([raw, raw[:, :1]], axis=1), axis=1)
    advance = closed[:, -1] - closed[:, 0]
    if not np.allclose(advance, sense * 2 * math.pi, atol=1e-9):
        raise AssertionError(f"the field does not turn once in the sense {sense}: {advance[:3]}")
    return fields, gradients


# ---- the reading over a turn ----------------------------------------------------------------------------------------
def readings(interp: Interpolant, s_values: np.ndarray, sense: int, variant: str, offset: float,
             extra_force: float = 0.0, uniform: bool = False, points: int = TURN_POINTS) -> list[dict[str, Any]]:
    """The reading at every position over one turn of the magnet (after ``TURNS - 1`` turns of transient for the
    torque variant): the mean swim speed ``U`` and what it reports."""
    s_values = np.asarray(s_values, dtype=np.float64)
    fields, gradients = field_table(s_values, sense, offset, points, uniform)
    period = 1.0 / FREQUENCY_HZ
    dt = period / points
    rows, cols = [0, 3], [0, 3]
    if variant == "zero_lag":
        field, gradient = fields[:, ::2], gradients[:, ::2]
        phi = np.unwrap(np.arctan2(field[..., 2], field[..., 1]), axis=1)
        t = np.arange(points) * dt
        drift = phi - sense * OMEGA * t
        frequencies = np.fft.rfftfreq(points, d=dt)
        rate = sense * OMEGA + np.fft.irfft(1j * 2 * np.pi * frequencies * np.fft.rfft(drift, axis=1), n=points, axis=1)
        r = interp(phi, rows, cols)
        force, torque = wrench(phi, field, gradient)
        u = (force[..., 0] + extra_force - r[..., 0, 1] * rate) / r[..., 0, 0]
        omega, lag = rate, np.zeros_like(rate)
        extra: dict[str, np.ndarray] = {}
    elif variant == "torque":
        def rate_of(phi: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            r = interp(phi, rows, cols)
            m = body_moment(phi)
            g, b = gradients[:, k % (2 * points)], fields[:, k % (2 * points)]
            fx = np.einsum("si,si->s", g[:, :, 0], m) + extra_force
            tx = m[:, 1] * b[:, 2] - m[:, 2] * b[:, 1]
            det = r[:, 0, 0] * r[:, 1, 1] - r[:, 0, 1] * r[:, 1, 0]
            return (r[:, 1, 1] * fx - r[:, 0, 1] * tx) / det, (r[:, 0, 0] * tx - r[:, 1, 0] * fx) / det

        phi = np.arctan2(fields[:, 0, 2], fields[:, 0, 1])
        start = phi.copy()
        record_u, record_omega, record_phi = [], [], []
        for step in range(TURNS * points):
            k = 2 * step
            if step == (TURNS - 1) * points:
                start = phi.copy()
            u1, o1 = rate_of(phi, k)
            if step >= (TURNS - 1) * points:
                record_u.append(u1)
                record_omega.append(o1)
                record_phi.append(phi.copy())
            _, o2 = rate_of(phi + 0.5 * dt * o1, k + 1)
            _, o3 = rate_of(phi + 0.5 * dt * o2, k + 1)
            _, o4 = rate_of(phi + dt * o3, k + 2)
            phi = phi + dt * (o1 + 2 * o2 + 2 * o3 + o4) / 6.0
        u = np.stack(record_u, axis=1)
        omega = np.stack(record_omega, axis=1)
        phi_turn = np.stack(record_phi, axis=1)
        field, gradient = fields[:, ::2], gradients[:, ::2]
        b_phi = np.arctan2(field[..., 2], field[..., 1])
        lag = np.angle(np.exp(1j * sense * (b_phi - phi_turn)))
        force, torque = wrench(phi_turn, field, gradient)
        extra = {"advance": phi - start}
    else:
        raise ValueError(variant)
    b_yz = np.hypot(field[..., 1], field[..., 2])
    found = []
    for i, s in enumerate(s_values):
        row = {"s_m": float(s), "U_m_s": float(np.mean(u[i])), "U_min_m_s": float(u[i].min()),
               "U_max_m_s": float(u[i].max()), "lag_mean_rad": float(np.mean(lag[i])),
               "lag_max_abs_rad": float(np.max(np.abs(lag[i]))),
               "Fx_mean_N": float(np.mean(force[i, :, 0])), "Fy_mean_N": float(np.mean(force[i, :, 1])),
               "Fz_mean_N": float(np.mean(force[i, :, 2])), "Fz_max_N": float(np.max(force[i, :, 2])),
               "Ty_mean_Nm": float(np.mean(torque[i, :, 1])), "Tz_mean_Nm": float(np.mean(torque[i, :, 2])),
               "B_yz_min_T": float(b_yz[i].min()), "B_yz_max_T": float(b_yz[i].max()),
               "Omega_mean_rad_s": float(np.mean(omega[i]))}
        row["u_along_pass_m_s"] = sense * row["U_m_s"]
        if variant == "torque":
            row["advance_over_turn_rad"] = float(extra["advance"][i])
            row["locked"] = bool(abs(extra["advance"][i] - sense * 2 * math.pi) < 1e-6)
            accel = np.gradient(omega[i], dt, edge_order=2)
            row["inertia_torque_over_field_torque_max"] = float(np.max(np.abs(SPIN_INERTIA * accel)) /
                                                              np.min(BODY_MOMENT * b_yz[i]))
        found.append(row)
    return found


def pass_metric(s: np.ndarray, u: np.ndarray, sense: int) -> dict[str, Any]:
    """The time-weighted slope 2L / int ds/u over the window (trapezoid rule on the given grid) for a pass in
    direction ``sense``, or its stall: the first position along the pass where ``u <= 0``, linearly between grid
    points."""
    order = np.argsort(s) if sense > 0 else np.argsort(-s)
    s_pass, u_pass = s[order], u[order]
    bad = np.nonzero(u_pass <= 0.0)[0]
    if bad.size:
        i = int(bad[0])
        where = float(s_pass[0]) if i == 0 else float(
            s_pass[i - 1] + (s_pass[i] - s_pass[i - 1]) * u_pass[i - 1] / (u_pass[i - 1] - u_pass[i]))
        return {"stalls": True, "stall_s_m": where, "slope_m_s": None}
    time_s = abs(np.trapezoid(1.0 / u_pass, s_pass))
    return {"stalls": False, "stall_s_m": None, "slope_m_s": float(2 * HALF_WINDOW_M / time_s), "time_s": float(time_s)}


def analyse(matrices: np.ndarray, offset: float, *, dense: bool = True) -> dict[str, Any]:
    interp = Interpolant(matrices)
    incline_force = WEIGHT_N * math.sin(math.radians(INCLINE_DEG))
    grid = np.asarray(STATIC_GRID)
    dense_grid = np.round(np.arange(-HALF_WINDOW_M, HALF_WINDOW_M + DENSE_STEP_M / 2, DENSE_STEP_M), 9)
    mean_r = interp.phase_mean()
    out: dict[str, Any] = {
        "weight_N": WEIGHT_N, "incline_force_N": incline_force, "offset_m": offset,
        "phase_mean_resistance_si": mean_r.tolist(),
        "phase_mean_axial_m_s": float(np.mean([-r[0, 3] * OMEGA / r[0, 0] for r in matrices])),
        "positive_definite": positive_definite_everywhere(interp),
        "eight_phase_min_eigenvalue_nd": float(min(bem.smallest_eigenvalue(r) for r in matrices))}
    inverse_rxx = float(np.mean(1.0 / np.asarray(matrices)[:, 0, 0]))
    out["mean_inverse_R_xx"] = inverse_rxx
    for variant in VARIANTS:
        block: dict[str, Any] = {}
        for label, extra in (("nominal", 0.0), ("incline", incline_force)):
            for sense, name in ((1, "plus_x"), (-1, "minus_x")):
                uniform = readings(interp, np.asarray([0.0]), sense, variant, offset, extra, uniform=True)[0]
                rows = readings(interp, grid, sense, variant, offset, extra)
                u = np.asarray([row["u_along_pass_m_s"] for row in rows])
                metric = pass_metric(grid, u, sense)
                entry = {"A_uniform_field": uniform, "grid": rows, "metric": metric,
                         "A_speed_m_s": uniform["u_along_pass_m_s"]}
                if dense:
                    dense_rows = readings(interp, dense_grid, sense, variant, offset, extra)
                    du = np.asarray([row["u_along_pass_m_s"] for row in dense_rows])
                    entry["dense"] = {"step_m": DENSE_STEP_M, "metric": pass_metric(dense_grid, du, sense),
                                      "s_m": dense_grid.tolist(), "u_m_s": du.tolist()}
                block[f"{label}/{name}"] = entry
        # the two-direction mean against A
        for label in ("nominal", "incline"):
            plus, minus = block[f"{label}/plus_x"], block[f"{label}/minus_x"]
            slopes = [plus["metric"]["slope_m_s"], minus["metric"]["slope_m_s"]]
            a = 0.5 * (plus["A_speed_m_s"] + minus["A_speed_m_s"])
            block[f"{label}/two_direction"] = {
                "A_two_direction_m_s": a,
                "mean_slope_m_s": None if None in slopes else 0.5 * (slopes[0] + slopes[1]),
                "mean_over_A": None if None in slopes else 0.5 * (slopes[0] + slopes[1]) / a,
                "stalls": [plus["metric"]["stalls"], minus["metric"]["stalls"]],
                "stall_s_m": [plus["metric"]["stall_s_m"], minus["metric"]["stall_s_m"]]}
        out[variant] = block
    out["control_incline"] = incline_control(out, inverse_rxx, incline_force)
    out["blind_metric"] = blind_metric(out)
    return out


def incline_control(out: dict[str, Any], inverse_rxx: float, force: float) -> dict[str, Any]:
    """The planted incline (section 5): in A's uniform field each direction's speed moves by +-F <1/R_xx> and the
    two-direction mean by second order only; with the magnet, each pass's time-weighted slope moves by the first-order
    prediction (V^2 / 2L) int du / u^2 ds, or, where the pass stalls, its stall moves by -du / (du/ds) at the stall,
    downhill (outwards for the pass the incline helps, inwards for the other). Each within its stated tolerance."""
    found: dict[str, Any] = {"predicted_uniform_shift_m_s": force * inverse_rxx}
    tolerance_uniform, tolerance_field = 1e-3, 0.10
    for variant in VARIANTS:
        rows = {}
        block = out[variant]
        for name, sense in (("plus_x", 1), ("minus_x", -1)):
            nominal, planted = block[f"nominal/{name}"], block[f"incline/{name}"]
            shift = planted["A_speed_m_s"] - nominal["A_speed_m_s"]
            predicted = sense * force * inverse_rxx
            row: dict[str, Any] = {"uniform_shift_m_s": shift, "uniform_predicted_m_s": predicted,
                                   "uniform_relative_miss": abs(shift - predicted) / abs(predicted)}
            row["uniform_holds"] = row["uniform_relative_miss"] <= tolerance_uniform
            source = nominal.get("dense", None)
            planted_dense = planted.get("dense", None)
            if not nominal["metric"]["stalls"] and not planted["metric"]["stalls"]:
                s = np.asarray(STATIC_GRID)
                u = np.asarray([r["u_along_pass_m_s"] for r in nominal["grid"]])
                du = np.asarray([p["u_along_pass_m_s"] - r["u_along_pass_m_s"]
                                 for p, r in zip(planted["grid"], nominal["grid"], strict=True)])
                v = nominal["metric"]["slope_m_s"]
                predicted_slope = v * v / (2 * HALF_WINDOW_M) * float(np.trapezoid(du / u ** 2, s))
                computed = planted["metric"]["slope_m_s"] - v
                row.update(kind="slope", field_shift_m_s=computed, field_predicted_m_s=predicted_slope,
                           field_relative_miss=abs(computed - predicted_slope) / abs(predicted_slope))
                row["field_holds"] = row["field_relative_miss"] <= tolerance_field and \
                    np.sign(computed) == sense
            elif source is not None and planted_dense is not None and source["metric"]["stalls"] \
                    and planted_dense["metric"]["stalls"]:
                s = np.asarray(source["s_m"])
                u = np.asarray(source["u_m_s"])
                du = np.asarray(planted_dense["u_m_s"]) - u
                where = source["metric"]["stall_s_m"]
                slope_at = np.interp(where, s, np.gradient(u, s))
                du_at = float(np.interp(where, s, du))
                predicted_move = -du_at / slope_at
                moved = planted_dense["metric"]["stall_s_m"] - where
                # the incline pushes along +x: the helped pass's stall moves out, the other's in, both downhill
                row.update(kind="stall", stall_moved_m=moved, stall_predicted_m=float(predicted_move),
                           field_relative_miss=abs(moved - predicted_move) / abs(predicted_move),
                           stall_moves_downhill=bool(moved > 0.0))
                row["field_holds"] = row["field_relative_miss"] <= tolerance_field and row["stall_moves_downhill"]
            else:
                row.update(kind="mixed", field_holds=None,
                           note="one of the two readings stalls and the other does not, or no dense grid")
            rows[name] = row
        nominal_mean = 0.5 * (block["nominal/plus_x"]["A_speed_m_s"] + block["nominal/minus_x"]["A_speed_m_s"])
        planted_mean = 0.5 * (block["incline/plus_x"]["A_speed_m_s"] + block["incline/minus_x"]["A_speed_m_s"])
        rows["uniform_two_direction_change_over_shift"] = abs(planted_mean - nominal_mean) / (force * inverse_rxx)
        rows["uniform_two_direction_cancels"] = rows["uniform_two_direction_change_over_shift"] <= tolerance_uniform
        found[variant] = rows
    found["tolerances"] = {"uniform_relative": tolerance_uniform, "field_relative": tolerance_field}
    return found


def blind_metric(out: dict[str, Any]) -> dict[str, Any]:
    """The withdrawn metric beside the kept one (review 1 #1): the arithmetic mean of u over the grid, against A, for
    the pass that reaches its end in each variant; it cancels the pull's odd part by construction."""
    found = {}
    for variant in VARIANTS:
        rows = {}
        for name in ("plus_x", "minus_x"):
            entry = out[variant][f"nominal/{name}"]
            u = np.asarray([r["u_along_pass_m_s"] for r in entry["grid"]])
            rows[name] = {"arithmetic_mean_over_A": float(np.mean(u) / entry["A_speed_m_s"])}
        found[variant] = rows
    return found


# ---- the reproduction checks (C1's immediate check, review 1's sizes) -----------------------------------------------
def reproduction(matrices: np.ndarray, offset: float) -> dict[str, Any]:
    """C1's immediate check and review 1's sizes, from the zero-lag reading's time-mean pull: F_x / (U0 R_xx) at 1, 2
    and 2.5 cm on the phase means, and the stall on that linear reading; the pull along and across the tube over W;
    the pitching torque over W L / 2; the field's range at the body."""
    mean_r = np.mean(np.asarray(matrices), axis=0)
    u0 = -mean_r[0, 3] * OMEGA / mean_r[0, 0]
    thrust = u0 * mean_r[0, 0]
    n = 2000
    t = np.arange(n) / (n * FREQUENCY_HZ)
    rows = {}
    for s_cm in (0.0, 1.0, 2.0, 2.5):
        field, gradient = dipole_at(s_cm / 100.0, t, 1, offset)
        phi = np.arctan2(field[:, 2], field[:, 1])
        force, torque = wrench(phi, field, gradient)
        rows[f"{s_cm}"] = {"Fx_mean_over_W": float(force[:, 0].mean() / WEIGHT_N),
                           "Fx_mean_over_thrust": float(-force[:, 0].mean() / thrust),
                           "Fz_mean_over_W": float(force[:, 2].mean() / WEIGHT_N),
                           "Fz_peak_over_W": float(force[:, 2].max() / WEIGHT_N),
                           "pitching_torque_over_WL2": float(abs(torque[:, 1].mean()) / (WEIGHT_N * BODY_LENGTH_M / 2)),
                           "B_min_mT": float(1e3 * np.linalg.norm(field, axis=1).min()),
                           "B_max_mT": float(1e3 * np.linalg.norm(field, axis=1).max())}
    dense = np.arange(0.0, 0.03 + 1e-9, 2.5e-4)
    pull = []
    for s in dense:
        field, gradient = dipole_at(float(s), t[::4], 1, offset)
        phi = np.arctan2(field[:, 2], field[:, 1])
        pull.append(wrench(phi, field, gradient)[0][:, 0].mean())
    u = u0 + np.asarray(pull) / mean_r[0, 0]
    bad = np.nonzero(u <= 0.0)[0]
    stall = None
    if bad.size:
        i = int(bad[0])
        stall = float(dense[i - 1] + (dense[i] - dense[i - 1]) * u[i - 1] / (u[i - 1] - u[i]))
    return {"U0_m_s": float(u0), "thrust_N": float(thrust), "R_xx": float(mean_r[0, 0]), "R_xO": float(mean_r[0, 3]),
            "rows": rows, "linear_stall_s_m": stall}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resistance", required=True, help="a0f_solves.floored's JSON, or capfix_forward.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-dense", action="store_true")
    args = parser.parse_args()
    import jax

    if not jax.config.jax_enable_x64:
        raise SystemExit("run with JAX_ENABLE_X64=1")
    held = json.loads(Path(args.resistance).read_text())
    if "fixed_library_path" in held:  # the cap fix's record at 0.178 mm
        matrices = np.asarray([p["resistance_si"] for p in held["fixed_library_path"]["per_phase"]])
        offset = float(held["offset_m"])
        source = {"kind": "capfix_forward", "spacing_mm": held["spacing_mm"]}
    else:
        phases = held["phases"]
        matrices = np.asarray([phases[str(k)]["resistance_si"] for k in range(8)])
        offset = float(held["rung"]["offset_m"])
        source = {"kind": "a0f_floored", **held["rung"],
                  "wall_accepted_every_phase": all(phases[str(k)]["wall_accepted"] for k in range(8))}
    payload = {"source": source, "reproduction": reproduction(matrices, offset),
               "statics": analyse(matrices, offset, dense=not args.no_dense)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
