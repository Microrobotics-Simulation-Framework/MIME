"""Arms of spike w7bur, one a job (PROBE_ARM) or by name on the command line. Lengths mm, stresses Pa, times s."""
import json, os, sys, time
import numpy as np
from gel import Grid, Gel
from geoms import rotor, thread

OUT = os.environ.get("A_OUT", "probe/out")
G0 = 1000.0
L, R0, RO = 8.0, 1.2, 3.5                     # the disc: box, body radius, clamp radius
ROOT, TIP = 0.85, 1.2                         # the thread's core and tip radii
LAM, Y0, HT, GELH = 1.0, 0.125, 0.35, 2.0     # the unrolled thread: wavelength, base, height, gel above the crest
TANA = 1.0 / (2 * np.pi * 0.5 * (ROOT + TIP))  # helix angle at the mean radius
CA, SA = 1 / np.hypot(1, TANA), TANA / np.hypot(1, TANA)


def kex(Ri, Ro=RO, G=G0):
    return 4 * np.pi * G * Ri ** 2 * Ro ** 2 / (Ro ** 2 - Ri ** 2)


def wall_law(s1, n, V1=1.0, Vreg=1e-3, G=G0):
    """sigma_w = G s1 (V/V1)^n, linear below Vreg (the stuck state as a slow creep)."""
    def law(V):
        return G * s1 * np.where(V > Vreg, (np.maximum(V, Vreg) / V1) ** n, (Vreg / V1) ** n * V / Vreg)
    return law


def paste_law(tau0, mu, greg):
    return lambda gam: tau0 * (1 - np.exp(-gam / greg)) + mu * gam


def save(name, res):
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}), flush=True)


# ---- (a) and its control: bonded bodies ---------------------------------------------------------------------------------
def a_cyl(N, order=3, plant=1.0, dth=0.01, th_end=0.5):
    g = Grid(N, N, L / N)
    gel = Gel(g, G0, rotor(L, R0, RO, 1.0), dth, order=order, plant_G=plant)
    rows = []
    for k in range(int(round(th_end / dth))):
        o = gel.step()
        rows.append([o["t"], -o["Tz"] / (kex(R0) * o["t"]), o["kappa_max"]])
    r = np.array(rows)
    pick = lambda th: float(r[np.argmin(abs(r[:, 0] - th)), 1])
    return dict(N=N, cells_a_radius=R0 * N / L, order=order, plant=plant, ratio_005=pick(0.05), ratio_025=pick(0.25),
                ratio_050=pick(0.5), kappa_end=float(r[-1, 2]), rows=rows)


def a_lob(N, lobes=2, dth=0.005, th_end=0.6):
    """A lobed rotor (the thread's cross-section as lobes between the core and the tip), bonded: torque against angle."""
    g = Grid(N, N, L / N)
    R, A = 0.5 * (ROOT + TIP), 0.5 * (TIP - ROOT)
    gel = Gel(g, G0, rotor(L, R, RO, 1.0, amp=A, lobes=lobes), dth)
    rows = []
    for k in range(int(round(th_end / dth))):
        o = gel.step()
        rows.append([o["t"], -o["Tz"] / o["t"], o["Fx"], o["Fy"], o["teq_max"] / G0])
    r = np.array(rows)
    m = r[:, 0] > 0.05
    K = r[m, 1]
    return dict(N=N, cells_a_blade=0.3 * N / L, K_mean=float(K.mean()), K_ripple_rms=float(K.std() / K.mean()),
                K_ripple_pp=float((K.max() - K.min()) / K.mean()), K_root=kex(ROOT), K_tip=kex(TIP),
                frac_of_bracket=float((K.mean() - kex(ROOT)) / (kex(TIP) - kex(ROOT))), rows=rows)


def thread_setup(nx, power, Vn, Vs, height=HT, **kw):
    h = LAM / nx
    ytop = Y0 + height + GELH
    ny = int(np.ceil((ytop + 3 * h) / h))
    g = Grid(nx, ny, h)
    return g, thread(LAM, Y0, height, power, Vn, Vs, ytop, **kw), ytop


def a_wav(nx, power, height=HT, travel=0.25):
    """The bonded wavy wall, moved in its plane and out of it by a quarter wavelength: stiffness and ripple."""
    g, geom, ytop = thread_setup(nx, power, 1.0, 1.0, height)
    dt = 0.25 * g.h
    gel = Gel(g, G0, geom, dt, antiplane=True)
    rows = []
    for k in range(int(round(travel * LAM / dt))):
        o = gel.step()
        rows.append([o["t"], -o["Fx"] / (o["t"] * LAM), -o["Fz"] / (o["t"] * LAM), o["teq_max"] / G0])
    r = np.array(rows); m = r[:, 0] > 0.04
    flat = lambda gap: G0 / gap
    res = dict(nx=nx, power=power, height=height, cells_a_blade=0.3 / g.h,
               kz_mean=float(r[m, 2].mean()), kz_ripple_pp=float((r[m, 2].max() - r[m, 2].min()) / r[m, 2].mean()),
               kx_first=float(r[m, 1][0]), kx_last=float(r[m, 1][-1]),
               flat_root=flat(GELH + height), flat_crest=flat(GELH), rows=rows)
    return res


# ---- convection controls ------------------------------------------------------------------------------------------------
def ctl_trans(N=96, cfl=0.37, boxes=2.0):
    """A rigid translation of everything through the grid, no solve: a failed disc and a failed layer 2 cells wide, carried on
    material coordinates, as a convected scalar (linear) and as one (cubic)."""
    g = Grid(N, N, 1.0 / N)
    res = {}
    r = np.hypot(g.xc - 0.5, g.yc - 0.5)
    pat = ((r < 0.12) | (abs(r - 0.3) < g.h)).astype(float)
    for name, order in (("linear", 1), ("cubic", 3)):
        gel = Gel(g, 1.0, None, cfl * g.h, d_order=order)
        gel.d_ref = pat.copy(); gel.d_adv = pat.copy()
        U = np.concatenate([np.full(g.n, 1.0), np.full(g.n, 0.61803)])
        nst = int(round(boxes / (cfl * g.h)))
        for k in range(nst):
            gel.convect(U, np.zeros(g.n), None)
        mat = gel.damage_at_centres()
        # the exact pattern, moved
        sx, sy = nst * gel.dt * 1.0, nst * gel.dt * 0.61803
        rr = np.hypot((g.xc - sx - 0.5 + 0.5) % 1.0 - 0.5, (g.yc - sy - 0.5 + 0.5) % 1.0 - 0.5)
        ex = ((rr < 0.12) | (abs(rr - 0.3) < g.h)).astype(float)
        adv = gel.d_adv
        res[name] = dict(steps=nst, area_exact=float(ex.sum()), area_material=float((mat > 0.5).sum()),
                         mismatch_material=float(abs((mat > 0.5) - ex).sum() / ex.sum()),
                         area_convected=float((adv > 0.5).sum()), mismatch_convected=float(abs((adv > 0.5) - ex).sum() / ex.sum()),
                         convected_max=float(adv.max()), convected_min=float(adv.min()),
                         layer_peak_convected=float(adv[abs(rr - 0.3) < g.h].mean()),
                         map_error_cells=float(max(abs(gel.xi[0] + sx).max(), abs(gel.xi[1] + sy).max()) / g.h))
    res.update(N=N, cfl=cfl, boxes=boxes)
    return res


# ---- (b): the wall law -----------------------------------------------------------------------------------------------------
def b_cyl(N, V, s1=0.1, n=0.4, band_cells=3, turns=0.3, plant=1.0, dth=0.01, order=3):
    """A cylinder turning at surface speed V (mm/s) with the wall law on a band; steady torque against 2 pi R^2 sigma_w(V)."""
    g = Grid(N, N, L / N)
    om = V / R0
    law = wall_law(s1, n, Vreg=1e-3 * V)
    gel = Gel(g, G0, rotor(L, R0, RO, om), dth / om, band=band_cells * g.h, wall_law=law, plant_wall=plant, order=order)
    Tlaw = 2 * np.pi * R0 ** 2 * float(law(np.array(V)))
    rows = []; snap = {}
    nst = int(round(turns * 2 * np.pi / dth))
    for k in range(nst):
        o = gel.step()
        rows.append([o["t"] * om, -o["Tz"] / Tlaw, o["teq_max"] / G0])
        turn = o["t"] * om / (2 * np.pi)
        for mark in (1, 2, 5, 10):
            if mark not in snap and turn >= mark:
                snap[mark] = gel.xi.copy()
    r = np.array(rows)
    res = dict(N=N, V=V, s1=s1, n=n, band_cells=band_cells, plant=plant, order=order, turns=turns, T_over_law_end=float(r[-1, 1]),
               T_over_law_last_tenth=float(r[-max(1, nst // 10):, 1].mean()), teq_over_G=float(r[-1, 2]), rows=rows[::max(1, nst // 400)])
    if 1 in snap:
        gelm = gel.last["gel_c"] & (gel.last["ph_c"] > (band_cells + 3) * g.h)
        for mark in snap:
            if mark > 1:
                d = snap[mark] - snap[1]
                res["map_drift_cells_turn_%d" % mark] = float(np.sqrt(d[0] ** 2 + d[1] ** 2)[gelm].max() / g.h)
        res["torque_turn_1"] = float(r[np.argmin(abs(r[:, 0] - 2 * np.pi)), 1])
    return res


# ---- the thread, unrolled: (b), (c), (d) -------------------------------------------------------------------------------
def t_run(nx, power, V, mode, height=HT, s1=0.1, n=0.4, band_cells=3, tau_y=1.0, tau_p=0.1, mu_p=0.01, damage="material",
          waves=6.0, plant=1.0, cfl=0.4):
    """mode b: wall law, no failure. c: bonded, failure at tau_y G, paste tau_p G + mu_p gam. d: both.
    The screw's surface speed V splits into the phase speed along x (V sin a) and the sliding along the groove (V cos a)."""
    Vn, Vs = V * SA, V * CA
    g, geom, ytop = thread_setup(nx, power, Vn, Vs, height, ramp=0.05 * LAM / Vn)
    dt = cfl * g.h / Vn
    kw = dict(antiplane=True, damage=damage)
    if mode in "bd":
        kw.update(band=band_cells * g.h, wall_law=wall_law(s1, n, Vreg=1e-3 * V), plant_wall=plant)
    if mode in "cd":
        kw.update(tau_y=tau_y * G0, paste=paste_law(tau_p * G0, mu_p, 1e-2 * V / LAM))
    gel = Gel(g, G0, geom, dt, **kw)
    rows = []
    nst = int(round(waves * LAM / (Vn * dt)))
    groove = lambda: gel.last["gel_c"] & (g.yc < Y0 + height) & (gel.last["ph_c"] > band_cells * g.h)
    for k in range(nst):
        o = gel.step()
        tn, ts = -o["Fx"] / LAM, -o["Fz"] / LAM                     # mean stresses the gel puts on the wall, against its motion
        gr = groove()
        u = gel.last["U"][:g.n].reshape(g.nx, g.ny); w = gel.last["W"].reshape(g.nx, g.ny)
        rows.append([o["t"] * Vn / LAM, tn / G0, ts / G0, (ts * CA + tn * SA) / G0, o["teq_max"] / G0, o.get("failed_cells", 0),
                     float(u[gr].mean() / Vn) if gr.any() else 0.0, float(w[gr].mean() / Vs) if gr.any() else 0.0])
    r = np.array(rows)
    last = r[:, 0] > waves - 1.0
    mid = (r[:, 0] > 1.0) & (r[:, 0] <= 2.0)
    res = dict(nx=nx, power=power, V=V, mode=mode, height=height, s1=s1, n=n, tau_y=tau_y, tau_p=tau_p, mu_p=mu_p, damage=damage,
               plant=plant, waves=waves, cells_a_blade=0.3 / g.h,
               wall_law_over_G=float(wall_law(s1, n)(np.array(V)) / G0) if mode in "bd" else None,
               tau_az_over_G=float(r[last, 3].mean()), tau_n_over_G=float(r[last, 1].mean()), tau_s_over_G=float(r[last, 2].mean()),
               tau_az_wave2=float(r[mid, 3].mean()), tau_az_ripple_pp=float((r[last, 3].max() - r[last, 3].min()) / abs(r[last, 3].mean())),
               teq_max_over_G=float(r[:, 4].max()), teq_last=float(r[last, 4].max()),
               failed_wave1=int(r[np.argmin(abs(r[:, 0] - 1.0)), 5]), failed_wave2=int(r[mid, 5].mean()), failed_last=int(r[last, 5].mean()),
               groove_u_over_Vn=float(r[last, 6].mean()), groove_w_over_Vs=float(r[last, 7].mean()),
               rows=rows[::max(1, nst // 300)])
    if mode in "cd":
        d = gel.damage_at_centres() * gel.last["gel_c"]
        res["failed_by_height"] = [float(x) for x in d.mean(axis=0)]
    return res


def main(arm):
    t0 = time.time()
    p = arm.split("_")
    if p[0] == "cyl":
        res = a_cyl(int(p[1]), order=int(p[2]), plant=float(p[3]) if len(p) > 3 else 1.0)
    elif p[0] == "lob":
        res = a_lob(int(p[1]))
    elif p[0] == "wav":
        res = a_wav(int(p[1]), int(p[2]), height=float(p[3]) if len(p) > 3 else HT)
    elif p[0] == "ctl":
        res = ctl_trans(int(p[1]), float(p[2]))
    elif p[0] == "bcyl":      # bcyl_N_V_turns_plant_order
        res = b_cyl(int(p[1]), float(p[2]), turns=float(p[3]), plant=float(p[4]) if len(p) > 4 else 1.0,
                    order=int(p[5]) if len(p) > 5 else 3)
    elif p[0] == "t":         # t_mode_nx_power_V_height_s1_taup_damage_waves_plant
        res = t_run(int(p[2]), int(p[3]), float(p[4]), p[1], height=float(p[5]), s1=float(p[6]), tau_p=float(p[7]),
                    damage={"m": "material", "a": "advected"}[p[8]], waves=float(p[9]), plant=float(p[10]) if len(p) > 10 else 1.0)
    else:
        raise SystemExit("unknown arm " + arm)
    res["arm"] = arm; res["wall_s"] = time.time() - t0
    save(arm, res)


if __name__ == "__main__":
    main(os.environ.get("PROBE_ARM") or sys.argv[1])
