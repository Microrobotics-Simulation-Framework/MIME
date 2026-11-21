"""w7bur free-body arms: the project's RigidInertialNode (ledger attached, every declared input fed) turning in the gel under the
project's MagneticTorqueNode; the gel's torque is delivered as a resistance and a background load each step.

  fr_<law>_<G>_<ratio>   law w: the wall law on a band (s1 0.1, exponent 0.4); law b: bonded (the control: no steady turning).
                         The field's m B is the steady torque the SAME wall-law model asks at 1 Hz in a gel of 1000 Pa (measured
                         first, kinematically); the field turns at ratio x 1 Hz. Step-out is read, not put in.
SI in the body; the gel in mm, Pa, s, per mm of depth, times the screw's length."""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gel import Grid, Gel
from geoms import rotor, DrivenRotor
from arms import wall_law, L, R0, RO
OUT = os.environ.get("A_OUT", "probe/out"); T0 = time.time()
LEN, RHO_B = 7.0, 1200.0                       # mm; kg/m^3
TQ = 1e-9 * LEN                                # N m per (Pa mm^2 a mm of depth): 1e-6 N a mm ... x 1e-3 m x LEN mm
N, S1, EXPN, BAND, FREF, GREF = 48, 0.1, 0.4, 3, 1.0, 1000.0


def steady_torque(G, f):
    """The wall-law model's steady torque at a prescribed rate (kinematic), N m."""
    g = Grid(N, N, L / N); om = 2 * math.pi * f
    gel = Gel(g, G, rotor(L, R0, RO, om), 0.01 / om, band=BAND * g.h, wall_law=wall_law(S1, EXPN, Vreg=1e-3 * om * R0, G=G))
    for k in range(120):
        o = gel.step()
    return -o["Tz"] * TQ


def run(law, G, ratio, periods=8.0, dth=0.02, plant=1.0):
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from mime.core import ledger as books
    from mime.nodes.body._rigid_inertial import RigidInertialNode
    from mime.nodes.magnetic._torque import MagneticTorqueNode
    mB = steady_torque(GREF, FREF)
    f = ratio * FREF; omega = 2 * math.pi * f; dt = dth / omega
    g = Grid(N, N, L / N); rot = DrivenRotor(L, R0, RO)
    kw = dict(band=BAND * g.h, wall_law=wall_law(S1, EXPN, Vreg=1e-3 * 2 * math.pi * FREF * R0, G=G)) if law == "w" else {}
    gel = Gel(g, G, rot, dt, **kw)
    mass = RHO_B * math.pi * (R0 * 1e-3) ** 2 * LEN * 1e-3; inertia = 0.5 * mass * (R0 * 1e-3) ** 2
    dl = [books.LedgerDelivery("resistance", "gel:S", books.DISSIPATIVE, books.SOLVED, books.CHECK_DEFINITE),
          books.LedgerDelivery("background_load", "gel:F0", books.UNMETERED, books.LOAD),
          books.LedgerDelivery("applied_wrench", "magnet", books.UNMETERED, books.LOAD)]
    body = RigidInertialNode("body", dt, mass=mass, inertia=(inertia,) * 3, initial_pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                             initial_twist=(0.0,) * 6, ledger=tuple(d.encoded() for d in dl))
    names = {d.law: books.law_input(d.field, i) for i, d in enumerate(dl)}
    magnet = MagneticTorqueNode("magnet", dt, moment=(1.0, 0.0, 0.0))
    st, ms = dict(body.initial_state()), dict(magnet.initial_state())
    def feed(vals):
        inp = {name: jnp.zeros(entry.shape, dtype=jnp.float64) for name, entry in body.boundary_input_spec().items()}
        for k, v in vals.items():
            inp[names[k]] = jnp.asarray(v, dtype=jnp.float64)
        return inp
    def torque(om, commit=False):
        rot.omega = om
        return gel.step(commit=commit)["Tz"] * TQ
    nst = int(round(periods * 2 * math.pi / dth))
    t, turned, slips, defect_max = 0.0, 0.0, 0, 0.0
    rows = []
    for k in range(nst):
        pose, q = np.asarray(st["pose"], float), np.asarray(st["twist"], float)
        B = np.array([math.cos(omega * (t + dt)), math.sin(omega * (t + dt)), 0.0]) * mB
        ms = magnet.update(ms, {"pose": jnp.asarray(pose), "flux_density": jnp.asarray(B)}, dt)
        w0 = q[5]; dw = 1e-3 * omega
        Ta, Tb = torque(w0), torque(w0 + dw)
        S = max(-(Tb - Ta) / dw, 0.0); F0 = Ta + S * w0
        S6 = np.diag([1e-6, 1e-6, 1e-6, 1e-12, 1e-12, plant * S])      # the gel's torque about the axis; the rest are placeholders (no load acts there)
        F6 = np.array([0, 0, 0, 0, 0, F0])
        st = body.update(st, feed({"gel:S": S6, "gel:F0": F6, "magnet": ms["wrench"]}), dt)
        w1 = float(np.asarray(st["twist"])[5])
        Tc = torque(w1, commit=True)
        defect_max = max(defect_max, abs(Tc - (F0 - S * w1)) / mB)
        t += dt; turned += dt * w1
        lag = omega * t - turned
        if lag > 2 * math.pi * (slips + 1): slips += 1
        if k % max(1, nst // 600) == 0:
            rows.append([t, turned, lag, w1 / omega, Tc / mB, float(np.asarray(ms["wrench"])[5]) / mB])
        if k % 200 == 0: print(k, nst, round(time.time() - T0), "lag", round(lag, 3), "rate", round(w1 / omega, 3), flush=True)
    rec = {k2: np.asarray(v).tolist() for k2, v in st.items() if k2.startswith("ledger_")}
    eps = float(np.finfo(np.float64).eps)
    r = np.array(rows); half = len(r) // 2
    return dict(law=law, G=G, ratio=ratio, f=f, mB=mB, dt=dt, steps=nst, slips=slips, mean_rate_ratio=turned / (omega * t),
                late_rate_ratio=float((r[-1, 1] - r[half, 1]) / (omega * (r[-1, 0] - r[half, 0]))), turned_rad=turned,
                defect_max_over_mB=defect_max, plant=plant,
                ledger=dict(identity_worst_eps=float(rec["ledger_identity_worst"]) / eps, unattributed_worst=float(rec["ledger_unattributed_worst"]),
                            indefinite_steps=rec["ledger_indefinite_steps"], gift_steps=rec["ledger_gift_steps"]), rows=rows)


if __name__ == "__main__":
    arm = os.environ.get("PROBE_ARM") or sys.argv[1]
    p = arm.split("_")
    res = run(p[1], float(p[2]), float(p[3]), periods=float(p[4]) if len(p) > 4 else 8.0, plant=float(p[5]) if len(p) > 5 else 1.0)
    res["arm"] = arm; res["wall_s"] = time.time() - T0
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, arm + ".json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
