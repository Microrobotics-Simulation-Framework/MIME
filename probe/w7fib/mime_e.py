"""Arm E: the needle (a sphere tip, R 5 mm) as MIME's RigidInertialNode, pushed by a constant force, the bond zone's
quasi-static response delivered each step through the node's own inputs: background_load = F(d) (the contact force at the
current depth) and resistance = k(d) dt (its stiffness times the step), so the node's implicit step is an implicit spring.
Every declared input is fed every step. The equilibrium depth is compared with the direct quasi-static solve (Newton on
F(d) = P with the same medium). Plants: F0 x 1.05 (must move the depth by (1/1.05)^(2/3) - 1 = -3.2 %); S x 1.05 (cannot
move the equilibrium: a premise check, not a detection).

    python probe/w7fib/mime_e.py --mode clean|plantF|plantS --out FILE
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lat3d  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="clean", choices=("clean", "plantF", "plantS"))
ap.add_argument("--h", type=float, default=0.4)
ap.add_argument("--corr", default="skin")
ap.add_argument("--steps", type=int, default=80)
ap.add_argument("--out", required=True)
a = ap.parse_args()
T0 = time.time()
mm = 1e-3
E, nu = 50e3, 0.4
mu = E / (2 * (1 + nu))
h = a.h * mm
band = 3 * h
zone = 1.2 * mm
B = lat3d.Block(h, zone + band + 5 * h, 10 * mm, zone, band, mu, nu, corr=a.corr)
R = 5 * mm
Estar = E / (1 - nu ** 2)
top = np.hypot(B.X[..., 0], B.Y[..., 0])
gap = R - np.sqrt(np.maximum(R * R - top ** 2, 0.0))
cand = top <= 1.6 * mm
state = dict(active=None, u=None)


def medium(d):
    """active-set contact at depth d: the force F(d) and the stiffness dF/dd with the active set held"""
    if d <= 0:
        return 0.0, 0.0, 0
    active = state["active"] if state["active"] is not None else cand & (top <= math.sqrt(R * d))
    target = -d + gap
    u = state["u"]
    for _ in range(40):
        free = B.free0.copy()
        free[:, :, 0, 2][active] = 0.0
        up = np.zeros(B.shape3 + (3,))
        up[:, :, 0, 2] = np.where(active, target, 0.0)
        u, _, _ = B.solve(up, free, x0=u)
        f = np.asarray(B.grad(jnp.asarray(u.reshape(-1)))).reshape(B.shape3 + (3,))
        fz = f[:, :, 0, 2]
        rel = active & (fz > 0)
        pen = cand & (~active) & (u[:, :, 0, 2] > target + 1e-9 * d)
        if not rel.any() and not pen.any():
            break
        active = (active & ~rel) | pen
    F = -4.0 * float(np.sum(fz[active]))
    free = B.free0.copy()
    free[:, :, 0, 2][active] = 0.0
    up = np.zeros(B.shape3 + (3,))
    up[:, :, 0, 2] = np.where(active, -1.0, 0.0)
    uk, _, _ = B.solve(up, free)
    fk = np.asarray(B.grad(jnp.asarray(uk.reshape(-1)))).reshape(B.shape3 + (3,))
    k = -4.0 * float(np.sum(fk[:, :, 0, 2][active]))
    state["active"], state["u"] = active, u
    return F, k, int(active.sum())


# target depth 0.03 R by Hertz -> the push
d_h = 0.03 * R
P = 4.0 / 3.0 * Estar * math.sqrt(R) * d_h ** 1.5
# direct quasi-static reference: Newton on F(d) = P
d = d_h
ref_hist = []
for it in range(12):
    F, k, na = medium(d)
    ref_hist.append((d, F, k, na))
    dn = d + (P - F) / k
    if abs(dn - d) < 1e-10 * d:
        d = dn
        break
    d = dn
F_ref, _, _ = medium(d)
d_ref = d
print(json.dumps(dict(P=P, d_ref=d_ref, F_ref=F_ref, hist=ref_hist)), flush=True)

from mime.core import ledger as books  # noqa: E402
from mime.nodes.body._rigid_inertial import RigidInertialNode  # noqa: E402

rho = 7900.0  # a steel needle tip
mass = rho * 4.0 / 3.0 * math.pi * R ** 3
k0 = ref_hist[-1][2]
omega = math.sqrt(k0 / mass)
dt = 3.0 / omega  # backward Euler, strongly damped
dl = [books.LedgerDelivery("resistance", "medium:S", books.DISSIPATIVE, books.SOLVED, books.CHECK_DEFINITE),
      books.LedgerDelivery("background_load", "medium:F0", books.UNMETERED, books.LOAD),
      books.LedgerDelivery("applied_wrench", "push", books.UNMETERED, books.LOAD)]
body = RigidInertialNode("needle", dt, mass=mass, inertia=(0.4 * mass * R * R,) * 3,
                         initial_pose=(0.0, 0.0, R - 0.5 * d_h, 1.0, 0.0, 0.0, 0.0), initial_twist=(0.0,) * 6,
                         ledger=tuple(x.encoded() for x in dl))
names = {x.law: books.law_input(x.field, i) for i, x in enumerate(dl) if x.kind != books.HOSTED}
st = dict(body.initial_state())
state["active"], state["u"] = None, None
fF = 1.05 if a.mode == "plantF" else 1.0
fS = 1.05 if a.mode == "plantS" else 1.0
rows = []
for step in range(a.steps):
    pose = np.asarray(st["pose"], float)
    d = R - pose[2]
    F, k, na = medium(d)
    S = np.eye(6) * 1e-9 * max(k, 1.0) * dt
    S[2, 2] = fS * k * dt
    F0 = np.array([0.0, 0.0, fF * F, 0.0, 0.0, 0.0])
    inp = {nm: jnp.zeros(entry.shape, dtype=jnp.float64) for nm, entry in body.boundary_input_spec().items()}
    inp[names["medium:S"]] = jnp.asarray(S)
    inp[names["medium:F0"]] = jnp.asarray(F0)
    inp[names["push"]] = jnp.asarray([0.0, 0.0, -P, 0.0, 0.0, 0.0])
    st = body.update(st, inp, dt)
    v = float(np.asarray(st["twist"])[2])
    rows.append(dict(step=step, d=d, F=F, k=k, active=na, vz=v))
    if step % 5 == 0:
        print(json.dumps(rows[-1]), flush=True)
d_end = R - float(np.asarray(st["pose"])[2])
led = {kk: np.asarray(vv).tolist() for kk, vv in st.items() if kk.startswith("ledger_")}
eps = float(np.finfo(np.float64).eps)
rec = dict(args=vars(a), P=P, d_ref=d_ref, F_ref=F_ref, d_end=d_end, rel=d_end / d_ref - 1,
           predicted_plantF_rel=(1 / 1.05) ** (2 / 3) - 1, d_hertz=d_h, dt=dt, omega=omega, mass=mass,
           ledger=dict(identity_worst_eps=float(led["ledger_identity_worst"]) / eps,
                       unattributed_worst=float(led["ledger_unattributed_worst"]),
                       indefinite_steps=led["ledger_indefinite_steps"], gift_steps=led["ledger_gift_steps"]),
           rows=rows, nodes=B.N, wall_s=time.time() - T0)
json.dump(rec, open(a.out, "w"), indent=1)
print(json.dumps({kk: vv for kk, vv in rec.items() if kk != "rows"}))
