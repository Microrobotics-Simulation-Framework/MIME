"""Semi-analytic anchors (r39). Every function returns dimensionless resistance functions.

- brenner_normal(delta): sphere moving normal to a plane wall, F/(6 pi mu a U); Brenner 1961 closed form as quoted by the Q1
  researcher (checked there against Mitchell-Spagnolie Table 4 [read]); reproduced here.
- twosphere_normal(delta, lam): sphere radius a moving along the line of centres towards a HELD sphere of radius lam*a,
  bispherical Stimson-Jeffery stream function, one 4x4 solve per mode n (my own code). lam = inf is the plane wall; the
  force constant is fixed by that limit against brenner_normal, then must reproduce it at every delta (a control).
- jeffery_normal_rotation(delta): torque/(8 pi mu a^3 Omega), rotation about the wall normal: sinh^3(al) sum 1/sinh^3(n al)
  [recalled formula, Q1 researcher; limits checked].
- tangential asymptotics (Ekiel-Jezewska et al. 2008, [read: fetch summary] via Q1): f_tt (6 pi mu a U), c_t (8 pi mu a^2),
  c_r (8 pi mu a^3), with their O(eps) terms.
- Brinkman unbounded sphere: F = 6 pi (1 + sqrt s + s/9) [read, Vilfan et al. eq 12]; T = 8 pi (1 + sqrt s + s/3)/(1 + sqrt s)
  [derived by the Q2 researcher; checked below by my own ODE quadrature].
"""
import sys, json
import mpmath as mp

mp.mp.dps = 50


def brenner_normal(delta, tol=mp.mpf("1e-30")):
    al = mp.acosh(1 + mp.mpf(delta))
    s = mp.mpf(0)
    n = 1
    while True:
        num = 2 * mp.sinh((2 * n + 1) * al) + (2 * n + 1) * mp.sinh(2 * al)
        den = 4 * mp.sinh((n + mp.mpf(1) / 2) * al) ** 2 - (2 * n + 1) ** 2 * mp.sinh(al) ** 2
        t = mp.mpf(n * (n + 1)) / ((2 * n - 1) * (2 * n + 3)) * (num / den - 1)
        s += t
        if abs(t) < tol * abs(s) and n > 10:
            break
        n += 1
    return mp.mpf(4) / 3 * mp.sinh(al) * s, n


def _k(n, xi, V, c2):
    """Coefficient of C_{n+1}^{-1/2} in (cosh xi - mu)^{3/2} * (-V rho^2/2): the rigid-translation stream function."""
    e1 = mp.exp(-(n - mp.mpf(1) / 2) * abs(xi))
    e2 = mp.exp(-(n + mp.mpf(3) / 2) * abs(xi))
    k = -V * c2 / mp.sqrt(2) * n * (n + 1) * (e1 / (2 * n - 1) - e2 / (2 * n + 3))
    sg = 1 if xi >= 0 else -1
    dk = -V * c2 / mp.sqrt(2) * n * (n + 1) * (-(n - mp.mpf(1) / 2) * e1 / (2 * n - 1) + (n + mp.mpf(3) / 2) * e2 / (2 * n + 3)) * sg
    return k, dk


def _basis(n, xi):
    p, q = n - mp.mpf(1) / 2, n + mp.mpf(3) / 2
    f = [mp.cosh(p * xi), mp.sinh(p * xi), mp.cosh(q * xi), mp.sinh(q * xi)]
    df = [p * mp.sinh(p * xi), p * mp.cosh(p * xi), q * mp.sinh(q * xi), q * mp.cosh(q * xi)]
    return f, df


def twosphere_sum(delta, lam, tol=mp.mpf("1e-28")):
    """Returns (sum_n (a+b+c+d), c, alpha, nterms) for sphere 1 (radius 1, at xi = alpha, moving with V = 1) and the held
    body at xi = -beta (lam = inf: plane at xi = 0)."""
    delta = mp.mpf(delta)
    if lam == "inf":
        al = mp.acosh(1 + delta)
        c = mp.sinh(al)
        be = mp.mpf(0)
    else:
        lam = mp.mpf(lam)
        # centres distance d = 1 + lam + delta; c^2 = (d^2 - (1+lam)^2)(d^2 - (1-lam)^2)/(4 d^2)
        d = 1 + lam + delta
        c = mp.sqrt((d * d - (1 + lam) ** 2) * (d * d - (1 - lam) ** 2)) / (2 * d)
        al = mp.asinh(c / 1)
        be = mp.asinh(c / lam)
    c2 = c * c
    S = mp.mpf(0)
    n = 1
    while True:
        f1, df1 = _basis(n, al)
        k1, dk1 = _k(n, al, 1, c2)
        f2, df2 = _basis(n, -be)
        A = mp.matrix([f1, df1, f2, df2])
        rhs = mp.matrix([k1, dk1, 0, 0])
        x = mp.lu_solve(A, rhs)
        t = x[0] + x[1] + x[2] + x[3]
        S += t
        if n > 10 and abs(t) < tol * abs(S):
            break
        n += 1
    return S, c, al, n


def twosphere_normal(delta, lam, K):
    S, c, al, n = twosphere_sum(delta, lam)
    return K * S / c, n


def jeffery_normal_rotation(delta, tol=mp.mpf("1e-30")):
    al = mp.acosh(1 + mp.mpf(delta))
    s, n = mp.mpf(0), 1
    while True:
        t = 1 / mp.sinh(n * al) ** 3
        s += t
        if t < tol * s and n > 5:
            break
        n += 1
    return mp.sinh(al) ** 3 * s, n


def tangential_asym(eps):
    eps = mp.mpf(eps)
    L = mp.log(eps)
    ftt = -mp.mpf(8) / 15 * L + mp.mpf("0.95429") - mp.mpf(64) / 375 * eps * L + mp.mpf("0.42945") * eps
    ct = -mp.mpf(1) / 10 * L - mp.mpf("0.19295") - mp.mpf(43) / 250 * eps * L + mp.mpf("0.10058") * eps
    cr = -mp.mpf(2) / 5 * L + mp.mpf("0.37089") - mp.mpf(66) / 125 * eps * L + mp.mpf("0.34008") * eps
    return ftt, ct, cr


def brinkman_unbounded(s):
    s = mp.mpf(s)
    r = mp.sqrt(s)
    return 1 + r + s / 9, (1 + r + s / 3) / (1 + r)


def brinkman_torque_ode(s):
    """Independent check of the torque: u_phi = f(r) sin th, f = (1 + k r) e^{-k(r-1)}/(r^2 (1+k)) solves
    f'' + 2f'/r - 2f/r^2 - k^2 f = 0 (checked by finite differences), torque/(8 pi) = -(1/2) r^3 d(f/r)/dr at r = 1... computed
    by numerical differentiation."""
    k = mp.sqrt(mp.mpf(s))
    f = lambda r: (1 + k * r) * mp.exp(-k * (r - 1)) / (r * r * (1 + k))
    res = max(abs(mp.diff(f, r, 2) + 2 * mp.diff(f, r) / r - 2 * f(r) / r ** 2 - k * k * f(r)) for r in [mp.mpf(1), mp.mpf(1.5), mp.mpf(3)])
    g = lambda r: f(r) / r
    tau = -mp.diff(g, 1)  # shear stress tau_r_phi / (mu Omega sin th) at r=1 = r d(f/r)/dr, sign: resisting
    # torque = int tau * a sin th * dA = tau * 2 pi * int sin^3 th dth = tau * 8 pi / 3 ; normalise by 8 pi
    return tau / 3, res


if __name__ == "__main__":
    out = {}
    deltas = ["0.1", "0.01", "0.001", "0.0453385", "0.1276260"]
    # control: Mitchell-Spagnolie Table 4 exact normal values [read via Q1], alpha = 0.5, 0.3
    ms = {"0.5": "9.251765", "0.3": "23.66048", "1": "3.036064"}
    for a_, v in ms.items():
        dl = mp.cosh(mp.mpf(a_)) - 1
        val, n = brenner_normal(dl)
        print(f"control M-S alpha {a_}: delta {mp.nstr(dl, 10)} brenner {mp.nstr(val, 10)} table {v}")
    bren = {}
    for dl in ["0.1", "0.01", "0.001"]:
        val, n = brenner_normal(dl)
        bren[dl] = val
        print(f"brenner normal delta {dl}: {mp.nstr(val, 15)} ({n} terms)")
    # fix K from the plane limit at delta 0.1, then check the plane limit at 0.01, 0.001 and two-sphere values
    S, c, al, n = twosphere_sum("0.1", "inf")
    K = bren["0.1"] * c / S
    print("K (should be a fixed constant):", mp.nstr(K, 15), " K/(2 sqrt2 pi /6pi)=", mp.nstr(K / (2 * mp.sqrt(2) * mp.pi / (6 * mp.pi)), 12))
    for dl in ["0.01", "0.001"]:
        v, n = twosphere_normal(dl, "inf", K)
        print(f"plane via 4x4 delta {dl}: {mp.nstr(v, 15)} vs brenner {mp.nstr(bren[dl], 15)} rel {mp.nstr((v - bren[dl]) / bren[dl], 3)} ({n} terms)")
    out["brenner_normal"] = {k: mp.nstr(v, 15) for k, v in bren.items()}
    ts = {}
    for lam in ["4", "1", "1e6"]:
        for dl in ["0.1", "0.01", "0.001"]:
            v, n = twosphere_normal(dl, lam, K)
            ts[f"lam{lam}_d{dl}"] = mp.nstr(v, 15)
            print(f"two-sphere lam {lam} delta {dl}: {mp.nstr(v, 15)} ({n} terms)")
    out["twosphere_normal"] = ts
    jr = {}
    for dl in ["0.1", "0.01", "0.001"]:
        v, n = jeffery_normal_rotation(dl)
        jr[dl] = mp.nstr(v, 15)
        print(f"jeffery normal rotation delta {dl}: {mp.nstr(v, 15)}")
    out["jeffery_normal_rotation"] = jr
    ta = {}
    for dl in ["0.1", "0.01", "0.001", "0.0453385", "0.1276260"]:
        ftt, ct, cr = tangential_asym(dl)
        ta[dl] = [mp.nstr(ftt, 10), mp.nstr(ct, 10), mp.nstr(cr, 10)]
        print(f"tangential asym delta {dl}: ftt {mp.nstr(ftt, 8)} ct {mp.nstr(ct, 8)} cr {mp.nstr(cr, 8)}")
    out["tangential_asym"] = ta
    br = {}
    for s in [130, 1381]:
        F, T = brinkman_unbounded(s)
        Tode, res = brinkman_torque_ode(s)
        br[str(s)] = [mp.nstr(F, 15), mp.nstr(T, 15)]
        print(f"brinkman s {s}: F/6pi {mp.nstr(F, 12)} T/8pi {mp.nstr(T, 12)} T(ode) {mp.nstr(Tode, 12)} ode residual {mp.nstr(res, 3)}")
    out["brinkman_unbounded"] = br
    json.dump(out, open(sys.argv[1], "w"), indent=1)
