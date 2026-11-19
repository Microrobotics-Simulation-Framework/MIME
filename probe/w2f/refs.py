"""The published references for a rigid sphere on the axis of a long tube, read in Singh, Sharma and Gupta-Bhaya, "Studies on
falling ball viscometry", arXiv:1202.1400v2 (2012), section 3.2 (the originals not opened):

  X = F_D / (6 pi mu a V), the drag of a sphere moving at V along the axis of a tube of quiescent fluid over Stokes's
  unbounded drag at the same speed; lambda = d / D; Re = V d / nu (the sphere's diameter and speed).

  haberman_sayre(lam)       Haberman and Sayre (1958), Re -> 0 (the review's Eq. 22, coefficients as in its Eq. 20a)
  cheng(re_inf)             Cheng (2009), the unbounded sphere (Eq. 16)
  kehlenbeck_difelice(...)  Kehlenbeck and Di Felice (1999), the velocity wall factor f = V_t / V_inf at one force,
                            fitted to data over 2 < Re_inf < 185, 0.1 < lambda < 0.9 (Eq. 19)
  wham(re, lam)             Wham, Basaran and Byers (1996), a fit to their finite-element runs of a falling sphere,
                            0.08 < lambda < 0.7, Re < 200 (Eq. 20; the review writes it in Re / 2, Wham's radius-based Re)

In the tube the same force moves the sphere at V_t = f V_inf, so X_tube(Re) = cheng(Re_inf) / f(Re_inf) with
Re = f(Re_inf) Re_inf (the review's Eqs. 12 and 15)."""
import math


def haberman_sayre(lam):
    return (1.0 - 0.75857 * lam ** 5) / (1.0 - 2.1050 * lam + 2.0865 * lam ** 3 - 1.7068 * lam ** 5 + 0.72603 * lam ** 6)


def cheng(re_inf):
    return (1.0 + 0.27 * re_inf) ** 0.43 + re_inf / 24.0 * 0.47 * (1.0 - math.exp(-0.04 * re_inf ** 0.38))


def kehlenbeck_difelice(re_inf, lam):
    q = 0.041 * re_inf ** 0.524
    lam0 = (0.283 + 1.2 * q) / (1.0 + q)
    p = 1.44 + 0.5466 * re_inf ** 0.434 if re_inf <= 35.0 else 2.3 + 37.3 * re_inf ** -0.8686
    return (1.0 - lam ** p) / (1.0 + (lam / lam0) ** p)


def kd_drag(re, lam):
    """X in the tube at the tube's Re by Kehlenbeck and Di Felice with Cheng: solve Re = f(Re_inf) Re_inf by bisection
    (monotone over the range), then X = cheng(Re_inf) / f. Returns (X, Re_inf, f)."""
    lo, hi = 1e-8, 1e5
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if kehlenbeck_difelice(mid, lam) * mid < re:
            lo = mid
        else:
            hi = mid
    r = math.sqrt(lo * hi)
    f = kehlenbeck_difelice(r, lam)
    return cheng(r) / f, r, f


def wham(re, lam):
    r2 = re / 2.0
    k = 0.6628 + 1.458 * math.exp(-0.05635 * r2)
    inertial = 1.0 + 0.03708 * r2 ** (1.514 - 0.1016 * math.log(r2)) if r2 > 0 else 1.0
    return inertial * (1.0 - 0.75857 * lam ** 5) / (1.0 - k * lam + 2.0865 * lam ** 3 - 1.7068 * lam ** 5 + 0.72603 * lam ** 6)


def table(lam=0.4, res=(1.0, 2.0, 10.0, 20.0, 50.0, 100.0, 150.0, 200.0)):
    rows = []
    for re in res:
        x, ri, f = kd_drag(re, lam)
        rows.append({"re": re, "kd_X": x, "kd_re_inf": ri, "kd_f": f, "wham_X": wham(re, lam), "cheng_unbounded_X": cheng(re),
                     "cd_kd": 24.0 / re * x, "cd_wham": 24.0 / re * wham(re, lam)})
    return {"lambda": lam, "haberman_sayre": haberman_sayre(lam), "kd_lambda_to_0_check": kd_drag(1e-3, 1e-6)[0], "rows": rows}


if __name__ == "__main__":
    import json
    print(json.dumps(table(), indent=1))
