"""Writes ``src/mime/data/sphere_plane_resistance.json``: the exact resistance of a sphere near a plane wall, at every
gap the near-contact law reads it (``mime.nodes.hydro.near_contact``), from the bispherical solutions.

**The solutions.** In bispherical coordinates (the plane at xi = 0, the sphere at xi = alpha, cosh alpha = 1 + h/a)
every harmonic function regular in the fluid and vanishing far off is a series of ``sqrt(cosh xi - mu) g_n(xi)
P_n^m(mu) cos(m phi)`` (or ``sin``), ``mu = cos eta``. The normal translation is Brenner's (1961) series and the spin
about the normal Jeffery's (1915), both summed here in closed form. The motions along the wall, a translation and a
rotation about an axis along it, are Dean and O'Neill's (1963) and O'Neill's (1964) problems, solved here in their
representation of a Stokes flow, ``u = x p / (2 mu) + w`` with ``p`` and the Cartesian components of ``w``
harmonic (``p`` and ``w_z`` of order m = 1, ``w``'s components along the wall of orders 0 and 2 with ``w_y``'s
coefficients ``w_x``'s), each a bispherical series to N terms; the coefficients are found by collocation, the no-slip
condition on both surfaces and ``div u = 0`` there (a harmonic function, so zero everywhere) at Gauss-Legendre nodes in
mu, by least squares, rather than by the papers' recurrences. Force and torque are the traction integrated over the
sphere (Gauss-Legendre in mu, the trapezoid rule in phi, exact for the orders the integrands hold).

**Checks the script records** (``checks``): the same solver's normal and spin against Brenner's and Jeffery's series;
the coupling computed both ways (the force per rotation and the torque per translation, reciprocal); the no-slip
condition at nodes that were not collocated; each knot's change when N and the nodes grow by a quarter (at the
smallest gaps, where the series is longest); and the traction's quadrature doubled.

**The table.** Five functions of the gap ``d = h / a``, for a unit sphere in a medium of unit viscosity, the force and
torque it exerts on the fluid per unit motion about its centre (MIME's resistance convention), the wall's normal +z:
``zz`` (normal translation), ``xx`` (translation along the wall), ``ryry`` (rotation about an axis along it), ``x_ry``
(the force along x per rotation about y, which is the torque about y per translation along x) and ``rzrz`` (spin
about the normal). The knots are uniform in ``ln d`` from ``D_LOW`` to ``D_HIGH``; the midpoints between them are
computed too, for the test that holds the interpolation to them (``tests/oracles/test_near_contact.py``).

**Reproducible bytes.** The file holds the numbers and nothing of the run that made them: each solve's time and the
environment's versions go to the build's log beside the parts (``sphere_plane_parts/log.json``), so that a rebuild can
match the shipped file's sha256. Its numbers are made in the processor's own kernels (LAPACK's, XLA's, NumPy's), which
differ between processors in their last bits: measured, two builds on GitHub's runners of different processors
differed in 64 of the 193 rows, by up to 2e-10 relative. The script pins them before it imports NumPy or JAX
(:data:`PINNED`: OpenBLAS's Haswell kernels, XLA's and NumPy's code no newer than AVX2), so that a rebuild on any
x86-64 processor with AVX2 makes the same bytes; ``tests/oracles/test_near_contact.py`` rebuilds a few knots and
compares them with the shipped file's.

Not collected; run once, in float64 (the smallest gaps take a few minutes each and about 4 GB):

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/sphere_plane_table.py part 0 1
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/sphere_plane_table.py checks
    python scripts/sphere_plane_table.py merge

``part i n`` builds every n-th of the knots and midpoints from the i-th into ``sphere_plane_parts/`` (one job's share
of n, the costly small gaps spread over the jobs); ``checks`` the checks; ``merge`` writes the table from the parts.
``knots i j ...`` prints the knots named, as the file holds them, for the test that compares a rebuild with it."""

from __future__ import annotations

import os

#: The kernels the numbers are made in, pinned before NumPy and JAX are imported (the module's docstring): OpenBLAS's
#: Haswell kernels, NumPy's dispatch no newer than AVX2, and XLA's code no newer than AVX2 on one thread (so that its
#: sums do not follow the processor's count of cores); XLA's flags are added to any the environment sets.
PINNED = {"OPENBLAS_CORETYPE": "Haswell", "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR"}
XLA_PINNED = ("--xla_cpu_max_isa=AVX2", "--xla_cpu_multi_thread_eigen=false")
os.environ.update(PINNED)
os.environ["XLA_FLAGS"] = " ".join([os.environ.get("XLA_FLAGS", ""),
                                    *(f for f in XLA_PINNED if f not in os.environ.get("XLA_FLAGS", ""))]).strip()

import json  # noqa: E402 - after the kernels are pinned
import math  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg  # noqa: E402

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "mime" / "data" / "sphere_plane_resistance.json"
PARTS = ROOT / "sphere_plane_parts"
#: The table's range of gaps, in the sphere's radius, and its knots (uniform in the gap's logarithm).
D_LOW, D_HIGH, KNOTS = 1e-3, 10.0, 97
#: The series' length: the coefficients decay as exp(-n alpha), so 40 / alpha terms leave them below float64's
#: rounding, and a margin.
TERMS_PER_ALPHA, TERMS_MARGIN = 40.0, 24
#: The angle the collocation nodes sit at (any but a multiple of pi / 4: the condition's three components there fix the
#: three profiles of a field of the problem's symmetry).
PHI = 0.7


def legendre(mu: jax.Array, last: int, m: int) -> jax.Array:
    """The orthonormal associated Legendre functions ``P_n^m(mu)``, n = m..last, by their three-term recurrence."""
    s = jnp.sqrt(1.0 - mu * mu)
    first = ((-1) ** m) * math.sqrt((2 * m + 1) / 2.0 * math.factorial(2 * m)) / (2 ** m * math.factorial(m)) * s ** m
    first = first * jnp.ones_like(mu)
    second = math.sqrt(2 * m + 3) * mu * first
    n = np.arange(m + 2, last + 1, dtype=np.float64)
    a = np.sqrt((4 * n ** 2 - 1) / (n ** 2 - m * m))
    b = np.sqrt(((n - 1) ** 2 - m * m) / (4 * (n - 1) ** 2 - 1))

    def step(carry: tuple[jax.Array, jax.Array], ab: tuple[jax.Array, jax.Array]) -> tuple[tuple, jax.Array]:
        before, last_ = carry
        an, bn = ab
        found = an * (mu * last_ - bn * before)
        return (last_, found), found

    _, rest = jax.lax.scan(step, (first, second), (jnp.asarray(a), jnp.asarray(b)))
    return jnp.concatenate([jnp.stack([first, second]), rest])


def sinh_ratio(x: jax.Array, y: jax.Array) -> jax.Array:
    """``sinh(x) / sinh(y)`` for ``0 <= x <= y``, without overflow."""
    return jnp.exp(x - y) * (-jnp.expm1(-2.0 * x)) / (-jnp.expm1(-2.0 * y))


class Bispherical:
    """The harmonic series about a unit sphere whose surface is ``d`` above the plane z = 0, to ``terms`` terms: each
    order's functions as two families, one equal to its Legendre function on the sphere and zero on the plane, the
    other the reverse (``sinh((n + 1/2) xi) / sinh((n + 1/2) alpha)`` and its mirror), so that no column outgrows the
    others."""

    def __init__(self, d: float, terms: int):
        self.alpha = math.acosh(1.0 + d)
        self.c = math.sinh(self.alpha)
        self.terms = terms
        self.centre = np.array([0.0, 0.0, math.cosh(self.alpha)])

    def harmonics(self, x: jax.Array, m: int, trig: str) -> jax.Array:
        X, Y, Z = x[0], x[1], x[2]
        c, alpha = self.c, self.alpha
        rho2 = X * X + Y * Y
        A = rho2 + Z * Z - c * c
        mu = A / jnp.sqrt(A * A + 4.0 * c * c * rho2)
        xi = 0.5 * jnp.log((rho2 + (Z + c) ** 2) / (rho2 + (Z - c) ** 2))
        k = jnp.arange(m, self.terms + 1, dtype=jnp.float64) + 0.5
        if m == 0:
            turn = 1.0
        elif m == 1:
            turn = (X if trig == "cos" else Y) / jnp.sqrt(rho2)
        else:
            turn = ((X * X - Y * Y) if trig == "cos" else 2.0 * X * Y) / rho2
        common = jnp.sqrt(jnp.cosh(xi) - mu) * legendre(mu, self.terms, m) * turn
        return jnp.concatenate([common * sinh_ratio(k * xi, k * alpha), common * sinh_ratio(k * (alpha - xi), k * alpha)])

    def fields(self, x: jax.Array, problem: str) -> tuple[jax.Array, jax.Array]:
        """The velocity (3, K) and pressure (K,) at ``x`` per unit coefficient, unit viscosity."""
        h = self.harmonics
        if problem == "along":  # p, w_z of order 1 (cos); w_x of order 0 and 2 (cos), w_y order 2 (sin), shared
            p, wz, w0, w2c, w2s = h(x, 1, "cos"), h(x, 1, "cos"), h(x, 0, "cos"), h(x, 2, "cos"), h(x, 2, "sin")
            o = jnp.zeros_like
            u = jnp.stack([jnp.concatenate([0.5 * x[0] * p, o(wz), w0, w2c]),
                           jnp.concatenate([0.5 * x[1] * p, o(wz), o(w0), w2s]),
                           jnp.concatenate([0.5 * x[2] * p, wz, o(w0), o(w2c)])])
            return u, jnp.concatenate([p, o(wz), o(w0), o(w2c)])
        if problem == "normal":  # p, w_z of order 0; w_x, w_y of order 1 (cos, sin), shared
            p, wz, w1c, w1s = h(x, 0, "cos"), h(x, 0, "cos"), h(x, 1, "cos"), h(x, 1, "sin")
            o = jnp.zeros_like
            u = jnp.stack([jnp.concatenate([0.5 * x[0] * p, o(wz), w1c]),
                           jnp.concatenate([0.5 * x[1] * p, o(wz), w1s]),
                           jnp.concatenate([0.5 * x[2] * p, wz, o(w1c)])])
            return u, jnp.concatenate([p, o(wz), o(w1c)])
        # spin: w = W (-sin phi, cos phi, 0) with W of order 1, no pressure
        w1c, w1s = h(x, 1, "cos"), h(x, 1, "sin")
        u = jnp.stack([-w1s, w1c, jnp.zeros_like(w1c)])
        return u, jnp.zeros_like(w1c)

    def rows(self, x: jax.Array, problem: str) -> jax.Array:
        """(4, K): the velocity's three components and its divergence at ``x`` per unit coefficient."""
        u = self.fields(x, problem)[0]
        jacobian = jax.jacfwd(lambda y: self.fields(y, problem)[0])(x)
        return jnp.concatenate([u, (jacobian[0, :, 0] + jacobian[1, :, 1] + jacobian[2, :, 2])[None, :]])

    def plane_points(self, mu: np.ndarray, phi: float) -> np.ndarray:
        r = self.c * np.sqrt((1.0 + mu) / (1.0 - mu))
        return np.stack([r * math.cos(phi), r * math.sin(phi), np.zeros_like(mu)], -1)

    def sphere_points(self, mu: np.ndarray, phi: float) -> np.ndarray:
        D = math.cosh(self.alpha) - mu
        r = self.c * np.sqrt(1.0 - mu * mu) / D
        return np.stack([r * math.cos(phi), r * math.sin(phi), self.c * math.sinh(self.alpha) / D], -1)


MOTIONS = {"along": ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
           "normal": ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)), "spin": ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))}


def solve(d: float, problem: str, terms: int | None = None, nodes: int | None = None,
          traction_nodes: int | None = None) -> dict:
    """The force and torque on the fluid per unit motion for ``problem`` at the gap ``d``, with the collocation's
    unmet condition at the nodes used and at nodes not used."""
    alpha = math.acosh(1.0 + d)
    terms = terms or int(math.ceil(TERMS_PER_ALPHA / alpha)) + TERMS_MARGIN
    nodes = nodes or terms + max(16, terms // 8)
    series = Bispherical(d, terms)
    mu, _ = np.polynomial.legendre.leggauss(nodes)
    assemble = jax.jit(jax.vmap(lambda x: series.rows(x, problem)))

    def system(mu_: np.ndarray, phi: float) -> tuple[np.ndarray, np.ndarray]:
        plane, sphere = series.plane_points(mu_, phi), series.sphere_points(mu_, phi)
        blocks = [np.asarray(assemble(jnp.asarray(chunk))) for points in (plane, sphere)
                  for chunk in np.array_split(points, max(1, len(points) // 128))]
        A = np.concatenate([block.reshape(-1, block.shape[2]) for block in blocks])
        motions = MOTIONS[problem]
        right = []
        for U, W in zip(motions[::2], motions[1::2], strict=True):
            on = np.asarray(U)[None, :] + np.cross(np.asarray(W)[None, :], sphere - series.centre)
            right.append(np.concatenate([np.zeros(4 * len(mu_)),
                                         np.concatenate([on, np.zeros((len(mu_), 1))], 1).reshape(-1)]))
        return A, np.stack(right, -1)

    started = time.time()
    A, B = system(mu, PHI)
    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0.0] = 1.0
    coefficients = scipy.linalg.lstsq(A / scale, B, lapack_driver="gelsy")[0] / scale[:, None]
    unmet = float(np.abs(A @ coefficients - B).max())
    other = np.polynomial.legendre.leggauss(nodes + 37)[0]
    A2, B2 = system(other, 1.3)
    unmet_elsewhere = float(np.abs(A2 @ coefficients - B2).max())
    del A, A2
    loads = [traction_load(series, problem, coefficients[:, j], traction_nodes or 2 * nodes)
             for j in range(coefficients.shape[1])]
    return {"terms": terms, "nodes": nodes, "unmet": unmet, "unmet_elsewhere": unmet_elsewhere, "loads": loads,
            "seconds": round(time.time() - started, 1)}


def held(solved: dict) -> dict:
    """What the file keeps of a solve: its numbers, not its time (which goes to the log)."""
    return {k: v for k, v in solved.items() if k not in ("loads", "seconds")}


def traction_load(series: Bispherical, problem: str, coefficients: np.ndarray, nodes: int) -> tuple[list, list]:
    """The force and torque the sphere exerts on the fluid (minus the traction's integral over it), about its
    centre."""
    a = jnp.asarray(coefficients)
    centre = jnp.asarray(series.centre)

    def velocity(x: jax.Array) -> jax.Array:
        return series.fields(x, problem)[0] @ a

    def traction(x: jax.Array) -> jax.Array:
        gradient = jax.jacfwd(velocity)(x)
        pressure = series.fields(x, problem)[1] @ a
        stress = -pressure * jnp.eye(3) + gradient + gradient.T
        return stress @ (x - centre)

    on = jax.jit(jax.vmap(traction))
    mu, weights = np.polynomial.legendre.leggauss(nodes)
    angles = 4
    D = math.cosh(series.alpha) - mu
    area = weights * (series.c / D) ** 2 * (2.0 * math.pi / angles)
    force, torque = np.zeros(3), np.zeros(3)
    for i in range(angles):
        points = series.sphere_points(mu, 2.0 * math.pi * i / angles + 0.1)
        found = np.concatenate([np.asarray(on(jnp.asarray(chunk))) for chunk in np.array_split(points, 8)])
        force += (found * area[:, None]).sum(0)
        torque += (np.cross(points - series.centre, found) * area[:, None]).sum(0)
    return (-force).tolist(), (-torque).tolist()


def brenner(d: float) -> float:
    """Brenner (1961): the normal resistance over ``6 pi mu a``, the series summed until its terms vanish."""
    alpha = math.acosh(1.0 + d)
    total = 0.0
    for n in range(1, int(60.0 / alpha) + 200):
        x = (2 * n + 1) * alpha
        if x >= 700.0:
            break
        sech = 2.0 * math.exp(-x) / (1.0 + math.exp(-2.0 * x))
        numerator = 2.0 * math.tanh(x) + (2 * n + 1) * math.sinh(2.0 * alpha) * sech
        denominator = 2.0 - 2.0 * sech - (2 * n + 1) ** 2 * math.sinh(alpha) ** 2 * sech
        total += n * (n + 1) / ((2 * n - 1) * (2 * n + 3)) * (numerator / denominator - 1.0)
    return 4.0 / 3.0 * math.sinh(alpha) * total


def jeffery(d: float) -> float:
    """Jeffery (1915): the spin about the normal's resistance over ``8 pi mu a^3``, ``sinh^3 alpha sum_n csch^3(n
    alpha)``."""
    alpha = math.acosh(1.0 + d)
    total = 0.0
    for n in range(1, int(60.0 / alpha) + 200):
        x = n * alpha
        if x >= 700.0:
            break
        total += (2.0 * math.exp(-x) / (1.0 - math.exp(-2.0 * x))) ** 3
    return math.sinh(alpha) ** 3 * total


def entry(d: float, **settings: int) -> dict:
    """The table's five functions at ``d`` and what each solve reports."""
    along = solve(d, "along", **settings)
    (force_u, torque_u), (force_w, torque_w) = along["loads"]
    return {"d": d, "zz": 6.0 * math.pi * brenner(d), "xx": force_u[0], "ryry": torque_w[1], "x_ry": force_w[0],
            "ry_x": torque_u[1], "rzrz": 8.0 * math.pi * jeffery(d), "solve": held(along),
            "seconds": along["seconds"]}


def knots() -> np.ndarray:
    return np.exp(np.linspace(math.log(D_LOW), math.log(D_HIGH), KNOTS))


def midpoints() -> np.ndarray:
    t = np.linspace(math.log(D_LOW), math.log(D_HIGH), KNOTS)
    return np.exp(0.5 * (t[1:] + t[:-1]))


def checks() -> dict:
    """The solver against the closed-form series, and its convergence at the smallest gap."""
    found: dict = {"series": {}, "growth": {}}
    for d in (0.3, 0.03, 0.003):
        normal, spin = solve(d, "normal"), solve(d, "spin")
        found["series"][repr(d)] = {
            "normal_over_brenner": normal["loads"][0][0][2] / (6.0 * math.pi * brenner(d)) - 1.0,
            "spin_over_jeffery": spin["loads"][0][1][2] / (8.0 * math.pi * jeffery(d)) - 1.0,
            "unmet": max(normal["unmet_elsewhere"], spin["unmet_elsewhere"])}
        print("series", d, found["series"][repr(d)], flush=True)
    for d in (D_LOW, 0.01):
        alpha = math.acosh(1.0 + d)
        terms = int(math.ceil(TERMS_PER_ALPHA / alpha)) + TERMS_MARGIN
        base, grown = entry(d), entry(d, terms=int(1.25 * terms), nodes=int(1.25 * (terms + max(16, terms // 8))))
        doubled = solve(d, "along", traction_nodes=4 * (terms + max(16, terms // 8)))["loads"]
        found["growth"][repr(d)] = {
            "terms_and_nodes_x1.25": {k: grown[k] / base[k] - 1.0 for k in ("xx", "ryry", "x_ry")},
            "traction_nodes_x2": {"xx": doubled[0][0][0] / base["xx"] - 1.0, "ryry": doubled[1][1][1] / base["ryry"] - 1.0,
                                  "x_ry": doubled[1][0][0] / base["x_ry"] - 1.0}}
        print("growth", d, found["growth"][repr(d)], flush=True)
    return found


def environment() -> dict:
    """The run's environment, for the build's log (never the file): the versions, the processor and the kernels
    pinned."""
    model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() else \
        platform.processor()
    return {"python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__,
            "scipy": scipy.__version__, "processor": model,
            "pinned": {name: os.environ.get(name) for name in (*PINNED, "XLA_FLAGS")}}


def build_part(share: int, shares: int) -> None:
    points = [("knot", i, float(d)) for i, d in enumerate(knots())] + \
        [("midpoint", i, float(d)) for i, d in enumerate(midpoints())]
    points.sort(key=lambda point: point[2])
    PARTS.mkdir(exist_ok=True)
    log = {"environment": environment(), "seconds": {}}
    for kind, i, d in points[share::shares]:
        found = entry(d)
        log["seconds"][f"{kind}_{i:03d}"] = found.pop("seconds")
        (PARTS / f"{kind}_{i:03d}.json").write_text(json.dumps(found))
        print(kind, i, d, {k: found[k] for k in ("xx", "x_ry", "ry_x", "ryry")}, found["solve"], flush=True)
    (PARTS / f"log_{share:02d}_of_{shares:02d}.json").write_text(json.dumps(log, indent=1))


def merge() -> None:
    def read(kind: str, count: int) -> list[dict]:
        return [json.loads((PARTS / f"{kind}_{i:03d}.json").read_text()) for i in range(count)]

    held = read("knot", KNOTS)
    middle = read("midpoint", KNOTS - 1)
    worst = max(abs(row["x_ry"] / row["ry_x"] - 1.0) for row in held + middle)
    names = ("zz", "xx", "ryry", "x_ry", "rzrz")
    record = {
        "description": "the exact resistance of a unit sphere near a plane wall in a medium of unit viscosity: the force "
                       "and torque it exerts on the fluid per unit motion about its centre, the wall's normal +z, at the "
                       "gap d (its surface's height over the radius); from the bispherical solutions (Brenner 1961, "
                       "Jeffery 1915, Dean and O'Neill 1963, O'Neill 1964), built by scripts/sphere_plane_table.py",
        "entries": {"zz": "normal translation", "xx": "translation along the wall (x, as y)",
                    "ryry": "rotation about an axis along the wall (y, as x)",
                    "x_ry": "force along x per rotation about y (= torque about y per translation along x; "
                            "the y-force per rotation about x is its negative)",
                    "rzrz": "spin about the normal"},
        "range": [D_LOW, D_HIGH], "knots": KNOTS,
        "d": [row["d"] for row in held], **{name: [row[name] for row in held] for name in names},
        "midpoints": {"d": [row["d"] for row in middle], **{name: [row[name] for row in middle] for name in names}},
        "checks": {**json.loads((PARTS / "checks.json").read_text()), "reciprocity_worst": worst,
                   "unmet_elsewhere_worst": max(row["solve"]["unmet_elsewhere"] for row in held + middle)},
        "solves": [row["solve"] for row in held + middle],
    }
    OUT.write_text(json.dumps(record, indent=1) + "\n")
    logs = {path.name: json.loads(path.read_text()) for path in sorted(PARTS.glob("log_*.json"))}
    (PARTS / "log.json").write_text(json.dumps({"merged_by": environment(), "parts": logs}, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["merge"]:
        merge()
    elif sys.argv[1:] == ["checks"]:
        PARTS.mkdir(exist_ok=True)
        (PARTS / "checks.json").write_text(json.dumps(checks(), indent=1))
        (PARTS / "log_checks.json").write_text(json.dumps({"environment": environment()}, indent=1))
    elif sys.argv[1:2] == ["part"]:
        build_part(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1:2] == ["knots"]:
        at = knots()
        rebuilt = {}
        for i in map(int, sys.argv[2:]):
            found = entry(float(at[i]))
            rebuilt[str(i)] = {"d": found["d"], **{name: found[name] for name in ("zz", "xx", "ryry", "x_ry", "rzrz")},
                               "solve": found["solve"]}
        print(json.dumps(rebuilt))
    else:
        raise SystemExit(__doc__)
