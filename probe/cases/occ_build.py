"""r39 cases: Netgen-OCC builders for the analytic definitions in geom.py (requires netgen-mesher from PyPI).

Face names: wall, inlet, outlet (G4: outletA, outletB), body, outer (anchor containers), symx / symy / symz (mirror planes
x = xc, y = 0, z = 0). Every builder returns the lumen solid; fluid() subtracts the body and applies the mirror cuts."""
import numpy as np
from netgen.occ import (ArcOfCircle, Pnt, Vec, Dir, Axis, Axes, Cylinder, Sphere, Box, Face, Wire, Circle, Segment, Edge, Pipe,
                        WorkPlane, X, Y, Z, Glue, ThruSections)


def _P(v):
    return Pnt(float(v[0]), float(v[1]), float(v[2]))


def _D(v):
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    return Dir(float(v[0]), float(v[1]), float(v[2]))


def lumen_G0(Lc):
    c = Cylinder(Pnt(0, 0, 0), Z, Lc, Lc)
    c.faces.name = "outer"
    c.faces.Min(Z).name = "wall"
    return c


def lumen_G0b(b, Lc):
    box = Sphere(Pnt(0, 0, 0), Lc)
    box.faces.name = "outer"
    s = Sphere(Pnt(0, 0, -b), b)
    s.faces.name = "wall"
    return box - s


def lumen_G1(R, x0, x1):
    c = Cylinder(Pnt(x0, 0, 0), X, R, x1 - x0)
    c.faces.name = "wall"
    c.faces.Min(X).name = "inlet"
    c.faces.Max(X).name = "outlet"
    return c


def _circle_face(centre, normal, r):
    return Face(Wire([Circle(_P(centre), _D(normal), r)]))


def lumen_G2(g, L_in=None, L_out=None, arc_only=False):
    """straight inlet + 90-degree torus section (revolved circle) + straight outlet."""
    L_in = g.L_in if L_in is None else L_in
    L_out = g.L_out if L_out is None else L_out
    disc = _circle_face(g.P_in, g.t_in, g.R)
    ang = np.degrees(2 * g.ph)
    tor = disc.Revolve(Axis(_P(g.C), Dir(0, -1, 0)), ang)
    if abs(tor.center.z - (-g.Rc + g.Rc * np.sin(g.ph) / g.ph)) > 1e-6 * g.Rc:     # revolved the wrong way
        tor = disc.Revolve(Axis(_P(g.C), Dir(0, 1, 0)), ang)
    parts = [tor]
    if L_in > 0:
        parts.append(Cylinder(_P(g.P_in - L_in * g.t_in), _D(g.t_in), g.R, L_in))
    if L_out > 0:
        parts.append(Cylinder(_P(g.P_out), _D(g.t_out), g.R, L_out))
    sol = parts[0]
    for p in parts[1:]:
        sol = sol + p
    sol.faces.name = "wall"
    # ends: planar faces perpendicular to the centreline
    e_in = g.P_in - L_in * g.t_in
    e_out = g.P_out + L_out * g.t_out
    for f in sol.faces:
        c = np.array([f.center.x, f.center.y, f.center.z])
        if np.linalg.norm(c - e_in) < 1e-6 * g.R:
            f.name = "inlet"
        if np.linalg.norm(c - e_out) < 1e-6 * g.R:
            f.name = "outlet"
    return sol


def _xy(x, y):
    return Pnt(float(x), float(y), 0.0)


def _arc3(O, r, A, B):
    m = _arc_mid(O, r, A, B)
    return ArcOfCircle(_xy(*A), _xy(*m), _xy(*B))


def lumen_G3(g, x0, x1):
    """revolve the arc profile about x; the profile is drawn in the (x, y = r) half plane."""
    xt, rt = g.T
    A0, A1, A2 = (x0, 0.0), (x0, g.R), (-g.xj, g.R)
    T1, T2, B1, B2, B3 = (-xt, rt), (xt, rt), (g.xj, g.R), (x1, g.R), (x1, 0.0)
    edges = [Segment(_xy(*A0), _xy(*A1)), Segment(_xy(*A1), _xy(*A2)),
             _arc3(np.array([-g.xj, g.R - g.rj]), g.rj, np.array(A2), np.array(T1)),
             _arc3(g.Os, g.rs, np.array(T1), np.array(T2)),
             _arc3(g.Oj, g.rj, np.array(T2), np.array(B1)),
             Segment(_xy(*B1), _xy(*B2)), Segment(_xy(*B2), _xy(*B3)), Segment(_xy(*B3), _xy(*A0))]
    face = Face(Wire(edges))
    sol = face.Revolve(Axis((0, 0, 0), X), 360)
    sol.faces.name = "wall"
    sol.faces.Min(X).name = "inlet"
    sol.faces.Max(X).name = "outlet"
    return sol


def _arc_mid(O, r, A, B):
    """a point on the shorter arc of circle (O, r) between A and B, for ArcTo(x, y, end)."""
    a = np.arctan2(A[1] - O[1], A[0] - O[0])
    b = np.arctan2(B[1] - O[1], B[0] - O[0])
    d = (b - a + np.pi) % (2 * np.pi) - np.pi
    m = a + 0.5 * d
    return (float(O[0] + r * np.cos(m)), float(O[1] + r * np.sin(m)))


def lumen_G4(g, fillet=True):
    # parent + bulb as one revolved solid: (x, r) profile line to (0, R) then a quarter circle to (R, 0)
    q = g.R / np.sqrt(2)
    edges = [Segment(_xy(-g.L_in, 0), _xy(-g.L_in, g.R)), Segment(_xy(-g.L_in, g.R), _xy(0, g.R)),
             ArcOfCircle(_xy(0, g.R), _xy(q, q), _xy(g.R, 0)), Segment(_xy(g.R, 0), _xy(-g.L_in, 0))]
    pb = Face(Wire(edges)).Revolve(Axis((0, 0, 0), X), 360)
    pb.faces.name = "wall"
    cA = Cylinder(Pnt(0, 0, 0), _D(g.eA), g.Rd, g.L_d)
    cB = Cylinder(Pnt(0, 0, 0), _D(g.eB), g.Rd, g.L_d)
    cA.faces.name = "wall"; cB.faces.name = "wall"
    AB = cA + cB
    if fillet:
        # closing_rc(A u B): fillet the A-B crease (in z = 0) of the daughters alone, then unite (geom.Bifurcation, carina).
        # The fillet faces (new, unnamed) are named "carina": OCC approximates them (measured 5e-4 a), so the solver moves
        # them onto the analytic surface (code/exactgeo.py).
        edges = [e for e in AB.edges if abs(e.center.z) < 1e-6 * g.R and e.center.x > 0.3 * g.R]
        AB = AB.MakeFillet(edges, g.rc)
        for f in AB.faces:
            if f.name != "wall":
                f.name = "carina"
    sol = pb + AB
    for f in sol.faces:
        c = np.array([f.center.x, f.center.y, f.center.z])
        if abs(c[0] + g.L_in) < 1e-6 * g.R:
            f.name = "inlet"
        if np.linalg.norm(c - g.L_d * g.eA) < 1e-6 * g.R:
            f.name = "outletA"
        if np.linalg.norm(c - g.L_d * g.eB) < 1e-6 * g.R:
            f.name = "outletB"
    return sol


# ---------------------------------------------------------------- bodies
def body_sphere(c):
    s = Sphere(_P(c), 1.0)
    s.faces.name = "body"
    return s


def _place(shape, c, Rm):
    """body frame -> lab: rotate by Rm (via axis-angle) then translate by c."""
    Rm = np.asarray(Rm, float)
    ang = np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))
    if ang > 1e-14:
        w = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
        if np.linalg.norm(w) < 1e-12:          # 180 degrees
            vals, vecs = np.linalg.eigh(0.5 * (Rm + Rm.T))
            w = vecs[:, np.argmax(vals)]
        shape = shape.Rotate(Axis((0, 0, 0), _D(w)), float(np.degrees(ang)))
    return shape.Move(Vec(float(c[0]), float(c[1]), float(c[2])))


def body_helix(c, Rm, H, method="loft", nsec=241):
    """method 'pipe': OCC Pipe of a circle along the exact helix edge (measured deviation 1.5e-5 a);
    'loft': ThruSections through nsec exact circles normal to the helix (cubic loft)."""
    p0 = H.X(np.array([0.0]))[0]
    p1 = H.X(np.array([H.t1]))[0]
    if method == "pipe":
        cyl = Cylinder(Pnt(0, 0, -1.5), Z, H.rh, 3.0)
        mantle = max(cyl.faces, key=lambda f: f.mass)
        heli = Edge(Segment((0, 0), (H.t1, H.pitch * H.t1 / (2 * np.pi))), mantle)
        tube = Pipe(Wire([heli]), _circle_face(p0, H.dX(np.array([0.0]))[0], H.rho))
    else:
        ts = np.linspace(0, H.t1, nsec)
        wires = [Wire([Circle(_P(H.X(np.array([t]))[0]), _D(H.dX(np.array([t]))[0]), H.rho)]) for t in ts]
        tube = ThruSections(wires, True)
    body = tube + Sphere(_P(p0), H.rho) + Sphere(_P(p1), H.rho)
    body.faces.name = "body"
    return _place(body, c, Rm)


def body_fins(c, Rm, F, nsec=None):
    core = Cylinder(Pnt(0, 0, -F.zh), Z, F.rcore, 2 * F.zh) + Sphere(Pnt(0, 0, -F.zh), F.rcore) + Sphere(Pnt(0, 0, F.zh), F.rcore)
    body = core
    nsec = 49 if nsec is None else nsec
    for k in range(3):
        wires = []
        for zz in np.linspace(-F.zh, F.zh, nsec):
            a0 = F.th(zz, k)
            ca, sa = np.cos(a0), np.sin(a0)
            P2 = lambda u, v: Pnt(float(ca * u - sa * v), float(sa * u + ca * v), float(zz))
            wires.append(Wire([Segment(P2(0, -F.t), P2(F.L, -F.t)), ArcOfCircle(P2(F.L, -F.t), P2(F.L + F.t, 0), P2(F.L, F.t)),
                               Segment(P2(F.L, F.t), P2(0, F.t)), Segment(P2(0, F.t), P2(0, -F.t))]))
        body = body + ThruSections(wires, True)
    body.faces.name = "body"
    return _place(body, c, Rm)


# ---------------------------------------------------------------- fluid with mirror cuts
def fluid(lumen, body, cuts=(), big=1e3):
    """cuts: subset of {"y", "x:<xc>", "z"}: keep y >= 0, x >= xc, z >= 0; the new planar faces are named symy, symx, symz."""
    f = lumen - body if body is not None else lumen
    for cut in cuts:
        if cut == "y":
            b = Box(Pnt(-big, 0, -big), Pnt(big, big, big)); b.faces.name = "symy"
        elif cut == "z":
            b = Box(Pnt(-big, -big, 0), Pnt(big, big, big)); b.faces.name = "symz"
        elif cut.startswith("x:"):
            xc = float(cut[2:])
            b = Box(Pnt(xc, -big, -big), Pnt(big, big, big)); b.faces.name = "symx"
        f = f * b
    return f
