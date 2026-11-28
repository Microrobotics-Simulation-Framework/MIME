# r39 cases: the shared definition (geometry, bodies, poses, flows, problems)

Written by r39 (2026-11-28). `geom.py` is the definition (numpy only); `occ_build.py` builds the same shapes in Netgen-OCC;
`check_cases.py` checks one against the other. Everything here is declared, not fitted. Units: lengths in a; mu = 1; velocities
in U_m (the vessel's mean speed). Flow is along +x; the wall the body is pressed towards is on -z; y = 0 is a mirror plane of
every wall.

## Bodies (body frame; H and F have their axis along z)
| | definition | notes |
|---|---|---|
| SPH | sphere, radius 1 | |
| H | points within 0.2 of the helix X(t) = (0.8 cos t, 0.8 sin t, 1.2 t/2pi - 1.5), t in [0, 5 pi] | w35m's H; 2.5 turns, right-handed, hemispherical ends, outer radius 1.0; no head |
| F | core (points within 0.3 of the axis segment abs(z) <= 1.5) united with three fins: for abs(z) <= 1.5, rotate (x, y) by -th_k(z), th_k = 2 pi k/3 + pi z/3; the planar stadium {abs(v) <= 0.04, 0 <= u <= 0.96} U disc((0.96, 0), 0.04) | w35m's twist (pi over length 3, tip radius 1.0); the fin's in-plane half thickness is 0.04; flat fin ends |

Placement in G1-G3: body z -> lab x (rotation y by +90 degrees after a spin about the body axis), centre c. The reference
point for torques is c.

## Walls (signed distance, positive in the fluid)
| | definition | parameters |
|---|---|---|
| G0 | plane z = 0 | anchor; container: cylinder radius L_c, height L_c, no slip |
| G0b | fixed sphere radius b centred (0, 0, -b) | b = 4 (REFUTE_W35's bend bulge radius); container sphere radius L_c |
| G1 | tube sqrt(y^2 + z^2) < R | SPH: R = 1/0.386 = 2.59067 (W7REF's tube); H, F: R = 2; extra SPH cross-check at R = 2 (Caldag et al. 2022's a/R 0.5) |
| G2 | tube of radius R round a centreline: straight inlet of length 4R, a 90-degree arc of radius R_c = 3R (centre of curvature (0, 0, -R_c), arc phi in [-45, 45] deg, phi = atan2(x, z + R_c)), straight outlet of length 12R | body at phi = 0 (the origin), pressed towards the inner wall z = -R; inner-wall bulge radius along the flow R_c - R = 2R |
| G3 | surface of revolution about x, radius R(x): throat R_s at x = 0, axial curvature radius rho_s (circle centre (0, R_s + rho_s)), joined tangentially to r = R by concave arcs of radius rho_j centred (+-x_j, R - rho_j), x_j^2 = 2 d (rho_s + rho_j) - d^2, d = R - R_s | SPH: R 2.59067, R_s = 0.9 R, rho_s = rho_j = R. H, F: R = 2, R_s = 1.5, rho_s = rho_j = 2. Inlet/outlet 4R / 12R from the throat |
| G4 | parent C (radius R, x <= 0) closed by a bulb (sphere radius R at the origin); daughters A, B (radius R_d) along (cos th, 0, +-sin th) from the origin; the carina (the A-B crease) rounded by a rolling ball of radius r_c: lumen = C u bulb u closing_rc(A u B); wall function = max of the three pieces' signed distances | R 2.59067, R_d = 0.76 R, th = 40 deg, r_c = 0.15 R; inlet 4R, daughters 8R. The daughter-bulb circles stay sharp ridges. The parameters satisfy R sin th < R_d (carina outside the bulb) and R_d <= R cos th (no daughter pokes over the parent) [derived] |

**G4 poses (SPH):**
- P1 (approaching the divider face): centre on the x axis, nearest wall point the rounded apex x = (R_d + r_c)/sin th - r_c.
  For gaps up to 0.1a the apex is the single nearest point (the arch's wall radius (R_d + r_c) sin th - r_c = 0.435 R exceeds
  a + h) [derived; sampled check].
- P2 (flank): nearest point on the carina fillet 30 degrees round from the apex towards A, in y = 0; centre on that normal.
- P3 (entering A): start at 1.5 R along A's axis, pressed towards A's carina-side wall.

## Gaps and windows
- Gaps h in {0.1, 0.01, 0.001} a; the centre is placed so that the minimum wall distance of the body surface is exactly h
  (closed form for SPH; a 1-D search along the centreline for H; a 2-D search on the fin tips for F).
- Stokes and Brinkman disturbances: the window is the declared domain (G2-G4) or +-6R about the body (G1), with zero
  disturbance on its ends; the disturbance decays exponentially along a tube.

## Flows and problems (also in work/PLAN.md section 2)
- G1 background: Poiseuille u_bg = 2 (1 - (y^2 + z^2)/R^2) e_x (U_m = 1). G2-G4: the body-free Stokes flow of the declared
  domain, Poiseuille at the inlet, do-nothing outlets.
- S (6x6): s u - div(2 eps(u)) + grad p = 0, div u = 0, u = rigid twist on the body, 0 on every other boundary; s in
  {0, 130, 1381}. Force on the body F = -S U.
- F_0 (Stokes): the body held in u_bg. F_0 (Brinkman s): one step of the step operator from u^n = u_bg with the body held,
  i.e. the disturbance problem s v - div(2 eps(v)) + grad q = 0, v = -u_bg on the body, 0 elsewhere.
- Finite Re: steady Navier-Stokes; each row of the reference tables names its frame (body frame for translation along a
  straight tube; frozen geometry in the lab frame otherwise).

## Checks (check_cases.py; results in work/check_cases*.jsonl, work/dbg_*.out)
- Mesh boundary vertices against the analytic wall function: G0, G0b, G1, G2, G3 at 1e-14 [measured].
- Sphere gaps: the closed form against the minimum over 6e4 to 2e5 surface samples, every geometry and G4 pose [measured].
- G4: the cylinders and the bulb are exact in OCC (7e-7); OCC's fillet faces (named `carina`) deviate from the analytic rolling
  ball by up to 5.1e-4 a, median 2.2e-4 a, at the apex [measured]. The solver moves the `carina` faces onto the analytic surface
  (code/exactgeo.py, a harmonic deformation of the curved mesh). A core that uses OCC's fillet as its wall inherits that error.
- H: OCC's Pipe surface deviates from the exact tube by up to 1.5e-5 a [measured]; the solver moves the `body` faces onto the
  analytic tube in the same way. The OCC loft alternative was too slow to build (over 15 min for 121 sections).
- F: OCC's Pipe twist is not implemented in netgen 6.2.2608; a 49-section loft of the stadium was not checked; F has no
  verified mesh (see the report).
- Construction note (G4): fillet the A-B crease of the two daughter cylinders alone, then unite with the parent and bulb.
  Filleting the crease on the full union gave a solid whose later booleans were wrong (lumen minus sphere 542.6 against
  703.2) [measured].
