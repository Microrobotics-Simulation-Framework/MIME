# Proposed figure captions

Draft captions for the five figures in this folder. Written to be self-contained, so a reader can
understand each figure from its caption alone.

---

**Figure A (`mod1_figA_designs.png`). All 17 de Jongh devices sit deep in the underdamped regime.**
Damping ratio ζ = C / (2√(I·mB)) for each of the 17 real de Jongh screw designs, on a logarithmic
axis. The FL group (green, 11 designs) varies the thread pitch at fixed length and radius; the FW
group (purple, 6 designs) varies the length. Every device falls in a narrow band between about 0.013
and 0.017, which is more than an order of magnitude below the critical value ζ_c = π/8 ≈ 0.39 (dashed
line). Below that value the step-out shows hysteresis (shaded region). Because the entire real device
family sits so far inside the underdamped regime, the family on its own cannot show the crossover to
the overdamped case; the scaling argument in Figure B provides that.

---

**Figure B (`mod1_figB_scaling.png`). The step-out window as a function of damping and of size.**
Two views of the same result.

(a) The two step-out thresholds, each divided by the natural frequency ω_n, as a function of the
damping ratio ζ. The black line is the analytic pull-out threshold (loss of lock as the drive is
raised), f_so/ω_n = 1/(2ζ). The gray line is the analytic pull-in floor (re-lock as the drive is
lowered), f_si/ω_n = 4/π. Blue circles and orange squares are the simulated values and follow the
analytic curves closely. The two thresholds meet at ζ_c = π/8 (green dashed line), where the
hysteresis window closes. The top axis shows the equivalent quality factor Q = 1/(2ζ), which links
the plot to the Josephson-junction literature. The orange band marks where the 17 real devices sit,
all at small ζ and large Q. The label at lower right notes that Fazeli et al. (2023) measured the
opposite limit for nanoswimmers, with f_so far below f_n, which is the overdamped endpoint of the
same curve.

(b) The same two thresholds in absolute units, as a function of body length L. Under isometric
scaling the pull-out threshold f_so = mB/C does not depend on size (flat black line), while the
pull-in floor f_si = (4/π)ω_n rises as the body shrinks (gray line). The two cross near L ≈ 302 µm
(green box). Below that length the system is overdamped and has a single step-out threshold (gray
region); above it the hysteresis band opens (blue region). Markers are the analytic thresholds for
the real devices (up triangles f_so, down triangles f_si; green FL, purple FW). The 11 FL designs
all have L = 7.47 mm, so their markers overlap into one stack (green label). The FW designs vary
length, so their markers spread out, and the purple dashed guides trace that trend: shorter screws
have less inertia and less drag (I and C both scale with L), so both thresholds rise. The real
devices sit near, but not exactly on, the isometric lines, because varying length at fixed radius is
not an isometric scaling.

---

**Figure C (`mod1_figC_coupled.png`). Coupled step-out loops for three geometries.**
Full two-scale simulations (an inertial body, the confined near field, and the vessel far field,
solved together) of the body spin against the drive frequency, taken up in frequency (blue) and then
back down (orange). The dashed diagonal is perfect synchronisation, where the spin equals the drive.
Each panel is one geometry. For the two elongated screws, FL-9 and FW-1, the body tracks the drive up
to a wobble instability near 230 Hz and 215 Hz (dotted blue line), then loses synchrony; on the way
down it stays tumbled and does not re-lock anywhere in the swept range, so the running state is sticky
and the hysteresis is wide. The vertical dashed orange line marks the pull-in floor f_si predicted by
the reduced model, and the shaded band is the resulting bistable window between that floor and the
coupled step-out. The short, stubby screw FW-6 behaves differently: it barely wobbles and tracks the
drive all the way to the 260 Hz ceiling, so it shows no step-out and no hysteresis in this range. The
comparison shows that the coupled step-out is set by the transverse wobble instability, which depends
on how elongated the screw is, and not by the axial pull-out of the reduced model.

---

**Figure D (`mod1_figD_regime_map.png`). Regime map and validation of the step-out equations.**
The two step-out thresholds, each divided by the natural frequency, as a function of the damping ratio
(both axes logarithmic). The black line is the pull-out threshold f_so = ω_n/(2ζ) = mB/C. The orange
line is the pull-in threshold written as one expression, f_si = ω_n·min(1/(2ζ), 4/π); it equals the
(4/π)ω_n floor while the system is underdamped and merges with the pull-out line once ζ reaches π/8.
The shaded band between the two lines is the hysteresis window, which closes at ζ_c = π/8 (green
dashed line). Circles and squares are the simulated pull-out and pull-in values and confirm both
expressions. For ζ at or above π/8 the two thresholds coincide, and the result reduces to the standard
single-valued overdamped step-out (right side of the plot). The top axis gives the quality factor
Q = 1/(2ζ) and the Josephson parameter β_c = Q². The 17 real devices (orange band) sit near β_c ≈ 1000,
which is far inside the underdamped regime, so the exact location of the boundary does not affect them.

---

**Figure E (`mod1_figE_wall_ratio.png`). The wall shifts the upper threshold but not the floor.**
The two thresholds as a function of the confinement ratio R_ves/R_cyl (the vessel radius divided by
the body radius), with the FL-9 body held fixed. The wall enters the physics only through the
rotational drag C (purple curve, right axis), which rises as the tube tightens. Since the pull-out
threshold is f_so = mB/C, it moves with the wall (black curve, falling from about 826 Hz in near-free
space to about 398 Hz in the tightest tube). The pull-in floor f_si = (4/π)ω_n has no drag term, so it
stays fixed at about 25 Hz across the whole range (orange line). The green dashed line marks the
1/4-inch tube used in this work (ratio 2.035). The practical point is that the lower threshold, which
is the one that matches the measured de Jongh value, does not depend on the vessel, while the upper
threshold does.
