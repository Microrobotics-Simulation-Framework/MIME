import sys, time, numpy as np
from gel import Grid, Gel
from geoms import rotor
N = int(sys.argv[1]); steps = int(sys.argv[2]); dth = float(sys.argv[3])
L, R0, Ro, G = 8.0, 1.2, 3.5, 1000.0      # mm, Pa -> forces in mPa mm^2... units arbitrary but consistent
g = Grid(N, N, L / N)
om = 1.0; dt = dth / om
gel = Gel(g, G, rotor(L, R0, Ro, om), dt)
Kex = 4 * np.pi * G * R0**2 * Ro**2 / (Ro**2 - R0**2)
t0 = time.time()
for k in range(steps):
    o = gel.step()
    th = om * o["t"]
    if k % max(1, steps // 8) == 0 or k == steps - 1:
        print(k, "theta %.4f  T/Texact %.5f  Fx %.2e  kappa %.2f" % (th, -o["Tz"] / (Kex * th), o["Fx"], o["kappa_max"]), flush=True)
print("s/step", (time.time() - t0) / steps)
