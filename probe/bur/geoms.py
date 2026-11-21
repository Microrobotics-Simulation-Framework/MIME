"""Bodies for the gel step: a (lobed) rotor in a clamped disc, and a thread unrolled to a wavy wall under a clamped lid."""
import numpy as np


def rotor(L, R0, Ro, omega, amp=0.0, lobes=0, ramp=0.0):
    cx = cy = 0.5 * L
    def theta(t):
        return omega * t if ramp <= 0 else omega * (t - ramp * (1 - np.exp(-t / ramp)))
    def rate(t):
        return omega if ramp <= 0 else omega * (1 - np.exp(-t / ramp))
    def geom(t):
        th, om = theta(t), rate(t)
        def phi(x, y):
            r = np.hypot(x - cx, y - cy)
            if amp == 0.0:
                return r - R0
            a = lobes * (np.arctan2(y - cy, x - cx) - th)
            return (r - R0 - amp * np.cos(a)) / np.sqrt(1 + (amp * lobes * np.sin(a) / np.maximum(r, 1e-12)) ** 2)
        vel = (lambda x, y: -om * (y - cy), lambda x, y: om * (x - cx), lambda x, y: 0.0 * x)
        return dict(phi=phi, vel=vel, clamp=lambda x, y: np.hypot(x - cx, y - cy) > Ro, centre=(cx, cy), theta=th)
    return geom


def thread(lam, y0, height, power, Vn, Vs, ytop, ramp=0.0):
    """The wall's surface y = y0 + height ((1 + cos 2 pi (x - s)/lam)/2)^power, moving at Vn along x and Vs out of the plane."""
    def fac(t):
        return 1.0 if ramp <= 0 else 1 - np.exp(-t / ramp)
    def pos(t):
        return Vn * t if ramp <= 0 else Vn * (t - ramp * (1 - np.exp(-t / ramp)))
    def geom(t):
        sft, fc = pos(t), fac(t)
        def phi(x, y):
            a = 2 * np.pi * (x - sft) / lam
            c = 0.5 * (1 + np.cos(a))
            ys = y0 + height * c ** power
            dys = height * power * c ** (power - 1) * (-0.5 * np.sin(a)) * 2 * np.pi / lam if power > 0 else 0.0
            return (y - ys) / np.sqrt(1 + dys ** 2)
        vel = (lambda x, y: Vn * fc + 0 * x, lambda x, y: 0.0 * x, lambda x, y: Vs * fc + 0 * x)
        return dict(phi=phi, vel=vel, clamp=lambda x, y: y > ytop, centre=(0.0, 0.0))
    return geom
