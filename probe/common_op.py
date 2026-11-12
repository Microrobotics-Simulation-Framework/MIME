"""Shared pieces of the review's operating-point probe: Landers' 1.4 mm sphere (a = 0.7 mm) as MIME's 338-point MRS
sphere scaled from the oracle's 1 mm, rule 2's fine cell a / 20, stage 1's corners."""
import sys
sys.path[:0] = ["tests", "tests/patch"]
import numpy as np
import scipy.linalg as sla

from lubrication_cases import RADIUS
from patch_cases import mrs_sphere

DENSITY = 1260.0
A = 0.7e-3
#: (viscosity Pa s, shift rho/dt): the steady limit, lambda 4.97 (50 mPa s, 0.5 ms), lambda 0.79 (200 mPa s, 5 ms).
REGIMES = {"steady": (0.05, 0.0), "l497": (0.05, DENSITY / 5e-4), "l079": (0.2, DENSITY / 5e-3)}


def sphere(viscosity, direction=2, speed=1e-3):
    """The scaled sphere in its own frame: force points, quadrature points, weights, owners, areas, eps^2, and the cell
    forces (N) of a free translation at ``speed`` along ``direction``."""
    held = mrs_sphere(0.25)
    s = A / RADIUS
    velocity = np.zeros(3); velocity[direction] = speed
    traction = sla.lu_solve(sla.lu_factor(held.matrix * s / viscosity), np.tile(velocity, held.count))
    area = held.area * s ** 2
    return dict(force=held.force * s, points=held.points * s, weights=held.weights * s ** 2, owners=held.owners,
                area=area, eps2=held.eps_squared * s ** 2, forces=traction.reshape(-1, 3) * area[:, None])
