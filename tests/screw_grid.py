"""The screw kind's settings the kind-conformance harness runs beside FL-9's case (``tests/saved/test_registry_walk.py``),
shared with the caps' own checks (``tests/scene/test_screw_caps.py``) so that both run on the same grid."""

import itertools

from mime.scene._screw import STARTS

#: The helix's settings beside FL-9's case: every count of starts the kind accepts, at the least and the most modulation
#: it accepts and FL-9's, and at no twist, FL-9's, the most of the left hand and, of the right, where its normals were
#: found furthest from the screw's (3.75): (starts, modulation, nu).
GRID_SETTINGS = list(itertools.product(STARTS, (0.02, 0.33, 0.5), (-4.0, 0.0, 2.33, 3.75)))
#: And off that grid, where the declared normal tolerance's measurement found its largest comparison: three starts, where
#: ``starts * theta`` rounds (at two and four it is exact), at a modulation of 0.2 to 0.35 and a twist near zero, where
#: the normal's sensitivity is small and the rounding of ``3 theta`` is carried whole.
OFF_GRID_SETTINGS = [(3, 0.2, 0.1), (3, 0.25, 0.1), (3, 0.3, 0.1), (3, 0.32, 0.11), (3, 0.35, 0.1)]
HELIX_SETTINGS = GRID_SETTINGS + OFF_GRID_SETTINGS
#: FL-9's radius and length.
FL9 = {"radius": 1.56e-3, "length": 7.47e-3}
#: The harness's spacings, as fractions of the kind's reach: a tenth (the grid's) and a fortieth (the declared cases').
REACH_FRACTIONS = (10.0, 40.0)
