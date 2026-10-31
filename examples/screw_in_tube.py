"""A magnetic screw swimming along a tube of viscous liquid, with no material records: every value is declared.

``screw_in_tube()`` is the experiment: a de Jongh FL-9 screw (de Jongh et al. 2025, Table 1) on the axis of a 1/4"
tube filled with glycerol, turned by a uniform field rotating about the tube's axis, its drag by the boundary-element
rung with the tube's wall kind. ``main()`` runs it and prints the speed over the last turn and the post-run watch.

    python examples/screw_in_tube.py                     # the starting size below
    python examples/screw_in_tube.py --spacing-mm 0.5    # finer samples: more accurate, and slower (see the table)

**Why glycerol, and why 1 Hz.** The rung is Stokes flow, declared up to a Reynolds number the post-run watch reads
back. An FL-9 turned at 5 Hz in water (the 0.2 line's script) has a turning Reynolds number in the hundreds, so a
run of that is flagged. At 1 Hz in glycerol (1.4 Pa s) it is a few hundredths, inside the range.

**Starting sizes** (``docs/choosing_spacing.md`` holds the measured tables): ``dt`` 100 ms (ten steps per turn of the
field) and a run of 3 s, the speed read over the last turn; ``spacing`` 0.7 mm, which gives the screw a few hundred
samples. What that number is worth is in the table: the speed converges slowly in ``spacing`` (the helix's thin turns
need fine samples) and is far below the finest spacing measured, so read the number as a first look, its sign and its
order, and not as a result.

**The inertia.** The screw's parts give an inertia tensor with a small product of inertia, which the build refuses to
use as it is and prints (with the line to paste). Here the declaration below is that line; in this regime (viscous,
Reynolds number a few hundredths) the speed does not depend on it, which the example's test checks.
"""

import argparse

from mime import Apparatus, Body, Experiment, Medium, State, Structure
from mime.dynamics import RigidInertial
from mime.fields import UniformRotating
from mime.interactions import Drag, Magnetic
from mime.observations import SwimSpeed
from mime.results import RunResult
from mime.scene import Scene, Screw, Tube, X, Z
from mime.things import Part
from mime.units import (
    A_m2,
    Hz,
    dimensionless,
    kg_m2,
    kg_per_m3,
    m_per_s,
    m_per_s2,
    mm,
    mm_per_s,
    mPa_s,
    ms,
    mT,
    rad_per_s,
    s,
)

#: The field's frequency and the run: three turns, the speed read over the last.
FREQUENCY_HZ, DURATION_S = 1.0, 3.0
#: The starting spacing (mm): `docs/choosing_spacing.md` says what it costs and what it is worth.
START_SPACING_MM = 0.7
#: The starting base step (ms): ten steps to a turn of the field. In this regime the speed does not depend on it down
#: to a few steps a turn (the table in `docs/choosing_spacing.md`), and the run costs what its steps do.
START_DT_MS = 100.0


def screw_in_tube(spacing_mm: float = START_SPACING_MM, *, inertia: tuple[float, float, float] | None = None,
                  dt_ms: float = START_DT_MS) -> Experiment:
    """The screw at rest on the tube's axis, its samples ``spacing_mm`` apart. ``inertia`` replaces the parts' tensor's
    diagonal (kg m^2, about the body's axes) that the build prints."""
    exp = Experiment("screw_in_tube", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(dt_ms))
    tube = exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                             length=mm(100.0))))
    liquid = exp.add_medium(Medium("glycerol", fills=tube.lumen))
    exp.declare("glycerol.density", kg_per_m3(1260.0), reason="glycerol at 20 degC, a handbook value")
    exp.declare("glycerol.viscosity", mPa_s(1400.0), reason="glycerol at 20 degC, a handbook value")
    umr = exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(Screw(
            radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33),
            starts=2))}))
    exp.declare("umr.parts.shell.density", kg_per_m3(1410.0), reason="FL-9 resin plus its magnets, mean density")
    exp.declare("umr.inertia", kg_m2(inertia or INERTIA),
                reason="the parts' inertia tensor about the body's axes with its products of inertia dropped, as the "
                       "build prints it: a modelling choice, not a measurement")
    exp.declare("umr.magnetic_moment", A_m2((0.0, 1.68e-3, 0.0)), reason="two magnets, across the screw's axis")
    field = exp.add_apparatus(Apparatus("field", fields={
        "magnetic": UniformRotating(rotation_axis=tube.axis, initial_direction=(0.0, 1.0, 0.0))},
        commands={"frequency": Hz(FREQUENCY_HZ), "amplitude": mT(5.0)}))
    exp.attach(Drag.BEM(spacing=mm(spacing_mm), blob=0.5, wall_kind="tube"), body=umr, medium=liquid)
    exp.attach(Magnetic.Torque(), body=umr, field=field.magnetic)
    exp.observe(SwimSpeed(umr, along=tube.axis, after=s(DURATION_S - 1.0 / FREQUENCY_HZ)))
    return exp


#: The diagonal of the parts' inertia tensor (kg m^2): what the refusal of the parts' own tensor prints to paste.
INERTIA = (1.3043197461318855e-10, 4.631130043564001e-10, 4.570054968582633e-10)


def run(spacing_mm: float = START_SPACING_MM) -> RunResult:
    return screw_in_tube(spacing_mm).build().run(s(DURATION_S), history=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spacing-mm", type=float, default=START_SPACING_MM)
    result = run(parser.parse_args().spacing_mm)
    print(result.text())
    print(f"speed over the last turn: {result['swim_speed:umr'].mean_speed.to(mm_per_s):.4g} mm/s")


if __name__ == "__main__":
    main()
