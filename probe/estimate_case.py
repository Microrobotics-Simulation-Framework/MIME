"""A memory proxy for the porous plug's measurement arms: the same periodic tube box without a plug (the plug adds a
few face-set arrays to the step's), built as an experiment so the estimator can compile it."""

from mime.dynamics import FVM, Band, Scheme
from mime.experiment import Experiment
from mime.interactions import BodyForce
from mime.scene import PeriodicBox, Scene, Tube, Z
from mime.things import Medium, Structure
from mime.units import kg_per_m3, m_per_s2, mm, mPa_s, ms


def tube(cells_across: int = 16, length_cells: int = 64, delta_in: float = 0.1) -> Experiment:
    exp = Experiment("plug_proxy", scene=Scene.new(up=Z, units="m"), dt=ms(20.0))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2), length=mm(10.0))))
    flow = FVM(domain=PeriodicBox(length=mm(length_cells / cells_across)), cells_across=cells_across, stokes=True,
               scheme=Scheme.Steady(), wall=Band(delta_in=delta_in))
    exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    exp.declare("water.viscosity", mPa_s(1.0), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(1.0)), medium="water")
    return exp
