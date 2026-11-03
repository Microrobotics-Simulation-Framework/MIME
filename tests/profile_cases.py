"""Small experiments for the profile tests: a steel ball settling in water, written the way the first-hour walk's new
user wrote it, each piece of paperwork optional (``paperwork`` toggles one by one), so a test can leave any one out,
give it pending, or seal it."""

from dataclasses import dataclass, field

from mime import Body, Experiment, Medium, State
from mime.dynamics import RigidInertial
from mime.interactions import Drag, Gravity
from mime.materials import Acceptance, Record, packaged_record
from mime.observations import SwimSpeed
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import degC, kg_per_m3, m_per_s, mm, ms, rad_per_s, s
from mime.validity import Basis

WATER = "openmaterica_example_water"
SHEAR = "condition_unstated:shear_rate"


@dataclass(frozen=True)
class Paperwork:
    """What the settling ball's script states: each ``None`` left out. ``acceptance`` is the water viscosity's
    acceptance reason, ``density`` the ball's declared density's reason (``"omit"`` leaves the reason off ``declare``),
    ``spacing``, ``blob`` and ``quadrature_refinement`` the mrs rung's settings, ``sources`` the ``exp.source`` lines, ``require`` whether water's
    record is read with its test method stated."""

    profile: str = "exploratory"
    acceptance: Basis | None = None
    density: Basis | str | None = "omit"
    spacing: object = None
    blob: object = None
    quadrature_refinement: object = None
    wall_kind: str | None = None
    sources: dict[str, Basis] = field(default_factory=dict)
    radius_reason: Basis | None = None
    require: bool = True


def settling_ball(paperwork: Paperwork = Paperwork()) -> Experiment:
    """A steel ball of 0.5 mm radius settling in water at 20 degC under the mrs drag, in free space."""
    exp = Experiment("settling_ball", scene=Scene.new(up=Z, units="m"), dt=ms(1.0), profile=paperwork.profile,
                     conditions={"temperature": degC(20.0, plus_minus=0.5)})
    accept = {} if paperwork.acceptance is None else {
        "fluid.dynamic_viscosity": Acceptance((SHEAR,), reason=paperwork.acceptance)}
    require = {"test_method": ("reference_formulation",)} if paperwork.require else {}
    water = exp.add_medium(Medium("water", material=Record("water", packaged_record(WATER), require=require,
                                                           accept=accept)))
    radius = mm(0.5, reason=paperwork.radius_reason) if paperwork.radius_reason is not None else mm(0.5)
    ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(Sphere(radius=radius))}))
    if paperwork.density == "omit":
        exp.declare("ball.parts.shell.density", kg_per_m3(7850.0))
    elif paperwork.density is not None:
        exp.declare("ball.parts.shell.density", kg_per_m3(7850.0), reason=paperwork.density)
    settings: dict[str, object] = {name: value for name, value in (
        ("spacing", paperwork.spacing), ("blob", paperwork.blob),
        ("quadrature_refinement", paperwork.quadrature_refinement), ("wall_kind", paperwork.wall_kind))
        if value is not None}
    exp.attach(Drag.MRS(**settings), body=ball, medium=water)
    exp.attach(Gravity(), body=ball, medium=water)
    exp.observe(SwimSpeed(ball, along=-Z, after=s(0.004)))
    for path, reason in paperwork.sources.items():
        exp.source(path, reason=reason)
    return exp


#: Every item the bare settling ball leaves pending in exploratory, by its target, in the outcome's canonical order.
BARE_PENDING = ("ball.parts.shell.density", "drag:ball.blob", "drag:ball.quadrature_refinement", "drag:ball.spacing",
                "material 'water' fluid.dynamic_viscosity")
