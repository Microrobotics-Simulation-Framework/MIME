"""A small robot asset, written by the tests: a body prim ``/UMR`` holding a shell and a magnet part ``Magnets``,
turned 90 degrees about z inside the body, whose magnetisation direction is design data in the ``mime:`` schema.
Phase 5 brings parts and materials; this stands in for a real robot until then."""

import tempfile
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom

from mime import Apparatus, Body, Experiment, Medium, PermanentMagnet, Scene, State, Structure, asset
from mime.dynamics import RigidInertial
from mime.fields import UniformRotating
from mime.interactions import Contact, Gravity, Magnetic
from mime.scene import MAGNET_API, MAGNET_DIRECTION, Asset, Cylinder, Ellipsoid, Tube, X, Z
from mime.things import Part
from mime.units import A_m2, Hz, Quantity, kg, kg_m2, kg_per_m3, m, m3, m_per_s, mm, ms, mT, rad_per_s, standard_gravity


def write_robot(path: Path, *, direction: tuple[float, float, float] | None = (1.0, 0.0, 0.0),
                schema: bool = True, turn_degrees: float = 90.0, meters_per_unit: float | None = 1.0,
                default_prim: bool = True, root: str = "UMR") -> Path:
    """Write the asset to ``path``. ``direction`` is the magnet's direction in its part's frame (``None`` writes
    none); ``schema`` applies ``MimeMagnetAPI`` on the part; ``turn_degrees`` turns the part about the body's z."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    if meters_per_unit is not None:
        UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    body = UsdGeom.Xform.Define(stage, f"/{root}")
    if default_prim:
        stage.SetDefaultPrim(body.GetPrim())
    shell = UsdGeom.Sphere.Define(stage, f"/{root}/Shell")
    shell.CreateRadiusAttr(1.56e-3)
    magnet = UsdGeom.Cylinder.Define(stage, f"/{root}/Magnets")
    magnet.CreateRadiusAttr(0.5e-3)
    magnet.CreateHeightAttr(2e-3)
    UsdGeom.Xformable(magnet).AddRotateZOp().Set(turn_degrees)
    prim = magnet.GetPrim()
    if schema:
        prim.AddAppliedSchema(MAGNET_API)
    if direction is not None:
        prim.CreateAttribute(MAGNET_DIRECTION, Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*direction))
    path.write_text(stage.GetRootLayer().ExportToString(), encoding="utf-8")
    return path


_written: dict[str, Path] = {}


def robot_file(**options: object) -> Path:
    """The robot written once per process and set of options, in a folder of this process's own, so that tests on
    several workers never write one file at once."""
    key = repr(sorted(options.items()))
    if key not in _written:
        folder = Path(tempfile.mkdtemp(prefix="mime-robot-"))
        _written[key] = write_robot(folder / "fl9_robot.usda", **options)  # type: ignore[arg-type]
    return _written[key]


def bound_robot(name: str = "bound_robot", *, robot: Path | Asset | None = None, at: str = "/World/UMR",
                magnet: PermanentMagnet | None = None, moment: Quantity | None = None,
                body_name: str = "") -> Experiment:
    """Script A's body side with the robot as an asset: a body bound to the asset's prim, its magnet's direction read
    from the asset, in a tube written Python-first, under gravity and a uniform field turning about the tube's axis.
    The body moves along the tube at 1 mm/s, so that every leaf of every node acts within the first steps."""
    scene = Scene.new(up=Z, units="m")
    scene.reference(robot if isinstance(robot, Asset) else asset(robot or robot_file()), at=at)
    experiment = Experiment(name, scene=scene, gravity=standard_gravity(), dt=ms(0.5))
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                                    length=mm(100.0))))
    water = experiment.add_medium(Medium("water", fills=tube.lumen))
    umr = experiment.add_body(Body(body_name, prim=scene["/World/UMR"], dynamics=RigidInertial(),
                                   magnets={"Magnets": magnet or PermanentMagnet()},
                                   initial=State(position=mm((10.0, 0.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                                 velocity=m_per_s((1e-3, 0.0, 0.0)),
                                                 angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    for value, held in (("mass", kg(8.5e-5)), ("inertia", kg_m2((1e-10,) * 3)), ("buoyant_volume", m3(6.0e-8)),
                        ("centre_of_buoyancy", m((0.0, 0.0, 0.0))), ("extent", mm(2.07)),
                        ("magnetic_moment", moment if moment is not None else A_m2(1.68e-3))):
        experiment.declare(f"{umr.name}.{value}", held, reason="the test robot")
    experiment.declare("water.density", kg_per_m3(1000.0), reason="water")
    coils = experiment.add_apparatus(Apparatus("coils", fields={"magnetic": UniformRotating(
        rotation_axis=tube.axis, initial_direction=(0.0, 1.0, 0.0))},
        commands={"frequency": Hz(10.0), "amplitude": mT(1.2)}))
    experiment.attach(Magnetic.Torque(), body=umr, field=coils.magnetic)
    experiment.attach(Gravity(), body=umr, medium=water)
    experiment.attach(Contact.Projection(margin=mm(0.1)), body=umr, surface=tube.inner_surface)
    return experiment


def bound_robot_with_parts(name: str = "bound_robot_with_parts") -> Experiment:
    """The test robot bound to its asset's prim with its parts declared: the shell, an ellipsoid of the sphere's
    radius at a declared density, and the magnet inside it, a cylinder of N45 (OpenMaterica's example record, at
    20 degC), placed and turned 90 degrees about the body's z by the asset's own transforms. Every value the laws
    read comes from the parts, and the contact holds the shell's points: the shell starts a micrometre above the floor,
    less the margin, and lands on it in the first steps, so that every leaf of the contact acts."""
    from script_a import CONDITIONS, N45_RECORD

    scene = Scene.new(up=Z, units="m")
    scene.reference(asset(robot_file()), at="/World/UMR")
    experiment = Experiment(name, scene=scene, gravity=standard_gravity(), dt=ms(0.5), conditions=CONDITIONS)
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                                    length=mm(100.0))))
    water = experiment.add_medium(Medium("water", fills=tube.lumen))
    umr = experiment.add_body(Body(prim=scene["/World/UMR"], dynamics=RigidInertial(),
                                   magnets={"Magnets": PermanentMagnet()},
                                   parts={"Shell": Part(Ellipsoid(semi_axes=mm((1.56, 1.56, 1.56)))),
                                          "Magnets": Part(Cylinder(radius=mm(0.5), length=mm(2.0), axis=Z),
                                                          material=N45_RECORD, inside="Shell")},
                                   initial=State(position=mm((10.0, 0.0, -1.514)), orientation=(1.0, 0.0, 0.0, 0.0),
                                                 velocity=m_per_s((1e-3, 0.0, 0.0)),
                                                 angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare(f"{umr.name}.parts.Shell.density", kg_per_m3(1410.0), reason="the test robot's mean density")
    experiment.declare("water.density", kg_per_m3(1000.0), reason="water")
    coils = experiment.add_apparatus(Apparatus("coils", fields={"magnetic": UniformRotating(
        rotation_axis=tube.axis, initial_direction=(0.0, 1.0, 0.0))},
        commands={"frequency": Hz(10.0), "amplitude": mT(1.2)}))
    experiment.attach(Magnetic.Torque(), body=umr, field=coils.magnetic)
    experiment.attach(Gravity(), body=umr, medium=water)
    experiment.attach(Contact.Projection(margin=mm(0.1)), body=umr, surface=tube.inner_surface)
    return experiment
