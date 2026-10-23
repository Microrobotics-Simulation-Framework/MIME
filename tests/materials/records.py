"""Records and small experiments for the materials tests: the example water record at 37 degC alone, written as
OpenMaterica's quickstart writes its own (its alternatives at 310.15 K, under a test id, with the content hash
recomputed); OpenMaterica's own table fixture; and script A's water, read at chosen conditions by the laws that read
it (gravity reads the density, the drag the viscosity)."""

import json
import shutil
from pathlib import Path

import materica
from script_a import EXTENT, INERTIA, LENGTH, MASS, MOMENT, NU, RADIUS, VOLUME, WATER

from mime import Apparatus, Body, Experiment, Medium, State, Structure
from mime.data import asset, asset_bytes
from mime.dynamics import RigidInertial
from mime.fields import UniformRotatingField
from mime.interactions import Drag, Gravity, Lubrication, Magnetic
from mime.materials import Record
from mime.scene import Scene, Tube, X, Z
from mime.units import Hz, m, m_per_s, m_per_s2, mm, ms, mT, rad_per_s

#: The example water record's values, every number a field holds: what the saved document must never hold.
WATER_VALUES = (998.207, 993.33, 0.0010016, 0.000691304)
N45_BOUNDS = (1.32, 1.37)


def example_water() -> dict:
    return json.loads(asset_bytes(asset("openmaterica_example_water")).decode(), parse_int=float)


def water_at_37(folder: Path) -> Path:
    """A record holding only the example's values at 310.15 K, pinned by its own content hash."""
    held = example_water()
    sections = held["body"]["sections"]
    record = {key: held[key] for key in ("schema_version", "record_type", "licence")}
    record.update({"id": "test/water_310", "version": "0.1.0", "name": "Liquid water at 310.15 K (test)",
                   "body": {"sections": {
                       "bulk": {"fields": {"density": sections["bulk"]["fields"]["density"]["alternatives"][1]}},
                       "fluid": {"fields": {"dynamic_viscosity":
                                            sections["fluid"]["fields"]["dynamic_viscosity"]["alternatives"][1]}}}}})
    record["content_hash"] = materica.content_hash(record)
    path = folder / "water_310.json"
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return path


def density_table(folder: Path) -> Path:
    """OpenMaterica's conformance fixture of a density table over temperature: 1000, 996, 990 and 986 kg/m^3 at 280,
    300, 320 and 340 K, linear, its density apparent."""
    source = Path(materica.__file__).parent / "data/conformance/plain_read/inputs/density_table.json"
    path = folder / "density_table.json"
    shutil.copyfile(source, path)
    return path


def water_experiment(material: Record | None, conditions=None, *, viscosity: bool = False,
                     name: str = "water_read") -> Experiment:
    """Script A's water, its material ``material``, read at ``conditions`` by gravity (the density) and, with
    ``viscosity``, by the drag (the viscosity); every other value declared as the reproduction arm declares it."""
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.5),
                            conditions=conditions)
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                                    length=mm(100.0))))
    water = experiment.add_medium(Medium("water", fills=tube.lumen, material=material))
    umr = experiment.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((10.0, 0.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    field = experiment.add_apparatus(Apparatus("field", fields={
        "magnetic": UniformRotatingField(rotation_axis=tube.axis, initial_direction=(0.0, 1.0, 0.0))},
        commands={"frequency": Hz(10.0), "amplitude": mT(1.2)}))
    for path, value in (("umr.mass", MASS), ("umr.inertia", INERTIA), ("umr.buoyant_volume", VOLUME),
                        ("umr.magnetic_moment", MOMENT), ("umr.extent", EXTENT),
                        ("umr.centre_of_buoyancy", m((0.0, 0.0, 0.0)))):
        experiment.declare(path, value, reason="script A's")
    if material is None:
        experiment.declare("water.density", WATER, reason="script A's")
    experiment.attach(Gravity(), body=umr, medium=water)
    experiment.attach(Magnetic.Torque(), body=umr, field=field.magnetic)
    if viscosity:
        for path, value in (("umr.helix_wavenumber", NU), ("umr.length", LENGTH), ("umr.radius", RADIUS)):
            experiment.declare(path, value, reason="script A's")
        experiment.attach(Drag.Surrogate(weights=asset("dejongh_mlp_v2"), lubrication=Lubrication(epsilon=mm(0.15))),
                          body=umr, medium=water)
    return experiment
