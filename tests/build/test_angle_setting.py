"""A geometry setting annotated ``Annotated[Quantity, Angle]`` reaches the entry's ``setting_problems`` in radians, and is
checked as an angle at build. It was left out of the entry's settings with no refusal (``mime.units.Angle`` was a type
alias, not a marker, and the reader skipped metadata it did not know), so a cone's half-angle raised ``KeyError`` in its
own ``setting_problems``, reported as a defect of the entry.

**What this bounds.** That the setting is kept and an entry reads it as a float in radians, from degrees and from radians;
that a bare number (30 of what?), a length and a number of turns are refused as ``setting.units``, and a value the entry
judges out of range is refused as ``setting.value`` with the entry's own message. **What it does not.** That the angle
means what the entry does with it.

**Controls.** The refusals are the controls for the accepted: each wrong kind of value is built beside the right one."""

import dataclasses
import math
from typing import Annotated, override

import pytest

from mime.build import BuildRefused
from mime.experiment import Experiment
from mime.gradients import GradientStatus
from mime.registry import Geometry, register_geometry
from mime.scene import Scene, Z
from mime.things import Structure
from mime.units import Angle, Hz, Quantity, deg, m_per_s2, mm, ms, rad, s
from mime.vocabulary import register_namespace

register_namespace("angles", "0.0.1")
SEEN: list[float] = []


@register_geometry
@dataclasses.dataclass(frozen=True, kw_only=True)
class Cone(Geometry):
    """A cone's lumen, only its half-angle declared: the entry's own judgement reads it in radians."""

    name = "angles.cone"
    version = "0.0.1"
    differentiability = GradientStatus.PIECEWISE_JUMP
    builds = {"": ()}
    regions = ("lumen",)
    region_shapes = {"lumen": "slab"}
    half_angle: Annotated[Quantity, Angle]

    @override
    def setting_problems(self, settings):
        SEEN.append(float(settings["half_angle"]))
        return [] if 0.0 < settings["half_angle"] < math.pi / 2 else ["half_angle is not inside (0, pi/2) radians"]

    @override
    def region_parameters(self, region, settings):
        return {"centre": (0.0, 0.0, 0.0), "axis": (0.0, 0.0, 1.0), "half_thickness": settings["half_angle"]}


def planned(half_angle):
    experiment = Experiment("cone", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    experiment.add_structure(Structure("cone", geometry=Cone(half_angle=half_angle)))
    return experiment.plan()


@pytest.mark.parametrize(("written", "radians"), [(deg(30), math.radians(30)), (rad(0.5), 0.5)], ids=["deg", "rad"])
def test_an_angle_setting_reaches_the_entry_in_radians(written, radians):
    SEEN.clear()
    planned(written)
    assert SEEN == [pytest.approx(radians)]


@pytest.mark.parametrize(("written", "code", "said"), [
    (30.0, "setting.units", "the bare value 30.0"),
    (mm(1), "setting.units", "it was given length"),
    (Hz(1) * s(0.1), "setting.units", "turns"),
    (deg(100), "setting.value", "half_angle is not inside"),
], ids=["bare", "length", "turns", "out-of-range"])
def test_an_angle_setting_given_the_wrong_kind_of_value_is_refused_with_the_line_to_write(written, code, said):
    with pytest.raises(BuildRefused) as refused:
        planned(written)
    assert refused.value.codes() == [code] and said in str(refused.value)
    if said == "the bare value 30.0":
        assert "half_angle=deg(30.0) if degrees" in str(refused.value) and "half_angle=rad(30.0) if radians" in str(refused.value)
