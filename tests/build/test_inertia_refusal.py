"""The refusal of an inertia tensor whose principal axes are not the body frame's (``parts.inertia-axes``) prints what
a user needs to act on it (dev1 phase 7l-c, design #191 item 7): the tensor the parts give in the body's frame with its
units, its principal moments and axes, and the declaration that states the diagonal with a reason, written out to paste.

**Controls.** The pasted declaration, executed, builds: the refusal's fix is a fix. A plain body (a sphere, whose tensor
is diagonal) is not refused, and a tensor whose product is planted large is refused with that product named, so the
message reads the tensor and not a constant."""

import re

import numpy as np
import pytest

from mime import Body, Experiment, State
from mime.build import BuildRefused
from mime.dynamics import RigidInertial
from mime.scene import Box, Scene, Screw, Sphere, Z
from mime.things import Part
from mime.units import dimensionless, kg_m2, kg_per_m3, m_per_s, m_per_s2, mm, ms, rad_per_s

AT_REST = dict(orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
               angular_velocity=rad_per_s((0.0, 0.0, 0.0)))


def screw_experiment() -> Experiment:
    exp = Experiment("screw", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(position=mm((0.0, 0.0, 0.0)), **AT_REST), parts={
        "shell": Part(Screw(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33),
                            modulation=dimensionless(0.33), starts=2))}))
    exp.declare("umr.parts.shell.density", kg_per_m3(1410.0), reason="FL-9 resin plus magnets, mean density")
    return exp


def refusal_of(exp: Experiment):
    with pytest.raises(BuildRefused) as refused:
        exp.build()
    (found,) = [r for r in refused.value.refusals if r.code == "parts.inertia-axes"]
    return found


def tensor_in(message: str) -> np.ndarray:
    rows = re.findall(r"^\s+\[([^\]]*)\]\s*$", message, re.M)
    return np.array([[float(c) for c in row.split()] for row in rows])


def test_the_refusal_prints_the_tensor_in_the_bodys_frame_its_principal_moments_and_the_declaration_to_paste():
    found = refusal_of(screw_experiment())
    assert "in the body's frame (kg m^2)" in found.message and "its principal moments, with their axes" in found.message
    tensor = tensor_in(found.message)
    assert tensor.shape == (3, 3) and np.allclose(tensor, tensor.T)
    assert tensor[1, 2] != 0.0 and tensor[0, 1] == 0.0
    moments = np.linalg.eigvalsh(tensor)
    printed = [float(n) for n in re.findall(r"^\s+([0-9.e+-]+) kg m\^2 about", found.message, re.M)]
    assert printed == pytest.approx(sorted(moments), rel=1e-5)
    assert 'exp.declare("umr.inertia", kg_m2((' in found.action and 'reason="' in found.action
    assert "its axes lie at most" in found.message and "differ from the diagonal's by at most" in found.message


def test_the_pasted_declaration_builds_and_declares_the_diagonal_the_message_prints():
    found = refusal_of(screw_experiment())
    line = re.search(r'(exp\.declare\("umr\.inertia", kg_m2\(\([^)]*\)\), reason="[^"]*"\))', found.action)
    assert line is not None
    exp = screw_experiment()
    exec(line.group(1), {"exp": exp, "kg_m2": kg_m2})  # noqa: S102 - the refusal's own line, run to show it works
    built = exp.build()
    declared = [v for v in built.report.values if v.path == "umr.inertia"]
    assert declared and tuple(declared[0].value) == pytest.approx(tuple(np.diag(tensor_in(found.message))), rel=1e-5)
    assert "dropped" in declared[0].reason


def test_a_body_with_a_diagonal_tensor_is_not_refused():
    exp = Experiment("ball", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
    exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(position=mm((0.0, 0.0, 0.0)), **AT_REST),
                      parts={"steel": Part(Sphere(radius=mm(0.5)))}))
    exp.declare("ball.parts.steel.density", kg_per_m3(7850.0), reason="steel")
    assert exp.build().plans.dt == pytest.approx(1e-3)


def test_the_message_reads_the_tensor_it_is_given_a_planted_product_is_named_and_shown():
    from mime.things import PartMass, derive

    cube = PartMass("a", "box", 1e-9, (0.0, 0.0, 0.0), ((1.0e-6, 0.0, 0.0), (0.0, 1.0e-6, 3.0e-7), (0.0, 3.0e-7, 2.0e-6)))
    (inertia,) = [derived for name, derived in derive("planted", [cube], {"a": 1000.0}, {}).items() if name == "inertia"]
    ((code, message, action),) = inertia.problems
    assert code == "parts.inertia-axes" and "3e-13 kg m^2" in message and "+3.000000e-13" in message
    numbers = re.search(r"kg_m2\(\(([^)]*)\)\)", action)
    assert numbers is not None
    assert [float(n) for n in numbers.group(1).split(",")] == pytest.approx([1e-12, 1e-12, 2e-12], rel=1e-9)
