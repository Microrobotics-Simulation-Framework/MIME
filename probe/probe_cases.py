"""The cost probe's cases: A0 and SPIKE_R2's coil-array sphere, each with no port (off) and with ports (on)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for held in (ROOT / "tests", ROOT / "tests" / "oracles"):
    if str(held) not in sys.path:
        sys.path.insert(0, str(held))

from mime import CommandPort, Measurement  # noqa: E402
from mime.units import A, ms, rad, s  # noqa: E402

#: A0's ports: one step of the arm's link and a slew on its six joints, and one step of a tracker on the body's pose,
#: which the controller reads in place of the body's state.
A0_PORTS = {"arm.joint_positions": CommandPort(latency=ms(0.5), slew=rad(3.0) / s(1.0), reason="the cost probe's link"),
            "pose": Measurement(of="umr.pose", latency=ms(0.5), reason="the cost probe's tracker")}
#: The sphere's ports: two steps of the coils' link, their bounds and a slew, and a tracker on the sphere's pose.
SPHERE_PORTS = {"currents": CommandPort(latency=ms(2), bounds=A(25.0), slew=A(4000.0) / s(1.0),
                                        reason="the cost probe's amplifier"),
                "pose": Measurement(of="sphere.pose", latency=ms(1), reason="the cost probe's tracker")}


def a0_off():
    from script_a0 import a0

    return a0("cost_a0")


def a0_on():
    from chain_cases import TRACKING_READS
    from script_a0 import a0

    exp = a0("cost_a0", ports=A0_PORTS)
    held = exp._declarations[-1]  # noqa: SLF001 - the controller's declaration, rebound to read the tracker
    import dataclasses

    exp._declarations[-1] = dataclasses.replace(held, reads={**TRACKING_READS, "body_pose": "rpm.pose"})  # noqa: SLF001
    return exp


def sphere_off():
    from test_calibrated_multipole import sphere_in_the_array

    return sphere_in_the_array("navion_2", 52, torque=False)


def sphere_on():
    import dataclasses

    from test_calibrated_multipole import sphere_in_the_array

    from mime.build._declarations import ThingDeclaration
    from mime.things import Apparatus

    exp = sphere_in_the_array("navion_2", 52, torque=False)
    for index, declaration in enumerate(exp._declarations):  # noqa: SLF001
        if isinstance(declaration, ThingDeclaration) and declaration.thing.name == "coils":
            thing = declaration.thing
            exp._declarations[index] = dataclasses.replace(  # noqa: SLF001
                declaration, thing=Apparatus(thing.name, fields=thing.fields, commands=thing.commands, ports=SPHERE_PORTS))
    return exp
