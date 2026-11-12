"""What the shape suite's files share (``test_shape_suite.py`` and ``test_shape_suite_tube_tree.py``): the fixed examples,
one near each declared set of each shape, their planted-tangent control, and the points-side property's body.

The tube tree's tests were split out of ``test_shape_suite.py`` so that each file can take a shard of its own: the tube
tree's union compiles for each of its examples and each of its probes (about 60 s a dtype on a runner), and the whole
file ran about 20 minutes on a slow runner, most of it those seven tests. The examples are made once per shape and dtype
in a process (``examples_of`` caches), so the tests that read the tube tree's examples stay together in one file."""

import functools

import numpy as np
from hypothesis import event

import mime.testing._shape_suite as suite
from mime.testing import (
    PLANTED_TANGENT,
    SHAPE_CASES,
    SHAPE_TOLERANCE,
    ShapeCheck,
    ShapeProbe,
    check_shape,
    shape_probes,
)

#: Each case's shape as an example places it: a size each shape is used at, turned off every coordinate axis.
TYPICAL = {
    "sphere": {"centre": (0.0, 0.0, 0.0), "radius": 1e-3},
    "cylinder": {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "radius": 3.175e-3},
    "capsule": {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "radius": 5e-4, "height": 2e-3},
    "slab": {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "half_thickness": 5e-4},
    "half_space": {"normal": (1.0, 0.0, 0.0), "offset": 1e-3},
    "screw": {"centre": (0.0, 0.0, 0.0), "orientation": (1.0, 0.0, 0.0, 0.0), "radius": 1.56e-3,
                      "length": 7.47e-3, "wavenumber": 2.33, "modulation": 0.33, "starts": 2},
    "tube_tree": suite._tube_tree(5e-4, 0.1),  # noqa: SLF001 - the case's own Y
}
TURN = np.array([0.9, 0.3, -0.2, 0.25]) / np.linalg.norm([0.9, 0.3, -0.2, 0.25])
DIRECTION = np.array([0.48, -0.6, 0.64])
DTYPES = ("float64", "float32")
#: How far an example lies from its set, in the shape's size, by dtype: where autodiff's floor lies below the planted
#: tangent. A thousandth in float64; a tenth in float32, about two hundred of its axis bands and the edge of its near
#: class, where that floor is about 1e-3 of the scale (a thousandth from a set it is about 1e-1).
EXAMPLE_OFFSET = {"float64": 1e-3, "float32": 0.1}
#: A moved parameter's rate in an example (per metre the point moves, over the size for a dimensionless one or a
#: turn), and the two uniform numbers that give a centre's direction or a turn's axis: off every axis.
RATE, TURNING = 1.0, np.array([0.31, 0.77])
#: The draws that place an example on a set; the second places the screw's focus at a crest where the first places it
#: at a trough.
DRAWS = (np.array([0.37, 0.61, 0.53]), np.array([0.62, 0.21, 0.18]))

def examples(dtype: str, shapes: tuple[str, ...] | None = None) -> tuple[tuple[ShapeProbe, str, str, ShapeCheck], ...]:
    """A probe near each declared set of each shape (of ``shapes``, every shape by default), for each parameter the
    case moves and each capability, in ``dtype``, ``EXAMPLE_OFFSET[dtype]`` of the size from the set on its positive
    side, along a direction off every axis with the parameter moved at ``RATE``, of the sign along which the float64
    difference finds the steeper derivative (so that no parameter's part cancels the point's); with the parameter and
    its unplanted check. Made once per shape and dtype, whichever test asks first."""
    return tuple(held for shape in sorted(SHAPE_CASES) if shapes is None or shape in shapes
                 for held in examples_of(dtype, shape))


@functools.cache
def examples_of(dtype: str, shape: str) -> tuple[tuple[ShapeProbe, str, str, ShapeCheck], ...]:
    case = SHAPE_CASES[shape]
    found = []
    own = TYPICAL[shape]
    size = case.size(own)
    rotation = suite._rotation(TURN)  # noqa: SLF001 - the suite's own rotation, as its probes turn
    shift = size * np.array([0.7, -1.3, 2.1])
    world = case.turned(own, TURN, shift)
    offset = EXAMPLE_OFFSET[dtype] * size
    for chosen in case.sets:
        for draw in DRAWS[:2 if chosen.owner == "screw.focus" else 1]:
            point = rotation @ chosen.place(own, offset, draw) + shift
            for name in case.moved:
                curves = [ShapeProbe(shape, world, point, DIRECTION / np.linalg.norm(DIRECTION),
                                     {name: suite.moved_rate(case, world, name, sign * RATE, TURNING, size)},
                                     chosen.owner, "near", offset, dtype, size) for sign in (1.0, -1.0)]
                for capability in case.capabilities:
                    checks = [(probe, check_shape(probe, capability)) for probe in curves]
                    probe, check = max(checks, key=lambda held: steepness(held[1]))
                    found.append((probe, capability, name, check))
    return tuple(found)

#: The shapes whose examples have tests of their own: the tube tree's union compiles for each of its examples (about
#: 60 s a dtype on a runner), which would take the other shapes' tests past their ceilings.
SEPARATE = ("tube_tree",)


def steepness(check: ShapeCheck) -> float:
    return float(np.max(np.abs(check.difference))) / check.scale if check.difference is not None else 0.0



def examples_pass(dtype: str, shapes: tuple[str, ...]) -> None:
    failed = [f"{check.line()} (moving {name}): " + "; ".join(check.problems)
              for _, _, name, check in examples(dtype, shapes) if not check.passed]
    assert failed == [], "\n".join(failed)


#: The examples whose derivative along the curve is zero wherever a difference is taken, so that no scaling of the
#: tangent can show: the slab's normal is its axis on one side of the mid-plane and its opposite on the other, so it
#: moves with neither the point, the half-thickness nor the centre (only the axis turns it); the half-space's normal is
#: its own everywhere, so it moves with neither the point nor the offset (only the normal turns it).
CONSTANT = {("slab", "normal", "half_thickness"), ("slab", "normal", "centre"), ("half_space", "normal", "offset")}


def planted_tangent_fails(dtype: str, shapes: tuple[str, ...]) -> None:
    caught, missed, constant, margins = [], [], set(), []
    for probe, capability, name, clean in examples(dtype, shapes):
        steep = float(np.max(np.abs(clean.autodiff))) / clean.scale
        if steep <= SHAPE_TOLERANCE[dtype]:  # zero, to rounding
            constant.add((probe.shape, capability, name))
            continue
        resolution = (SHAPE_TOLERANCE[dtype] * clean.scale + clean.floor + clean.truncation) \
            / float(np.max(np.abs(clean.autodiff)))
        assert clean.discriminating, clean.line()
        assert PLANTED_TANGENT[dtype] > resolution, \
            f"{clean.line()} (moving {name}): the comparison resolves {resolution:.3g} here, so the planted tangent " \
            f"cannot show"
        margins.append((PLANTED_TANGENT[dtype] / resolution, f"{clean.line()} (moving {name})"))
        planted = check_shape(probe, capability, planted=PLANTED_TANGENT[dtype])
        (caught if planted.problems else missed).append((probe.shape, probe.owner, capability, name))
    smallest = min(margins)
    print(f"\n{dtype}: the planted tangent (x {1 + PLANTED_TANGENT[dtype]}) is caught at {len(caught)} of "
          f"{len(caught) + len(missed)} examples, at least {smallest[0]:.3g} times what the comparison resolves there "
          f"({smallest[1]}); {sorted(constant)} have no derivative to scale")
    assert missed == [], f"missed at {missed}"
    assert constant == {held for held in CONSTANT if held[0] in shapes}
    sets = {(shape, s.owner, capability) for shape, case in SHAPE_CASES.items() if shape in shapes for s in case.sets
            for capability in case.capabilities}
    assert {held[:3] for held in caught} == sets


def points_side_holds(shape: str, name: str, data) -> None:
    """The property at one drawn probe: autodiff equals the difference on the point's side."""
    found = check_shape(data.draw(shape_probes(SHAPE_CASES[shape])), name)
    if found.one_sided is not None:
        print(f"\n{found.line()}: one-sided slopes {found.one_sided[0]} and {found.one_sided[1]}, autodiff "
              f"{found.autodiff}")
    if not found.discriminating:  # not a pass, not a failure: counted, and its share bounded by its own test below
        event("non-discriminating")
        print(f"\n{found.line()}")
    assert found.problems == (), f"{found.line()}: " + "; ".join(found.problems)

