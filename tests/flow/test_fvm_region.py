"""The open vessel's region built by the region builder, through the experiment (F2b, PR 4a): a tube tree cut at its
own ends or around a junction, each cut extended and bent onto the box's z faces, its ends' laws attached by the
vessel's own names; a tilted straight tube the same way; the backflow term on the outlets (by the centreline, never
the law) with its work booked under ``<node>.backflow``; the outlet watch, which acts on nothing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mime.build import BuildRefused
from mime.dynamics import FVM, Scheme
from mime.experiment import Experiment
from mime.interactions import Inflow, SharedDistal, Windkessel
from mime.observations import EnergyLedger
from mime.scene import Scene, Tube, TubeTree, VesselRegion, Z
from mime.testing import ledger_closes
from mime.things import Medium, Structure
from mime.units import Hz, Pa_s, Pa_s_per_m3, kg_per_m3, m3_per_Pa, m3_per_s, mm, ms

MU = 1e-3
#: A Y with unequal branches: a parent of radius 0.5 mm along z, daughters of 0.4 and 0.35 mm up and out, unequal in
#: length and angle (so a mix-up of an opening's name and its patch moves the split).
Y = dict(points=mm((0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 1.2, 0.0, 2.6, -1.0, 0.0, 2.0)),
         segments=((0, 1), (1, 2), (1, 3)), radii=mm((0.5, 0.5, 0.4, 0.4, 0.35, 0.35)))
FLUX = 1e-9


def tree_vessel(*, region=None, laws=None, cells_across=6, ledger=True, tree=None) -> Experiment:
    exp = Experiment("open_tree", scene=Scene.new(up=Z, units="m"), dt=ms(1.0))
    exp.add_structure(Structure("vessel", geometry=tree if tree is not None else TubeTree(**Y)))
    flow = FVM(domain=region if region is not None else VesselRegion(extension_diameters=1.0), cells_across=cells_across,
               stokes=True, scheme=Scheme.Steady())
    exp.add_medium(Medium("water", fills="vessel.lumen", fields={"flow": flow}))
    exp.declare("water.viscosity", Pa_s(MU), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    exp.attach(Inflow(steady=m3_per_s(FLUX)), medium="water", surface="vessel.root")
    for end, resistance in (laws if laws is not None else {"end_0": 2e9, "end_1": 6e9}).items():
        exp.attach(Windkessel(proximal=Pa_s_per_m3(resistance), compliance=m3_per_Pa(1e-18),
                              distal=Pa_s_per_m3(resistance)), medium="water", surface=f"vessel.{end}")
    if ledger:
        exp.observe(EnergyLedger())
    return exp


def _refused(exp: Experiment) -> list[str]:
    with pytest.raises(BuildRefused) as refused:
        exp.build()
    return [each.code for each in refused.value.refusals]


def test_an_open_tree_is_built_with_its_ends_as_openings_and_its_books_close(float64):
    built = tree_vessel().build()
    node = built.graph.get_node("water.flow")
    c = node.constants()
    assert c["openings"] == ("vessel.root", "vessel.end_0", "vessel.end_1")
    assert c["opening_planes"] == ("low", "high", "high")
    assert c["backflow_openings"] == (1, 2)
    text = built.report.text()
    assert "opening root: the upstream opening" in text and "opening end_0: an outlet by the centreline" in text
    result = built.run(ms(2.0))
    final = result.final["water.flow"]
    flux = np.asarray(final["opening_flux"])
    assert flux[0] == pytest.approx(-FLUX, rel=1e-9)
    assert abs(float(np.sum(flux))) < 1e-9 * FLUX
    assert np.all(flux[1:] > 0.0)
    assert float(final["wall_inexact"]) == 0.0
    assert ledger_closes(result).passed
    assert result.energy is not None and "water.flow.backflow" in result.energy.lost


def test_a_tree_with_no_root_is_refused_naming_its_upstream():
    rootless = TubeTree(points=mm((0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 1.2, 0.0, 2.6, -1.0, 0.0, 2.0)),
                        segments=((0, 2), (1, 0), (0, 3)), radii=mm((0.4, 0.4, 0.5, 0.5, 0.35, 0.35)))
    exp = tree_vessel(tree=rootless, laws={"end_1": 2e9, "end_2": 6e9})
    assert "flow.region-upstream" in _refused(exp) or "flow.opening-unknown" in _refused(exp)


def test_a_cut_around_the_junction_is_named_after_the_end_beyond_it(float64):
    built = tree_vessel(region=VesselRegion(around="j1", reach=mm(1.0), extension_diameters=1.0)).build()
    c = built.graph.get_node("water.flow").constants()
    assert c["openings"] == ("vessel.root", "vessel.end_0", "vessel.end_1")
    assert _refused(tree_vessel(region=VesselRegion(around="j7", reach=mm(1.0)))) == ["flow.region-cut"]


def test_the_outlet_watch_acts_on_nothing(float64, monkeypatch):
    """An observer (row #360): its line moved from never to always (the module's line, read when the step is traced),
    the flow and the openings bit for bit, its flag alone changed."""
    from mime.nodes.flow.fvm import _open_node

    runs = []
    for line in (math.inf, -1.0):
        monkeypatch.setattr(_open_node, "REVERSAL_LINE", line)
        runs.append(tree_vessel(ledger=False).build().run(ms(2.0)).final["water.flow"])
    for name in ("velocity_x", "velocity_y", "velocity_z", "opening_pressure", "opening_flux", "backflow_work"):
        np.testing.assert_array_equal(np.asarray(runs[0][name]), np.asarray(runs[1][name]))
    assert float(runs[0]["opening_reversed_flagged"]) == 0.0 and float(runs[1]["opening_reversed_flagged"]) == 1.0


def test_a_tilted_tube_is_built_with_bends_and_runs(float64):
    """A straight tube tilted 23 degrees from z at an azimuth on no symmetry plane, its centre off the grid by part of
    a cell: built through the region builder, each end bent onto its face, its books closing."""
    tilt = math.radians(23.0)
    axis = (math.sin(tilt) * math.cos(0.7), math.sin(tilt) * math.sin(0.7), math.cos(tilt))
    exp = Experiment("tilted", scene=Scene.new(up=Z, units="m"), dt=ms(1.0))
    exp.add_structure(Structure("tube", geometry=Tube(axis=axis, inner_radius=mm(0.5), wall=mm(0.2), length=mm(2.0),
                                                      centre=mm((0.013, -0.021, 0.0)))))
    exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": FVM(domain=VesselRegion(), cells_across=6,
                                                                           stokes=True, scheme=Scheme.Steady())}))
    exp.declare("water.viscosity", Pa_s(MU), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    exp.attach(Inflow(steady=m3_per_s(FLUX)), medium="water", surface="tube.start")
    exp.attach(Windkessel(proximal=Pa_s_per_m3(1e9), compliance=m3_per_Pa(1e-18), distal=Pa_s_per_m3(1e9)),
               medium="water", surface="tube.end")
    exp.observe(EnergyLedger())
    built = exp.build()
    c = built.graph.get_node("water.flow").constants()
    assert "tree_points" in c and c["backflow_openings"] == (1,)
    assert "a bend of 23 degrees" in built.report.text()
    result = built.run(ms(2.0))
    assert np.asarray(result.final["water.flow"]["opening_flux"]) == pytest.approx([-FLUX, FLUX], rel=1e-9)
    assert ledger_closes(result).passed


def test_the_two_outlet_shared_distal_law_binds_each_end_by_name(float64):
    """Scope 0's gate's wiring on the tree's unequal branches: the law over both ends, its first resistance at end_0;
    swapping the surfaces' order swaps which end takes which resistance, so the split moves (the twin rig's mirror could
    not see this)."""
    splits = []
    for surfaces in (("vessel.end_0", "vessel.end_1"), ("vessel.end_1", "vessel.end_0")):
        exp = tree_vessel(laws={})
        exp.attach(SharedDistal(first=Pa_s_per_m3(2e9), second=Pa_s_per_m3(8e9), distal=Pa_s_per_m3(1e9)),
                   medium="water", surface=surfaces)
        flux = np.asarray(exp.build().run(ms(2.0)).final["water.flow"]["opening_flux"])
        splits.append(flux[1] / FLUX)
    assert splits[0] > 0.5 > splits[1]


def test_a_backflow_term_of_the_wrong_sign_is_flagged_and_its_gift_counted(float64, monkeypatch):
    """R-15's sign check: an inflow reversing from its first step (Stokes, IMEX Euler) turns both outlets' flow
    inward, where the term works against it (negative, booked under ``<node>.backflow``); the term pushing the inflow
    in (beta -1) gives the flow energy: ``backflow_flagged`` and the ledger's gift count. The right sign raises
    neither."""
    from mime.nodes.flow.fvm import _open_node

    def reversing() -> Experiment:
        return _imex_tree(stokes=True, amplitude=2.0, phase=math.pi)

    clean = reversing().build().run(ms(3.0))
    monkeypatch.setattr(_open_node, "BACKFLOW_BETA", -1.0)
    flipped = reversing().build().run(ms(3.0))
    assert float(clean.final["water.flow"]["backflow_work"]) < 0.0
    assert float(clean.final["water.flow"]["backflow_flagged"]) == 0.0
    assert float(flipped.final["water.flow"]["backflow_flagged"]) == 1.0
    assert ledger_closes(clean).passed and not ledger_closes(flipped).passed


def _imex_tree(*, stokes: bool = False, amplitude: float = 0.3, phase: float = 0.0) -> Experiment:
    exp = Experiment("open_tree_imex", scene=Scene.new(up=Z, units="m"), dt=ms(1.0))
    exp.add_structure(Structure("vessel", geometry=TubeTree(**Y)))
    exp.add_medium(Medium("water", fills="vessel.lumen", fields={"flow": FVM(
        domain=VesselRegion(extension_diameters=1.0), cells_across=6, stokes=stokes, scheme=Scheme.IMEXEuler())}))
    exp.declare("water.viscosity", Pa_s(MU), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    exp.attach(Inflow(steady=m3_per_s(FLUX), amplitude=m3_per_s(amplitude * FLUX), frequency=Hz(5.0), phase=phase),
               medium="water", surface="vessel.root")
    exp.observe(EnergyLedger())
    for end, resistance in (("end_0", 2e9), ("end_1", 6e9)):
        exp.attach(Windkessel(proximal=Pa_s_per_m3(resistance), compliance=m3_per_Pa(1e-14),
                              distal=Pa_s_per_m3(resistance)), medium="water", surface=f"vessel.{end}")
    return exp


@pytest.mark.parametrize("path", ["water.viscosity", "windkessel:vessel.end_0.proximal",
                                  "windkessel:vessel.end_0.compliance", "windkessel:vessel.end_1.distal",
                                  "inflow:vessel.root.amplitude"])
def test_fb16_no_silent_bake_with_ports(float64, path):
    """FB16: two builds of the open tree differing in one declared value by a relative 1e-6 are one program (the
    unit fields, the port rows and the laws' states are made in the step from the leaves, round 5's item 13)."""
    from mime.testing import two_builds_of

    found = two_builds_of(_imex_tree(), path, relative=1e-6)
    assert found.verdict == "quiet", (found.verdict, found.lines)


def test_fb16_control_a_unit_field_made_at_build_is_caught(float64, monkeypatch):
    """The control: the node's unit fields started from one solved at build with the build's viscosity (a value made
    from a leaf and closed over) are a bake."""
    from mime.nodes.flow.fvm import _open_node
    from mime.testing import two_builds_of

    made = _open_node.OpenFlowNode.unit_field

    def baked(self, solve, force, start):
        scale = 1.0 + float(self.constants()["viscosity"])
        return made(self, solve, force, (start[0] * scale, start[1] * scale, start[2] * scale))

    monkeypatch.setattr(_open_node.OpenFlowNode, "unit_field", baked)
    assert two_builds_of(_imex_tree(), "water.viscosity", relative=1e-6).verdict == "bake"


def test_two_ends_on_one_plane_exchanging_flow_through_the_solid_are_flagged(float64):
    """The shared-plane exchange (#246's limit, large on a tree in a wide box against laws far above the branches'):
    flagged at the stiff interior 0.1h, and at 0.01h (the shared plane's default) more than ten times smaller, in the
    exchange and in the lumen's error against the openings' split, on a drawn tree of the two-outlet gate (laws 20 to
    50 times the branches': the flag can still fire at 0.01h there, and says so)."""
    from region_cases import draw, exchange_reading

    default, stiff = exchange_reading(draw(1), 0.1), exchange_reading(draw(1), 0.01)
    assert default["flagged"], default
    assert max(abs(x) for x in stiff["exchange"]) < 0.1 * max(abs(x) for x in default["exchange"]), (default, stiff)
    assert abs(default["lumen_minus_split"]) > 10.0 * abs(stiff["lumen_minus_split"])
