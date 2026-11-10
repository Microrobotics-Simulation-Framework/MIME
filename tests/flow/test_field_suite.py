"""SF-12 on both field kinds (F1b's gate 6): the FVM (IMEX Euler, its steady limit, RK3) and the prescribed flow pass
the same suite, each case applicable by what the kind declares and every skipped case listed with its reason; and the
suite's controls: a declared capability whose case is absent, a reading in the wrong units or the wrong shape, a kind
declaring it is outside that takes a delivery, an open row's value on a decided row."""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special
from fvm_cases import METERED, Waveform, in_dtype, step_of, tube_node

from mime.dynamics.flow._fields import fvm_field
from mime.dynamics.flow._womersley import Womersley
from mime.fields._solved import NOT_DECLARED
from mime.nodes.flow._waveform import waveform
from mime.nodes.flow._womersley import PrescribedFlowNode
from mime.nodes.flow.fvm import ExplicitFlowNode, ImplicitFlowNode, periodic_tube_layout
from mime.nodes.flow.fvm._reading import velocity_at
from mime.testing._field_kinds import FieldCase, FieldRig, field_conformance
from mime.vocabulary import FLOW_VELOCITY

V = FLOW_VELOCITY.name


def _kind(accepts):
    """A kind's entry as the suite reads it: its class's ``accepts``."""
    return type("FVM", (), {"accepts": accepts})()


def _scan(node, p, steps, drive, state=None):
    step = step_of(node, p, drive)
    start = dict(node.initial_state()) if state is None else state
    return dict(jax.jit(lambda s: jax.lax.scan(lambda c, _: (step(c), None), s, None, length=steps)[0])(start))


def _poiseuille_points(layout, count=7):
    """Points across the lumen's core (r up to 0.6 R) from the box's corner, and Poiseuille's velocity there."""
    r = np.linspace(0.0, 0.6 * layout.radius, count)
    points = np.stack([layout.centre[0] + r, np.full(count, layout.centre[1]), np.full(count, 0.5 * layout.spacing)],
                      axis=-1)
    return points, r


def fvm_rig(scheme: str, dtype: str, opened: bool = False) -> tuple[object, FieldRig]:
    """A D/h 16 aligned tube at its steady state under a body force, its reading against Poiseuille in the core; on an
    open vessel (``opened``) its affine case the open node's."""
    g, nu, dh = 8.0, 1.0, 16
    layout = periodic_tube_layout(0.5, 4 / dh, cells_across=dh)
    h = layout.spacing
    with in_dtype(dtype):
        if scheme == "rk3":
            dt = 0.15 * h * h / nu
            node = ExplicitFlowNode("flow", dt, cells_x=layout.cells[0], cells_y=layout.cells[1],
                                    cells_z=layout.cells[2], spacing=h, viscosity=nu, density=1.0,
                                    tube_centre=layout.centre, tube_axis=layout.axis, tube_radius=layout.radius)
            steps = 3000
        else:
            node = tube_node(dh, nu=nu, g=g, dt=0.05, convection=False, scheme=scheme)
            steps = 1 if scheme == "steady" else 60
        p = node._step_constants(None)  # noqa: SLF001
        state = _scan(node, p, steps, Waveform(steady=g))
    points, r = _poiseuille_points(layout)
    exact = np.zeros((len(r), 3))
    exact[:, 2] = g * (0.25 - r ** 2) / (4 * nu)

    def reading(params, s, pts):
        grid = node.grid(params)
        return velocity_at(grid, (s["velocity_x"], s["velocity_y"], s["velocity_z"]), jnp.asarray(pts))

    def metered():
        with in_dtype(dtype):
            if scheme == "rk3":
                m = ExplicitFlowNode(**{**_arguments(node), "ledger": METERED})
            else:
                m = tube_node(8, dt=0.002, convection=scheme != "steady", ledger=METERED, scheme=scheme)
            return _scan(m, m._step_constants(None), 10, Waveform(steady=g))  # noqa: SLF001

    def steady():
        with in_dtype(dtype):
            stepped = tube_node(dh, nu=nu, g=g, dt=0.05, convection=False, scheme="imex_euler", **{
                f"initial_velocity_{c}": np.asarray(state[f"velocity_{c}"]) for c in "xyz"})
            after = _scan(stepped, stepped._step_constants(None), 1, Waveform(steady=g))  # noqa: SLF001
        moved = max(float(np.max(np.abs(np.asarray(after[f"velocity_{c}"]) - np.asarray(state[f"velocity_{c}"]))))
                    for c in "xyz")
        return moved / float(np.max(np.abs(np.asarray(state["velocity_z"])))), (1e-6 if dtype == "float64" else 1e-4)

    def affine():
        """FB4's monolithic reference on the open box: a drawn pulsatile run's last step against one cold solve of it."""
        from open_cases import draw, monolithic_distance, point_node, scan

        with in_dtype(dtype):
            open_node = point_node(draw(0), ledger=False, scheme=scheme, convection=scheme != "steady")
            before, _ = scan(open_node, 4)
            after, _ = scan(open_node, 1, state=before)
            return monolithic_distance(open_node, before, after), AFFINE_LINE[dtype]

    rig = FieldRig(node, p, state, {V: reading}, points, {V: exact}, 0.05, metered=metered,
                   steady=steady if scheme == "steady" else None, affine=affine if opened else None)
    from mime.vocabulary import FLOW_BODY_FORCE, FLOW_OPENING, FLOW_WAVEFORM
    from mime.registry import Port

    return _kind({FLOW_BODY_FORCE.name: Port("body_force"), FLOW_WAVEFORM.name: Port("waveform"),
                  FLOW_OPENING.name: Port("openings")}), rig


#: The affine case's line: the step's superposed solution against the monolithic solve, over the largest speed (the
#: step's own tolerance, 1e-8 and float32's floor, through the operator's conditioning): on draw 0 while writing, 4.6e-10
#: and 1.8e-6 clean, a unit field scaled by 1.1 2.7e-4 in float32; the drawn spread is the probe's (the PR's table).
AFFINE_LINE = {"float64": 1e-6, "float32": 3e-5}


def _arguments(node):
    c = dict(node.constants())
    return {"name": "flow", "timestep": node.delta_t, **{k: v for k, v in c.items() if k != "ledger"}}


def womersley_rig(dtype: str) -> tuple[object, FieldRig]:
    """The prescribed flow at alpha 6 with a steady part, its reading against SciPy's Womersley and Poiseuille."""
    radius, nu, g0, amp = 0.5e-3, 1e-6, 2.0, 3.0
    omega = (6.0 / radius) ** 2 * nu
    with in_dtype(dtype):
        node = PrescribedFlowNode("flow", 2 * math.pi / omega / 200, viscosity=1e-3, density=1e3,
                                  tube_centre=(0.0, 0.0, 0.0), tube_axis=(0.0, 0.0, 1.0), tube_radius=radius)
        p = node._step_constants(None)  # noqa: SLF001
        w = waveform((0.0, 0.0, 1.0), jnp.asarray(g0), amp, omega / (2 * math.pi), 0.3)
        state = dict(jax.lax.scan(lambda s, _: (node.advance(p, s, {"waveform": w}, node.delta_t), None),
                                  dict(node.initial_state()), None, length=77)[0])
    r = np.linspace(0.0, 0.95 * radius, 9)
    points = np.stack([r, np.zeros(9), np.zeros(9)], axis=-1)
    t = 77 * node.delta_t
    rot = np.exp(-1j * np.pi / 4)
    harmonic = np.real((amp / (1j * omega)) * (1 - scipy.special.jv(0, rot * 6.0 * r / radius)
                                               / scipy.special.jv(0, rot * 6.0)) * np.exp(1j * (omega * t + 0.3)))
    exact = np.zeros((9, 3))
    exact[:, 2] = g0 * (radius ** 2 - r ** 2) / (4 * nu) + harmonic
    tolerance = 1e-10 if dtype == "float64" else 1e-4
    return Womersley(), FieldRig(node, p, state, {V: node.velocity_at}, points, {V: exact}, tolerance)


KINDS = {"imex_euler": lambda d: fvm_rig("imex_euler", d), "steady": lambda d: fvm_rig("steady", d),
         "rk3": lambda d: fvm_rig("rk3", d), "womersley": womersley_rig,
         "imex_euler_open": lambda d: fvm_rig("imex_euler", d, opened=True),
         "steady_open": lambda d: fvm_rig("steady", d, opened=True)}


def declared(name):
    """Each kind's declarations: a periodic box's by its scheme, an open vessel's (``_open``) with its ports."""
    if name == "womersley":
        return Womersley.field
    return fvm_field(name.removesuffix("_open"), opened=name.endswith("_open"))


@pytest.mark.parametrize("name", sorted(KINDS))
def test_each_kind_passes_the_suite_and_lists_what_it_skipped(dtype, name):
    kind, rig = KINDS[name](dtype)
    report = field_conformance(kind, declared(name), rig, dtype=dtype)
    assert report.passed, report.failures()
    assert (FieldCase.AFFINE in report.skipped) == (not name.endswith("_open"))
    if name == "womersley":
        assert set(report.skipped) == {FieldCase.LEDGER, FieldCase.STEADY, FieldCase.AFFINE}
        assert "outside" in report.skipped[FieldCase.LEDGER]
    else:
        assert FieldCase.OUTSIDE in report.skipped
        assert (FieldCase.STEADY in report.skipped) == (name.removesuffix("_open") != "steady")


@pytest.mark.parametrize("entry", ["fvm", "womersley"])
def test_a_kinds_declared_settings_are_its_entrys_own(entry):
    """SF-8: the settings a kind declares are its entry's own, by name (the entry's fields past the base's ``rate``):
    the FVM's wall group, ``wall=Band(...)``, replaced its flat band and interior. The control is the declaration as it
    stood before that, which names two settings the entry no longer has."""
    from mime.dynamics import FVM
    from mime.registry import Dynamics

    cls, declared = (FVM, fvm_field("rk3")) if entry == "fvm" else (Womersley, Womersley.field)
    base = {f.name for f in dataclasses.fields(Dynamics)}
    own = tuple(f.name for f in dataclasses.fields(cls) if f.name not in base)
    assert declared.settings == own
    if entry == "fvm":
        stale = ("domain", "spacing", "cells_across", "scheme", "stokes", "tolerance", "preconditioner", "band",
                 "interior")
        assert stale != own


def test_a_case_left_out_is_listed_as_left_out(float64):
    kind, rig = womersley_rig("float64")
    report = field_conformance(kind, Womersley.field, rig, cases=[FieldCase.DECLARATIONS])
    assert report.skipped[FieldCase.READING] == "left out of the cases asked for"


# -- the controls ---------------------------------------------------------------------------------------------------

def test_control_a_declared_steady_solve_whose_case_is_absent_fails(float64):
    kind, rig = womersley_rig("float64")
    report = field_conformance(kind, dataclasses.replace(Womersley.field, steady_solve=True), rig)
    assert not report.passed and any(f.startswith("steady") for f in report.failures())


@pytest.mark.parametrize("plant", ["units", "shape"])
def test_control_a_reading_in_the_wrong_units_or_shape_fails(dtype, plant):
    kind, rig = womersley_rig(dtype)
    read = rig.reading[V]
    wrong = (lambda p, s, x: 1e3 * read(p, s, x)) if plant == "units" else (lambda p, s, x: read(p, s, x)[:, 2])
    report = field_conformance(kind, Womersley.field, dataclasses.replace(rig, reading={V: wrong}))
    assert not report.passed and any(f.startswith("reading") for f in report.failures())


class _TakesALoad(PrescribedFlowNode):
    def boundary_input_spec(self):
        from mime.core import node_input

        return {**super().boundary_input_spec(), "load": node_input("N", (3,), additive=True)}


def test_control_a_kind_declaring_it_is_outside_that_takes_a_delivery_fails(float64):
    kind, rig = womersley_rig("float64")
    planted = _TakesALoad("flow", rig.node.delta_t, **{k: v for k, v in rig.node.constants().items()})
    report = field_conformance(kind, Womersley.field, dataclasses.replace(rig, node=planted))
    assert not report.passed and any(f.startswith("outside") for f in report.failures())


def test_control_an_open_rows_value_on_a_decided_row_fails(float64):
    kind, rig = womersley_rig("float64")
    report = field_conformance(kind, dataclasses.replace(Womersley.field, stores=NOT_DECLARED), rig)
    assert not report.passed and any("stores" in f for f in report.failures())


def test_control_a_route_one_kind_whose_steps_give_energy_fails(dtype, monkeypatch):
    from mime.nodes.flow.fvm import _rk3

    original = _rk3.step

    def gifted(*args, **recorded):
        u, p, parts = original(*args, **recorded)
        return tuple(1.5 * c for c in u), p, parts  # the kinetic energy x 2.25 a step, far over its dissipation

    monkeypatch.setattr(_rk3, "step", gifted)
    kind, rig = fvm_rig("rk3", dtype)
    report = field_conformance(kind, fvm_field("rk3"), rig, cases=[FieldCase.LEDGER])
    assert not report.passed


def test_a_prescribed_flow_in_a_medium_that_fills_no_tube_is_refused(float64):
    from mime.build import BuildRefused
    from mime.experiment import Experiment
    from mime.interactions import BodyForce
    from mime.scene import Scene, Z
    from mime.things import Medium
    from mime.units import kg_per_m3, m_per_s2, mPa_s, ms

    exp = Experiment("free", scene=Scene.new(up=Z, units="m"), dt=ms(1.0))
    exp.add_medium(Medium("water", fields={"flow": Womersley()}))
    exp.declare("water.viscosity", mPa_s(1.0), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(1.0)), medium="water")
    with pytest.raises(BuildRefused) as raised:
        exp.build()
    assert "flow.prescribed-region" in [r.code for r in raised.value.refusals]


@pytest.mark.parametrize("plant", ["unit_scaled", "unit_negated"])
def test_control_the_affine_case_fails_on_a_wrong_unit_field(dtype, monkeypatch, plant):
    """The affine case can fail: a unit field scaled by 1.1, or negated, moves the superposed solution from the
    monolithic solve past its line (in pulsatile flow, where the step's pressures move)."""
    from open_cases import PLANTS

    PLANTS[plant](monkeypatch.setattr)
    kind, rig = fvm_rig("imex_euler", dtype, opened=True)
    del kind
    assert rig.affine is not None
    moved, line = rig.affine()
    assert moved > line, (moved, line)
