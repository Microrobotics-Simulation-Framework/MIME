"""The generator of a screw surrogate's training data, and its held-out protocol (phase 7i).

**What it computes.** The boundary-element resistance of de Jongh's screw bodies resting on the floor of a straight
tube, through ``Drag.BEM`` in the ``tube`` wall kind, which reads the corrected table of the unit tube's image system
(MIME-ANO-117); never the 2026-05-21 tables. One experiment is built per design, tube and discretisation, and its drag
node is evaluated at each case's pose: the body's axis along the tube's, its centre ``offset`` from the tube's axis
towards the floor, so that the gap from its outer radius to the wall is the case's ``gap``. The resistance is
delivered in the surrogate's canonical frame (the offset's direction, the axis crossed with it, the tube's axis) about
the body's centroid, in SI.

**The domain** (``DOMAIN``) is declared from measurement: de Jongh's seventeen designs (FL-1 to FL-11, FW-1 to FW-6;
de Jongh et al. 2025, Tables 1 and 2, as the 0.2 line transcribed them) in the four tubes, each resting on the floor
at a contact margin between the two ``MARGIN`` bounds. Its near-wall edge is the smallest gap the boundary-element
method resolves at a spacing, measured as a multiple of the spacing for each use (``near_wall_ratios``, MIME-ANO-114's
near-wall check, recorded in ``tests/differential/references/near_wall_check.json``): the swim speed is resolved
nearer the wall than the resistance as a whole, whose normal translation and pitching rotation need the most room.
Below a use's edge its near-wall part is the ``Lubrication`` blend's, the hand-over gap. What lies outside is named in
``Domain.outside``.

**The protocol.** Training cases are every design in every tube at margins spaced geometrically from the swim speed's
near-wall edge to the largest margin, the body unturned about its axis. Two held-out sets are disjoint from them by
construction: ``between``, every design and tube at the geometric midpoints of the training margins (the gaps where the
0.2 line's surrogate missed by the most, MIME-ANO-114), which estimates the surrogate's error inside the domain; and
``roll``, every design in every tube at the middle training margin with the body turned about its own axis, an input
the surrogate does not read, which measures what leaving it out costs. ``accuracy`` judges a surrogate on a held-out
set per use: the resistance, block by block, and the swim speed. What it estimates is the surrogate's error against
this boundary-element method at this discretisation; it bounds nothing about the flow, since the method's own error
(MIME-ANO-116, the near-wall edge) is in neither side.

Each case's checks are part of the build: the table read every sample (``wall_accepted``), the resistance is finite,
symmetric and positive definite. The builder ``TrainingData``, run through ``mime.assets.derive``, refuses a set with a
failed check, so a failed check produces no data, and its derivation record names the tube's table by hash as its
``wall:tube`` input. The generator runs in float64 and refuses float32.

This module imports the experiment layer; ``mime.surrogates`` does not import it, so that the interactions, which read
the packaged surrogates, can be imported without it."""

from __future__ import annotations

import io
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import jax
import numpy as np

from mime.assets import AssetRef, CheckResult, Json, Produced
from mime.surrogates._type import Accuracy

#: The contact margin's range, in metres: the gap a body resting on the tube's floor keeps from the wall.
MARGIN = (0.02e-3, 0.20e-3)
#: How many training margins the protocol spaces from the near-wall edge to the largest margin.
TRAINING_MARGINS = 7
#: The body's turns about its own axis in the held-out ``roll`` set: a two-start screw's section repeats every half
#: turn, so these span its period.
ROLLS = (math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0)
#: The checks' tolerances, declared before any data: the resistance's asymmetry (relative to its largest entry) and its
#: smallest eigenvalue's floor (relative to its largest).
ASYMMETRY_TOLERANCE = 1e-8
EIGENVALUE_FLOOR = 0.0
#: The canonical frame's columns in the world of ``experiment``: the offset's direction (towards the floor, -z), the
#: tube's axis crossed with it (+y) and the tube's axis (+x).
CANONICAL = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


@dataclass(frozen=True)
class Design:
    """A screw body of de Jongh et al. (2025): its ``name``, ``wavenumber`` (``nu``), ``length`` and reference
    ``radius`` (m), and the section's ``modulation`` and ``starts``, fixed across both groups."""

    name: str
    wavenumber: float
    length: float
    radius: float = 1.56e-3
    modulation: float = 0.33
    starts: int = 2

    @property
    def outer_radius(self) -> float:
        """The body's radial reach, ``(1 + modulation) R``, from which its gap to the wall is measured."""
        return (1.0 + self.modulation) * self.radius


_FL = (0.25, 0.5, 1.0, 1.25, 1.4, 1.55, 1.8, 2.0, 2.33, 3.0, 3.5)
_FW = ((1.0, 9.73e-3), (1.25, 7.77e-3), (1.4, 6.88e-3), (1.55, 6.29e-3), (2.0, 4.82e-3), (2.33, 4.13e-3))
#: De Jongh et al. (2025), Table 1 (FL-1 to FL-11: length 7.47 mm) and Table 2 (FW-1 to FW-6: one turn each).
DESIGNS: tuple[Design, ...] = (
    *(Design(f"FL-{index}", nu, 7.47e-3) for index, nu in enumerate(_FL, start=1)),
    *(Design(f"FW-{index}", nu, length) for index, (nu, length) in enumerate(_FW, start=1)),
)
#: The tubes of de Jongh et al.'s experiments, by inner diameter, each with its inner radius in metres.
TUBES: Mapping[str, float] = {"3/16 in": 2.38125e-3, "1/4 in": 3.175e-3, "3/8 in": 4.7625e-3, "1/2 in": 6.35e-3}


def design_named(name: str) -> Design:
    (found,) = (design for design in DESIGNS if design.name == name)
    return found


@dataclass(frozen=True)
class Case:
    """One body resting on one tube's floor: its ``design``, the ``tube`` (a key of ``TUBES``), the ``gap`` from its
    outer radius to the wall (m) and its ``roll`` about its own axis (rad)."""

    design: Design
    tube: str
    gap: float
    roll: float = 0.0

    @property
    def tube_radius(self) -> float:
        return TUBES[self.tube]

    @property
    def offset(self) -> float:
        """The body's centre's distance from the tube's axis, towards the floor."""
        return self.tube_radius - self.design.outer_radius - self.gap

    def key(self) -> str:
        return f"{self.design.name} {self.tube} gap {self.gap * 1e3:.6f} mm roll {self.roll:.6f}"


@dataclass(frozen=True)
class Discretisation:
    """How ``Drag.BEM`` samples the body: neighbouring samples about ``spacing`` (m) apart, the blob ``blob`` times
    their mean spacing, on ``patches`` (every patch of the kind, its closed surface, when ``None``)."""

    spacing: float
    blob: float = 0.5
    patches: tuple[str, ...] | None = None

    def as_json(self) -> dict[str, Json]:
        return {"spacing": self.spacing, "blob": self.blob,
                "patches": list(self.patches) if self.patches is not None else None}


@dataclass(frozen=True)
class Domain:
    """Where a surrogate trained by this protocol applies: the ``designs`` in the ``tubes``, each resting on the floor
    at a contact margin in ``margin``, and, for each use, no nearer the wall than its ``near_wall_ratios`` entry times
    the discretisation's spacing (the near-wall edge, measured), where the ``Lubrication`` blend takes the body over.
    ``outside`` names what it does not cover."""

    designs: tuple[Design, ...]
    tubes: Mapping[str, float]
    margin: tuple[float, float]
    near_wall_ratios: Mapping[str, float]
    outside: tuple[str, ...]

    def edge(self, spacing: float, use: str = "swim_speed") -> float:
        """The near-wall edge of ``use`` at ``spacing``, in metres: its hand-over gap to the ``Lubrication`` blend."""
        return self.near_wall_ratios[use] * spacing

    def problems(self, case: Case, spacing: float, use: str = "swim_speed") -> list[str]:
        """Why ``case`` lies outside the domain at ``spacing`` for ``use``, each as a sentence; none inside it."""
        found: list[str] = []
        if case.design not in self.designs:
            found.append(f"{case.design.name} is not one of the domain's designs")
        if case.tube not in self.tubes:
            found.append(f"{case.tube!r} is not one of the domain's tubes, {sorted(self.tubes)}")
        low, high = max(self.margin[0], self.edge(spacing, use)), self.margin[1]
        if not low * (1.0 - 1e-12) <= case.gap <= high * (1.0 + 1e-12):
            found.append(f"the gap {case.gap!r} m lies outside {low!r} to {high!r} m (the contact margin's range, "
                         f"from the near-wall edge of the {use} at the spacing {spacing!r} m)")
        return found

    def as_json(self) -> dict[str, Json]:
        return {"designs": [design.name for design in self.designs], "tubes": dict(self.tubes),
                "margin": list(self.margin), "near_wall_ratios": dict(self.near_wall_ratios),
                "outside": list(self.outside)}


#: The near-wall edge as a multiple of the spacing, by use, from the near-wall check (MIME-ANO-114): the declared
#: analysis (``near_wall_ratio``, every quantity of ``NEAR_WALL_USES[use]`` within ``NEAR_WALL_TOLERANCE``) read on
#: the recorded check, to which ``tests/differential/test_near_wall_check.py`` holds these. ``resistance`` is the whole
#: resistance (the criterion declared before the check ran); ``swim_speed`` its one quantity, read from the same data
#: after it ran. At every spacing the check reached, the whole resistance's edge lies past the largest contact margin.
NEAR_WALL_RATIOS: Mapping[str, float] = {"resistance": 0.5e-3 / 0.24e-3, "swim_speed": 0.075e-3 / 0.12e-3}
#: The domain of a screw surrogate trained by this protocol.
DOMAIN = Domain(
    designs=DESIGNS,
    tubes=TUBES,
    margin=MARGIN,
    near_wall_ratios=NEAR_WALL_RATIOS,
    outside=(
        "free space, and any tube but the four (kappa is trained at their four values only)",
        "any screw but the seventeen designs (another wavenumber or length is read between designs, untrained)",
        "a body off the floor: centred, or resting with a gap above the largest contact margin",
        "a gap below a use's near-wall edge, where the boundary-element method does not resolve the wall and the "
        "Lubrication blend takes the body over: for the resistance as a whole (its translation normal to the wall and "
        "its pitching rotation), the whole contact margin at every spacing the near-wall check reached",
        "a body turned about its own axis from the trained roll (an input the surrogate does not read; the held-out "
        "roll set measures what it costs)",
        "a body tilted from the tube's axis, or not on the floor's side the offset names",
        "another modulation or number of starts, a body with no flat caps, a flow in the tube, and Reynolds numbers "
        "above zero",
    ),
)


def training_margins(domain: Domain, spacing: float, count: int = TRAINING_MARGINS) -> tuple[float, ...]:
    """The training margins at ``spacing``: ``count`` gaps spaced geometrically from the swim speed's near-wall edge (or
    the smallest margin, whichever is larger) to the largest margin; none when the edge lies past the largest."""
    low, high = max(domain.margin[0], domain.edge(spacing)), domain.margin[1]
    if low > high:
        return ()
    return tuple(float(gap) for gap in np.geomspace(low, high, count))


def training_cases(domain: Domain, spacing: float, count: int = TRAINING_MARGINS) -> tuple[Case, ...]:
    """Every design in every tube at each training margin, unturned about its axis."""
    return tuple(Case(design, tube, gap) for design in domain.designs for tube in domain.tubes
                 for gap in training_margins(domain, spacing, count))


def held_out_cases(domain: Domain, spacing: float, count: int = TRAINING_MARGINS) -> dict[str, tuple[Case, ...]]:
    """The held-out sets at ``spacing``: ``between`` (every design and tube at the geometric midpoints of the training
    margins) and ``roll`` (every design and tube at the middle training margin, turned by each of ``ROLLS``)."""
    gaps = training_margins(domain, spacing, count)
    if not gaps:
        return {"between": (), "roll": ()}
    midpoints = [math.sqrt(a * b) for a, b in zip(gaps, gaps[1:], strict=False)]
    middle = gaps[len(gaps) // 2]
    return {
        "between": tuple(Case(design, tube, gap) for design in domain.designs for tube in domain.tubes
                         for gap in midpoints),
        "roll": tuple(Case(design, tube, middle, roll) for design in domain.designs for tube in domain.tubes
                      for roll in ROLLS),
    }


def overlap(first: Iterable[Case], second: Iterable[Case]) -> list[str]:
    """The cases two sets share, by key: none for a held-out set and the training set."""
    return sorted({case.key() for case in first} & {case.key() for case in second})


@dataclass(frozen=True)
class Sample:
    """A case's boundary-element resistance: ``resistance`` in the canonical frame about the body's centroid (SI),
    whether every table read was ``accepted``, and the number of surface ``samples``."""

    case: Case
    resistance: np.ndarray
    accepted: bool
    samples: int

    def swim(self) -> np.ndarray:
        """The force-free velocity per unit spin about the tube's axis (m per radian), in the canonical frame."""
        return swim_per_spin(self.resistance)

    def checks(self) -> tuple[CheckResult, ...]:
        """The case's checks: every table read accepted, the resistance finite, symmetric and positive definite."""
        held = self.resistance
        finite = bool(np.all(np.isfinite(held)))
        scale = float(np.max(np.abs(held))) if finite else math.nan
        asymmetry = float(np.max(np.abs(held - held.T))) / scale if finite else math.nan
        smallest = float(np.min(np.linalg.eigvalsh((held + held.T) / 2.0))) / scale if finite else math.nan
        return (
            CheckResult("wall_accepted", float(self.accepted), "every table read inside its readable range",
                  self.accepted),
            CheckResult("finite", float(finite), "every entry finite", finite),
            CheckResult("asymmetry", asymmetry, f"at most {ASYMMETRY_TOLERANCE:g} of the largest entry",
                  asymmetry <= ASYMMETRY_TOLERANCE),
            CheckResult("smallest_eigenvalue", smallest, f"above {EIGENVALUE_FLOOR:g} of the largest entry",
                  smallest > EIGENVALUE_FLOOR),
        )


def swim_per_spin(resistance: np.ndarray) -> np.ndarray:
    """``-R_FU^-1 R_FO e_z``: the force-free velocity of a body turned about the canonical frame's axis at unit
    rate."""
    return -np.linalg.solve(resistance[:3, :3], resistance[:3, 3:] @ np.array([0.0, 0.0, 1.0]))


def into_canonical(resistance: np.ndarray) -> np.ndarray:
    """A world resistance of ``experiment``'s body turned into the canonical frame: ``T^T R T``, ``T = diag(F, F)``
    with ``F`` the canonical frame's columns. A roll turns the body, not the frame."""
    turn = np.zeros((6, 6))
    turn[:3, :3] = turn[3:, 3:] = CANONICAL
    return turn.T @ resistance @ turn


def _require_float64() -> None:
    if not bool(cast(Any, jax).config.jax_enable_x64):
        raise RuntimeError("the generator computes in float64: set JAX_ENABLE_X64=1 (training data are never drawn "
                           "in float32)")


def experiment(design: Design, tube: str, discretisation: Discretisation, *, wall_kind: str = "tube") -> Any:
    """``design`` resting in ``tube`` (along x, its floor at -z) in water, with ``Drag.BEM`` and nothing else, at its
    smallest training gap's pose; ``BEMEvaluator`` moves it to each case's."""
    from mime.dynamics import RigidInertial
    from mime.experiment import Experiment
    from mime.interactions import Drag
    from mime.scene import Scene, Screw, Tube, Z
    from mime.things import Body, Medium, Part, State, Structure
    from mime.units import dimensionless, kg, kg_m2, m, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s

    held = Experiment(f"generator_{design.name}_{tube}".replace("-", "_").replace("/", "_").replace(" ", "_"),
                      scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    held.add_structure(Structure("tube", geometry=Tube(axis=(1.0, 0.0, 0.0), inner_radius=m(TUBES[tube]),
                                                       wall=mm(0.5), length=mm(200.0))))
    held.add_medium(Medium("water", fills="tube.lumen"))
    held.declare("water.viscosity", mPa_s(1.0), reason="water; the resistance is linear in it")
    screw = Screw(radius=m(design.radius), length=m(design.length), wavenumber=dimensionless(design.wavenumber),
                  modulation=dimensionless(design.modulation), starts=design.starts)
    resting = Case(design, tube, MARGIN[1]).offset
    held.add_body(Body("screw", dynamics=RigidInertial(), parts={"shell": Part(screw)},
                       initial=State(position=m((0.0, 0.0, -resting)), orientation=(1.0, 0.0, 0.0, 0.0),
                                     velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    held.declare("screw.mass", kg(1e-5), reason="a millimetre body's order; no step is run")
    held.declare("screw.inertia", kg_m2((1e-11, 1e-11, 1e-11)), reason="a millimetre body's order; no step is run")
    settings: dict[str, Any] = {"spacing": m(discretisation.spacing), "blob": discretisation.blob,
                                "wall_kind": wall_kind}
    if discretisation.patches is not None:
        settings["patches"] = discretisation.patches
    held.attach(Drag.BEM(**settings), body="screw", medium="water")
    return held


class BEMEvaluator:
    """One design in one tube at one discretisation: the experiment built once, its drag node compiled once, and
    evaluated at each case's pose (its step's ``params``, the tube's table among them, are arguments, so a pose is
    one call)."""

    def __init__(self, design: Design, tube: str, discretisation: Discretisation, *, wall_kind: str = "tube") -> None:
        _require_float64()
        from mime.build._maddening import node_state, parameter_leaves

        self.design, self.tube, self.discretisation = design, tube, discretisation
        built = experiment(design, tube, discretisation, wall_kind=wall_kind).build()
        node = built.graph.get_node("drag:screw")
        state = node_state(built.graph, "drag:screw")
        self._params = parameter_leaves(built.graph)["drag:screw"]
        self.samples = int(sum(math.prod(counts) * (2 if patch == "caps" else 1) for patch, counts in
                               zip(node.constants()["patches"], node.constants()["counts"], strict=True)))
        def step(params: Any, pose: Any) -> Any:
            return node.update(state, {"pose": pose}, 1e-3, params=params)

        self._step: Callable[[Any, Any], Any] = cast(Any, jax).jit(step)

    def pose(self, case: Case) -> np.ndarray:
        """The body's pose for ``case``: its centre ``offset`` below the tube's axis, turned by ``roll`` about it."""
        half = case.roll / 2.0
        return np.array([0.0, 0.0, -case.offset, math.cos(half), math.sin(half), 0.0, 0.0])

    def __call__(self, case: Case) -> Sample:
        if (case.design, case.tube) != (self.design, self.tube):
            raise ValueError(f"{case.key()} is not {self.design.name} in {self.tube!r}")
        out = self._step(self._params, cast(Any, jax).numpy.asarray(self.pose(case)))
        world = np.asarray(out["resistance"], dtype=np.float64)
        return Sample(case, into_canonical(world), bool(out["wall_accepted"]), self.samples)


def evaluate(cases: Sequence[Case], discretisation: Discretisation, *,
             report: Callable[[Sample], None] | None = None) -> list[Sample]:
    """Every case's sample, one evaluator per design and tube, in the cases' order."""
    evaluators: dict[tuple[str, str], BEMEvaluator] = {}
    found: list[Sample] = []
    for case in cases:
        key = (case.design.name, case.tube)
        if key not in evaluators:
            evaluators[key] = BEMEvaluator(case.design, case.tube, discretisation)
        sample = evaluators[key](case)
        found.append(sample)
        if report is not None:
            report(sample)
    return found


def samples_npz(samples: Sequence[Sample]) -> bytes:
    """The samples as an ``.npz`` file: each case's design (name, wavenumber, length, radius, modulation, starts), tube
    radius, gap, offset and roll, and its resistance, acceptance and sample count."""
    buffer = io.BytesIO()
    cases = [sample.case for sample in samples]
    np.savez(buffer, allow_pickle=False,
             design=np.array([case.design.name for case in cases]),
             wavenumber=np.array([case.design.wavenumber for case in cases]),
             length=np.array([case.design.length for case in cases]),
             radius=np.array([case.design.radius for case in cases]),
             modulation=np.array([case.design.modulation for case in cases]),
             starts=np.array([case.design.starts for case in cases]),
             tube=np.array([case.tube for case in cases]),
             tube_radius=np.array([case.tube_radius for case in cases]),
             gap=np.array([case.gap for case in cases]),
             offset=np.array([case.offset for case in cases]),
             roll=np.array([case.roll for case in cases]),
             resistance=np.array([sample.resistance for sample in samples]).reshape(len(samples), 6, 6),
             accepted=np.array([sample.accepted for sample in samples], dtype=bool),
             samples=np.array([sample.samples for sample in samples]),
             frame=np.array("canonical: the offset's direction, the tube's axis crossed with it, the tube's axis; "
                            "about the centroid; SI"))
    return buffer.getvalue()


@dataclass(frozen=True)
class TrainingData:
    """The builder of a screw surrogate's training data (``mime.assets.Builder``): the protocol's cases (``split``,
    ``"training"`` or a held-out set's name, or ``cases`` given) at ``discretisation``, through the tube's table
    (the input ``wall:tube``); every case's checks are the build's."""

    name: str = "screw_training_data"
    version: str = "0.1.0"
    domain: Domain = DOMAIN

    def cases(self, parameters: Mapping[str, Json]) -> tuple[Case, ...]:
        spacing = float(cast(float, cast(Mapping[str, Json], parameters["discretisation"])["spacing"]))
        split = str(parameters["split"])
        if split == "training":
            return training_cases(self.domain, spacing)
        if split == "listed":
            listed = cast(list[Mapping[str, Json]], parameters["cases"])
            return tuple(Case(design_named(str(each["design"])), str(each["tube"]), float(cast(float, each["gap"])),
                              float(cast(float, each.get("roll", 0.0)))) for each in listed)
        return held_out_cases(self.domain, spacing)[split]

    def produce(self, parameters: Mapping[str, Json], inputs: Mapping[str, AssetRef]) -> Produced:
        from mime.nodes.hydro.bem._tube import UNIT_TUBE

        given = inputs.get("wall:tube")
        if given is None or given.name != UNIT_TUBE:
            raise ValueError(f"the generator reads the tube wall kind's table, {UNIT_TUBE!r}, as its input 'wall:tube'")
        held = cast(Mapping[str, Json], parameters["discretisation"])
        patches = held.get("patches")
        discretisation = Discretisation(float(cast(float, held["spacing"])), float(cast(float, held["blob"])),
                                        tuple(cast(list[str], patches)) if patches is not None else None)
        samples = evaluate(self.cases(parameters), discretisation)
        checks: list[CheckResult] = []
        for sample in samples:
            checks += [CheckResult(f"{sample.case.key()}: {check.name}", check.measured, check.tolerance, check.passed)
                       for check in sample.checks()]
        return Produced(samples_npz(samples), tuple(checks))


def accuracy(predicted: Sequence[np.ndarray], reference: Sequence[Sample], *, over: str) -> tuple[Accuracy, ...]:
    """A surrogate's accuracy per use against the boundary-element samples ``reference`` (its predictions
    ``predicted`` in the same frame, about the same point): ``resistance``, the largest difference in each 3 x 3 block
    relative to that block's largest magnitude over the set; and ``swim_speed``, the largest relative difference in the
    force-free speed along the tube's axis per unit spin. Each estimates the error against this method at this
    discretisation over ``over``; neither bounds the error against the flow."""
    if len(predicted) != len(reference) or not reference:
        raise ValueError("one prediction per reference sample, and at least one")
    found = np.array(predicted, dtype=np.float64)
    held = np.array([sample.resistance for sample in reference])
    worst = 0.0
    for rows in (slice(0, 3), slice(3, 6)):
        for columns in (slice(0, 3), slice(3, 6)):
            scale = float(np.max(np.abs(held[:, rows, columns])))
            worst = max(worst, float(np.max(np.abs(found[:, rows, columns] - held[:, rows, columns]))) / scale)
    speeds = np.array([swim_per_spin(each)[2] for each in found])
    expected = np.array([sample.swim()[2] for sample in reference])
    speed = float(np.max(np.abs(speeds - expected) / np.abs(expected)))
    against = "the boundary-element method through the tube wall kind, at the generator's discretisation"
    estimates = ("estimates the surrogate's error against that method over the held-out set; bounds nothing about the "
                 "flow, since the method's own error is in neither side (MIME-ANO-116, the near-wall edge)")
    return (Accuracy("resistance", "largest blockwise difference relative to the block's largest magnitude", worst,
                     against, over, estimates),
            Accuracy("swim_speed", "largest relative difference in the axial force-free speed per unit spin", speed,
                     against, over, estimates))


#: The near-wall check (MIME-ANO-114's first step), declared before it ran: FL-9 resting in the 1/4 in tube (script A's
#: body and tube) on its closed surface, at each of ``NEAR_WALL_SPACINGS`` and each of ``NEAR_WALL_GAPS``; the far gap,
#: the largest, stands for the body's own discretisation error, which the wall does not add to.
NEAR_WALL_DESIGN, NEAR_WALL_TUBE = "FL-9", "1/4 in"
NEAR_WALL_SPACINGS = (0.30e-3, 0.24e-3, 0.19e-3, 0.15e-3, 0.12e-3, 0.10e-3)
NEAR_WALL_GAPS = (0.02e-3, 0.03e-3, 0.05e-3, 0.075e-3, 0.10e-3, 0.15e-3, 0.20e-3, 0.30e-3, 0.50e-3)
#: How much error the wall may add, beyond the body's own at the far gap, before a gap counts as unresolved.
NEAR_WALL_TOLERANCE = 0.01
#: The quantities the check judges, each a function of the canonical resistance: the axial swim speed per unit spin,
#: the six diagonal entries, and the axial coupling (the force along the axis per unit spin about it).
NEAR_WALL_QUANTITIES: Mapping[str, Callable[[np.ndarray], float]] = {
    "swim_speed": lambda r: float(swim_per_spin(r)[2]),
    **{f"diagonal_{index}": (lambda r, i=index: float(r[i, i])) for index in range(6)},
    "axial_coupling": lambda r: float(r[2, 5]),
}
#: The quantities each use is judged on: the whole resistance on all of them (the criterion declared before the check
#: ran), the swim speed on itself.
NEAR_WALL_USES: Mapping[str, tuple[str, ...]] = {"resistance": tuple(NEAR_WALL_QUANTITIES),
                                                 "swim_speed": ("swim_speed",)}


def wall_excess(record: Mapping[str, Any], quantities: Sequence[str] = tuple(NEAR_WALL_QUANTITIES)
                ) -> dict[str, dict[str, float]]:
    """``E(g, h)``, by spacing and gap (keys in metres, as ``repr``), for every spacing but the finest: for each
    quantity ``q``, ``|q(g, h) / q(g, h_f) - q(g_far, h) / q(g_far, h_f)|``, the largest over the quantities: the
    relative error at the gap ``g`` beyond the one at the far gap, the finest spacing ``h_f`` the reference, over
    ``quantities`` (names of ``NEAR_WALL_QUANTITIES``). ``record["samples"]`` holds each sample's spacing, gap and
    canonical resistance."""
    resistance = {(float(each["spacing"]), float(each["gap"])): np.asarray(each["resistance"], dtype=np.float64)
                  for each in record["samples"]}
    spacings = sorted({spacing for spacing, _ in resistance}, reverse=True)
    gaps = sorted({gap for _, gap in resistance})
    finest, far = spacings[-1], gaps[-1]
    found: dict[str, dict[str, float]] = {}
    for spacing in spacings[:-1]:
        row: dict[str, float] = {}
        for gap in gaps:
            row[repr(gap)] = max(
                abs(quantity(resistance[(spacing, gap)]) / quantity(resistance[(finest, gap)])
                    - quantity(resistance[(spacing, far)]) / quantity(resistance[(finest, far)]))
                for quantity in (NEAR_WALL_QUANTITIES[name] for name in quantities))
        found[repr(spacing)] = row
    return found


def near_wall_ratio(record: Mapping[str, Any], tolerance: float = NEAR_WALL_TOLERANCE,
                    quantities: Sequence[str] = tuple(NEAR_WALL_QUANTITIES)) -> float:
    """The near-wall edge as a multiple of the spacing: the smallest ratio ``r`` among the check's gap-to-spacing
    ratios such that, at every spacing but the finest, every gap at least ``r`` spacings wide has ``E <= tolerance``
    (``wall_excess`` over ``quantities``). Infinite where no ratio does."""
    excess = wall_excess(record, quantities)
    pairs = [(float(gap) / float(spacing), value) for spacing, row in excess.items() for gap, value in row.items()]
    for ratio in sorted({ratio for ratio, _ in pairs}):
        if all(value <= tolerance for each, value in pairs if each >= ratio * (1.0 - 1e-12)):
            return ratio
    return math.inf


def near_wall_ratios(record: Mapping[str, Any]) -> dict[str, float]:
    """``near_wall_ratio`` for each of ``NEAR_WALL_USES``."""
    return {use: near_wall_ratio(record, quantities=quantities) for use, quantities in NEAR_WALL_USES.items()}


def protocol_json(domain: Domain, discretisation: Discretisation) -> str:
    """The protocol at ``discretisation`` as JSON: the domain, the training margins and each set's size."""
    held_out = held_out_cases(domain, discretisation.spacing)
    return json.dumps({"domain": domain.as_json(), "discretisation": discretisation.as_json(),
                       "edges": {use: domain.edge(discretisation.spacing, use) for use in domain.near_wall_ratios},
                       "training_margins": list(training_margins(domain, discretisation.spacing)),
                       "sizes": {"training": len(training_cases(domain, discretisation.spacing)),
                                 **{name: len(cases) for name, cases in held_out.items()}}}, indent=1)


__all__ = ["ASYMMETRY_TOLERANCE", "CANONICAL", "DESIGNS", "DOMAIN", "EIGENVALUE_FLOOR", "MARGIN", "NEAR_WALL_DESIGN",
           "NEAR_WALL_GAPS", "NEAR_WALL_QUANTITIES", "NEAR_WALL_RATIOS", "NEAR_WALL_SPACINGS", "NEAR_WALL_TOLERANCE",
           "NEAR_WALL_TUBE", "NEAR_WALL_USES", "ROLLS", "TRAINING_MARGINS", "TUBES", "BEMEvaluator", "Case", "Design",
           "Discretisation", "Domain", "Sample", "TrainingData", "accuracy", "design_named", "evaluate", "experiment",
           "held_out_cases", "into_canonical", "near_wall_ratio", "near_wall_ratios", "overlap", "protocol_json",
           "samples_npz", "swim_per_spin", "training_cases", "training_margins", "wall_excess"]
