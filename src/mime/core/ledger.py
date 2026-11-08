"""The energy ledger's machinery in a receiver's step: the one pairing rule, each law's paired work recorded beside the
step, the receiver's step identity, and running sums over its steps and sub-steps.

**Off unless attached, decided at trace time.** A receiver is metered only when it is built with ``ledger`` (the
structural constant :data:`LEDGER`, a tuple of encoded :class:`LedgerDelivery`, one per delivery it takes). Without it
the receiver's step is the step it always was, the same operations in the same order, and its inputs and state are
the ones it always had, so a program without the ledger keeps every bit.

**Per-law inputs.** With the ledger attached, each law delivers into an input of its own on its receiver
(:func:`law_input`), and the receiver keeps its usual input for anything unattributed. The receiver sums them into
what its step reads (:func:`summed`), and pairs each apart: attribution is then the check that the unattributed
inputs paired no work, exactly zero, at any size and in either dtype.

**The pairing** (:func:`paired_work`) is the velocity a delivery's force is evaluated at: the configuration's actual
change over the step for a load read at the step's start; the step's own solution for a force the step solves for; the
returned effort for a flow handed in; ``-(1/2) v^T dM v`` for a delivered inertia. A law hosted in its receiver's step
(a contact) is paired with the step's solution, under its own name. It is keyed by ``mime.vocabulary.DeliveryKind``'s
values, as strings, since ``mime.core`` imports nothing above it.

**The step identity.** A metered receiver writes its step's account (:class:`StepAccount`): each law's paired work,
each hosted law's loss and its share of the host's scheme, the unattributed inputs' work, the receiver's own scheme's
numerical terms (``MimeNode.LEDGER_TERMS``) and its storage at the step's start and end, the states the step takes and
returns. :func:`recorded` keeps running sums of them in the receiver's state, with the identity's residual, ``dH - W +
N``, judged each step against :data:`IDENTITY_ULPS` units of rounding on the step's scale. A potential law's energy
is not judged per step: its class is settled on the law alone, and its energy's change over a run is taken from the
run's ends. A dissipative law's class is read from its own row of the paired work: a step in which it was positive
beyond the same floor is counted (``gift_steps``), which the identity cannot see, since it books the gift under the
law's name.

**The books are read, never stepped on.** No state a step reads depends on a record: the magnitudes the scale is
formed from and the worst steps kept are diagnostics, so their non-smooth operations select no branch of the physics,
and a derivative through the run (of a law's paired work, say) passes through the paired work's own sums."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from jax import Array

from mime.core._node import as_run_array, jnp

if TYPE_CHECKING:
    from maddening.core.node import BoundaryInputSpec

    from mime.core._node import MimeNode

#: ``mime.vocabulary.DeliveryKind``'s values, and the receiver's own kind for a law hosted in its step.
LOAD, SOLVED, FLOW_IN, INERTIA, HOSTED = "load", "solved", "flow_in", "inertia", "hosted"
DELIVERY_KINDS = (LOAD, SOLVED, FLOW_IN, INERTIA)
#: ``mime.vocabulary.EnergyClass``'s values, and the class of a law that declares none (its reason stated by its
#: entry): its work on the receiver is recorded and attributed to it all the same.
DISSIPATIVE, POTENTIAL, WORKLESS, UNMETERED = "dissipative", "potential", "workless", "unmetered"
ENERGY_CLASSES = (DISSIPATIVE, POTENTIAL, WORKLESS)
#: ``mime.vocabulary.Definiteness.STATE``'s value: a law whose operator's definiteness is checked at run time.
CHECK_DEFINITE = "state"
#: The constructor argument that attaches the ledger, and the prefix of the fields it adds to a receiver's state.
LEDGER, RECORDS = "ledger", "ledger_"
#: The step identity's rounding floor, in units of the run's eps on the step's scale (the magnitude of the arithmetic
#: its terms are formed by, its solve's included, each product's terms taken in magnitude): measured at 1.5 to 2.7 eps
#: of the terms' own sum on script A's body; and a dissipative operator's, on its symmetric part scaled by its
#: diagonal, the floor it is shifted by before its factorisation (semidefinite at that floor passes).
IDENTITY_ULPS, DEFINITE_ULPS = 16.0, 16.0
#: What separates an encoded delivery's fields: no law id, input name or producer holds it.
_SEPARATOR = "|"
#: What joins a law's own input to the input it stands beside and its delivery's index.
_OWN = "__ledger"


@dataclass(frozen=True)
class LedgerDelivery:
    """One delivery a metered receiver takes: into its input ``field`` (``"confinement"`` for a hosted law), from
    ``law`` (the law's id, ``gravity:umr``), of the law's ``energy_class`` (:data:`UNMETERED` for a law that declares
    none), paired as ``kind`` (a :data:`DELIVERY_KINDS` value, or :data:`HOSTED`), with ``check`` set to
    :data:`CHECK_DEFINITE` where the law's operator is checked definite at run time, and ``supplied_by`` naming the
    producers outside the components whose work the law supplies (comma-joined), as the build works it out."""

    field: str
    law: str
    energy_class: str
    kind: str
    check: str = ""
    supplied_by: str = ""

    def __post_init__(self) -> None:
        if self.kind not in (*DELIVERY_KINDS, HOSTED):
            raise ValueError(f"a delivery's kind is one of {[*DELIVERY_KINDS, HOSTED]}, not {self.kind!r}")
        if self.energy_class not in (*ENERGY_CLASSES, UNMETERED):
            raise ValueError(f"a law's class is one of {[*ENERGY_CLASSES, UNMETERED]}, not {self.energy_class!r}")
        if self.check not in ("", CHECK_DEFINITE):
            raise ValueError(f"a delivery's run-time check is {CHECK_DEFINITE!r} or none, not {self.check!r}")
        if any(_SEPARATOR in getattr(self, f.name) for f in fields(self)):
            raise ValueError(f"no field of a delivery holds {_SEPARATOR!r}: {self}")

    def encoded(self) -> str:
        """The delivery as one string, a structural constant of its receiver."""
        return _SEPARATOR.join(getattr(self, f.name) for f in fields(self))

    @classmethod
    def decoded(cls, text: str) -> LedgerDelivery:
        return cls(*text.split(_SEPARATOR))


def deliveries(constants: Mapping[str, Any]) -> tuple[LedgerDelivery, ...]:
    """The deliveries a receiver's constants attach: none where the ledger is not attached."""
    return tuple(LedgerDelivery.decoded(text) for text in constants.get(LEDGER, ()))


def laws(held: Sequence[LedgerDelivery]) -> tuple[str, ...]:
    """Each law once, in the order of its first delivery: the rows of the receiver's per-law records."""
    return tuple(dict.fromkeys(d.law for d in held))


def law_input(field: str, index: int) -> str:
    """The input of the ``index``-th delivery's own on its receiver, beside ``field``, which takes anything
    unattributed."""
    return f"{field}{_OWN}{index}"


def own_input(name: str) -> tuple[str, int] | None:
    """The input a law's own input ``name`` stands beside and the index of its delivery (:func:`law_input`'s
    inverse); ``None`` for any other input."""
    field, found, index = name.rpartition(_OWN)
    return (field, int(index)) if found and field and index.isdigit() else None


def with_law_inputs(spec: Mapping[str, BoundaryInputSpec],
                    held: Sequence[LedgerDelivery]) -> dict[str, BoundaryInputSpec]:
    """A receiver's inputs with each delivery's own beside the input it would otherwise sum into: ``spec`` itself
    where no ledger is attached. A hosted law has no input: it is held in the step."""
    found = dict(spec)
    for index, delivery in enumerate(held):
        if delivery.kind != HOSTED:
            if delivery.field not in spec:
                raise ValueError(f"the ledger delivers {delivery.law} into {delivery.field!r}, which the receiver "
                                 f"does not take: it takes {sorted(spec)}")
            found[law_input(delivery.field, index)] = spec[delivery.field]
    return found


def summed(spec: Mapping[str, BoundaryInputSpec], held: Sequence[LedgerDelivery],
           boundary_inputs: Mapping[str, Array]) -> dict[str, Array]:
    """What a metered receiver's step reads: each of its own inputs (``spec``, its inputs without the ledger), the
    unattributed value or its default, with every law's own input into it added in the order the deliveries are held."""
    given = {name: boundary_inputs[name] if name in boundary_inputs else jnp.asarray(entry.default)
             for name, entry in spec.items()}
    for index, delivery in enumerate(held):
        if delivery.kind != HOSTED:
            given[delivery.field] = given[delivery.field] + boundary_inputs[law_input(delivery.field, index)]
    return given


def paired_work(kind: str, effort: Array, *, change: Array | None = None, solution: Array | None = None,
                dt: float | None = None, velocity: Array | None = None, returned: Array | None = None) -> Array:
    """A delivery's paired work over one step of its receiver, by the one rule:

    - :data:`LOAD` (a load read at the step's start): ``effort . change``, the configuration's actual change;
    - :data:`SOLVED` and :data:`HOSTED` (a force the step solves for, ``effort`` evaluated at ``solution``):
      ``dt effort . solution``;
    - :data:`FLOW_IN`: ``dt returned . velocity``, the effort returned for the flow handed in;
    - :data:`INERTIA` (``effort`` the inertia's change ``dM``): ``-(1/2) velocity^T dM velocity``."""
    if kind == LOAD and change is not None:
        return jnp.dot(effort, change)
    if kind in (SOLVED, HOSTED) and solution is not None and dt is not None:
        return dt * jnp.dot(effort, solution)
    if kind == FLOW_IN and returned is not None and velocity is not None and dt is not None:
        return dt * jnp.dot(returned, velocity)
    if kind == INERTIA and velocity is not None:
        return -0.5 * velocity @ effort @ velocity
    raise ValueError(f"no pairing for a delivery of kind {kind!r} with what was given")


def definite_at_floor(matrix: Array, ulps: float = DEFINITE_ULPS) -> Array:
    """Whether ``matrix``'s symmetric part is positive semidefinite to ``ulps`` units of rounding: one Cholesky
    factorisation of that part, scaled by its diagonal (a congruence, which keeps the signs of its eigenvalues, so a
    force's and a torque's rows compare), shifted up by the floor. A semidefinite matrix passes, where a plain
    factorisation fails on it; a negative diagonal entry fails. **A zero diagonal entry passes only with its whole row
    zero**, which is what a positive semidefinite matrix has (a nonzero entry beside a zero diagonal makes a ``2 x 2``
    minor negative), judged exactly: no scale stands in for the zero entry's own, so a coupling of any size in SI units
    is seen, where scaling that row by 1 beside rows scaled to unit diagonal put it below the shift. Traced: a boolean
    array."""
    symmetric = 0.5 * (matrix + matrix.T)
    diagonal = jnp.diagonal(symmetric)
    zero = diagonal == 0.0
    scale = jnp.sqrt(jnp.where(zero, 1.0, jnp.abs(diagonal)))
    scaled = symmetric / (scale[:, None] * scale[None, :])
    floor = ulps * jnp.finfo(matrix.dtype).eps
    factor = jnp.linalg.cholesky(scaled + floor * jnp.eye(matrix.shape[0], dtype=matrix.dtype))
    rows_zero = jnp.all(symmetric == 0.0, axis=1)
    return jnp.all(jnp.isfinite(factor)) & jnp.all(rows_zero | ~zero)


@dataclass(frozen=True)
class StepAccount:
    """One step of a metered receiver, as its identity's terms, each in joules over the step: ``work`` per law (its
    paired work, a hosted law's included), ``hosted`` per law (a hosted law's loss and its share of its host's scheme,
    zero for any other law), ``unattributed`` (the work its unattributed inputs paired), ``terms`` (its own scheme's
    numerical terms, in ``MimeNode.LEDGER_TERMS``' order), its storage at the step's start and at its end, ``scale``
    (the step's scale, what its rounding is judged on: the magnitude of the arithmetic that forms its terms, each
    product's factors taken in magnitude, the step's solve included, so that terms that cancel, and a stiff solve,
    are judged on the size of what was rounded) and, by row, each operator whose definiteness is checked at run
    time."""

    work: Array
    hosted: Array
    unattributed: Array
    terms: Array
    storage_start: Array
    storage_end: Array
    scale: Array
    operators: Mapping[int, Array]

    def residual(self) -> Array:
        """``dH - W + N``: the change in storage less every paired work, the scheme's terms added back."""
        supplied = jnp.sum(self.work) + self.unattributed
        numerical = jnp.sum(self.terms) + jnp.sum(self.hosted)
        return (self.storage_end - self.storage_start) - supplied + numerical


def initial_records(held: Sequence[LedgerDelivery], terms: int, storage: Any,
                    configuration: Any = None) -> dict[str, Array]:
    """The ledger's fields in a metered receiver's state at the run's start: every running sum at zero, the storage at
    the initial state, and the configuration there where the receiver states one (``MimeNode.configuration``): what
    a potential law's energy is evaluated at, at the run's two ends."""
    rows = len(laws(held))

    def zero(*shape: int) -> Array:
        return as_run_array(jnp.zeros(shape))

    start = {} if configuration is None else {f"{RECORDS}configuration_start": as_run_array(configuration)}
    return {**start,
        f"{RECORDS}work": zero(rows), f"{RECORDS}step_work": zero(rows), f"{RECORDS}hosted": zero(rows, 2),
        f"{RECORDS}terms": zero(terms), f"{RECORDS}step_terms": zero(terms),
        f"{RECORDS}unattributed": zero(), f"{RECORDS}unattributed_worst": zero(),
        f"{RECORDS}storage_start": as_run_array(storage), f"{RECORDS}storage": as_run_array(storage),
        f"{RECORDS}step_identity": zero(), f"{RECORDS}identity_worst": zero(), f"{RECORDS}scale": zero(),
        f"{RECORDS}indefinite_steps": zero(rows), f"{RECORDS}gift_steps": zero(rows),
    }


def unrecorded(state: Mapping[str, Array]) -> dict[str, Array]:
    """A metered receiver's state without the ledger's fields: what its step takes."""
    return {key: value for key, value in state.items() if not key.startswith(RECORDS)}


def recorded(state: Mapping[str, Array], step: StepAccount, held: Sequence[LedgerDelivery] = ()) -> dict[str, Array]:
    """The ledger's fields after ``step``: running sums, each check's worst over the run, and the step's own values
    beside them (what a recorded history shows per step). The identity's residual is kept relative to the step's scale
    (``step_identity``; over :data:`IDENTITY_ULPS` times eps it has failed), the unattributed work as its largest
    magnitude (above zero it has failed), and per law the count of steps whose operator failed its definiteness
    (``indefinite_steps``) and, per **dissipative** law of ``held``, the count of steps in which its paired work was
    positive beyond :data:`IDENTITY_ULPS` units of eps on the step's scale, or, for a law held in the step, its rule's
    removal at arrival negative beyond it (``gift_steps``: a dissipative law gives no energy, whatever it is
    delivered, neither through the step's solve nor by its rule at arrival; this is its class's sign, read from the
    law's own row, and it is silent on the identity, which books the gift under the law's name)."""
    dtype = step.work.dtype
    relative = jnp.abs(step.residual()) / jnp.maximum(step.scale, jnp.finfo(dtype).tiny)
    indefinite = jnp.zeros_like(state[f"{RECORDS}indefinite_steps"])
    for row, operator in step.operators.items():
        indefinite = indefinite.at[row].set((~definite_at_floor(operator)).astype(dtype))
    gifts = jnp.zeros_like(state[f"{RECORDS}gift_steps"])
    rows = laws(held)
    floor = IDENTITY_ULPS * jnp.finfo(dtype).eps * step.scale
    for row in sorted({rows.index(d.law) for d in held if d.energy_class == DISSIPATIVE}):
        gave = (step.work[row] > floor) | (step.hosted[row, 0] < -floor)
        gifts = gifts.at[row].set(gave.astype(dtype))
    r = RECORDS
    start = {key: state[key] for key in (f"{r}configuration_start",) if key in state}
    return {
        **start,
        f"{r}work": state[f"{r}work"] + step.work, f"{r}step_work": step.work,
        f"{r}hosted": state[f"{r}hosted"] + step.hosted,
        f"{r}terms": state[f"{r}terms"] + step.terms, f"{r}step_terms": step.terms,
        f"{r}unattributed": state[f"{r}unattributed"] + step.unattributed,
        f"{r}unattributed_worst": jnp.maximum(state[f"{r}unattributed_worst"], jnp.abs(step.unattributed)),
        f"{r}storage_start": state[f"{r}storage_start"], f"{r}storage": step.storage_end,
        f"{r}step_identity": relative, f"{r}identity_worst": jnp.maximum(state[f"{r}identity_worst"], relative),
        f"{r}scale": state[f"{r}scale"] + step.scale,
        f"{r}indefinite_steps": state[f"{r}indefinite_steps"] + indefinite,
        f"{r}gift_steps": state[f"{r}gift_steps"] + gifts,
    }


def balance(records: Mapping[str, Any]) -> Any:
    """A metered receiver's books over its run, from its records: the change in its storage less every paired work,
    its scheme's and its hosted laws' terms added back. The sum of its steps' residuals."""
    r = RECORDS
    supplied = jnp.sum(records[f"{r}work"]) + records[f"{r}unattributed"]
    numerical = jnp.sum(records[f"{r}terms"]) + jnp.sum(records[f"{r}hosted"])
    return (records[f"{r}storage"] - records[f"{r}storage_start"]) - supplied + numerical


def losses(records: Mapping[str, Any], held: Sequence[LedgerDelivery]) -> Any:
    """What each law of ``held`` took from its receiver's motion over the run (J, a row per law, as its records keep
    them): a dissipative law's paired work turned to a loss, and, for one held in the step, the velocity rule's
    removal at arrival added (its projection's term is its host's scheme's, shown apart). Zero for any other class.
    Traced, as the records are."""
    rows = laws(held)
    dissipative = jnp.asarray([any(d.law == law and d.energy_class == DISSIPATIVE for d in held) for law in rows])
    taken = -records[f"{RECORDS}work"] + records[f"{RECORDS}hosted"][:, 0]
    return jnp.where(dissipative, taken, jnp.zeros_like(taken))


def stored(records: Mapping[str, Any]) -> Any:
    """The change in what a metered receiver stores over the run (J). Traced, as the records are."""
    return records[f"{RECORDS}storage"] - records[f"{RECORDS}storage_start"]


def balance_floor(records: Mapping[str, Any]) -> Any:
    """The rounding the balance is judged against: :data:`IDENTITY_ULPS` units of eps on the run's summed step scale."""
    scale = records[f"{RECORDS}scale"]
    return IDENTITY_ULPS * jnp.finfo(jnp.asarray(scale).dtype).eps * scale


def with_records(node: MimeNode, state: Mapping[str, Array]) -> dict[str, Array]:
    """``state``, a receiver's initial state, with the ledger's fields where the ledger is attached to ``node``."""
    held = deliveries(node.constants())
    if not held:
        return dict(state)
    terms = node.LEDGER_TERMS or ()
    p = node.constants()
    return {**state, **initial_records(held, len(terms), node.storage(p, state), node.configuration(p, state))}


Step = Callable[[Mapping[str, Any], Mapping[str, Array], Mapping[str, Array], float], tuple[dict[str, Array], Any]]


def metered_step(node: MimeNode, spec: Mapping[str, BoundaryInputSpec], p: Mapping[str, Any],
                 state: Mapping[str, Array], boundary_inputs: Mapping[str, Array], dt: float,
                 step: Step) -> dict[str, Array]:
    """One step of a metered receiver: its own inputs summed from the per-law ones (``spec``, its inputs without the
    ledger), its ``step`` taken on its state without the ledger's fields (returning the new state and what its books
    read of the step), its account formed by its ``identity_terms`` and recorded."""
    held = deliveries(node.constants())
    own = unrecorded(state)
    ended, parts = step(p, own, summed(spec, held, boundary_inputs), dt)
    account = node.identity_terms(p, held, own, boundary_inputs, ended, parts, dt)
    return {**ended, **recorded(state, account, held)}


#: The fields a route-1 receiver adds to its records (:func:`route_one_records`).
ROUTE_ONE_FIELDS = ("route_one_step", "route_one_worst", "route_one_negative_steps")


def route_one_records(state: Mapping[str, Array]) -> dict[str, Array]:
    """``state``'s ledger fields with a route-1 receiver's own: each step's dissipation over its scale, its smallest over
    the run, and the count of steps that gave energy (all zero at the run's start)."""
    zero = as_run_array(jnp.zeros(()))
    return {**state, **{f"{RECORDS}{name}": zero for name in ROUTE_ONE_FIELDS}}


def route_one_recorded(state: Mapping[str, Array], step: StepAccount,
                       held: Sequence[LedgerDelivery] = ()) -> dict[str, Array]:
    """The records after ``step`` of a receiver whose scheme claims no step identity (route 1, F0's SF-9: storage at
    least; S1's RK3, whose split penalty has no formula term). The running sums are :func:`recorded`'s; its identity
    is not judged, so its residual, the scheme's numerical dissipation that no formula books, is left in the balance
    (``dH - W + N`` summed, as for any receiver) and the identity's fields read zero. **Route 1's check is that every
    step dissipated**: the work every law that is not dissipative supplied, the unattributed included, less the change
    in storage, ``D = W - dH``, is at least ``-IDENTITY_ULPS`` units of eps on the step's scale. A step that gave energy
    (an energy gift larger than the step's own dissipation, by the identity ``D = D_true - G``) is counted
    (``route_one_negative_steps``); ``route_one_step`` is the step's ``D`` over its scale, ``route_one_worst`` its
    smallest over the run."""
    out = recorded(state, step, held)
    dtype = step.work.dtype
    rows = laws(held)
    dissipative = jnp.asarray([any(d.law == law and d.energy_class == DISSIPATIVE for d in held) for law in rows],
                              dtype=bool)
    supplied = jnp.sum(jnp.where(dissipative, 0.0, step.work)) + step.unattributed
    dissipated = supplied - (step.storage_end - step.storage_start)
    relative = dissipated / jnp.maximum(step.scale, jnp.finfo(dtype).tiny)
    floor = IDENTITY_ULPS * jnp.finfo(dtype).eps
    r = RECORDS
    zero = jnp.zeros((), dtype=dtype)
    first = jnp.asarray(state[f"{r}scale"]) == 0.0
    worst = jnp.where(first, relative, jnp.minimum(state[f"{r}route_one_worst"], relative))
    return {**out, f"{r}step_identity": zero, f"{r}identity_worst": zero,
            f"{r}route_one_step": relative, f"{r}route_one_worst": worst,
            f"{r}route_one_negative_steps": state[f"{r}route_one_negative_steps"] + (relative < -floor).astype(dtype)}


def metered_route_one_step(node: MimeNode, spec: Mapping[str, BoundaryInputSpec], p: Mapping[str, Any],
                           state: Mapping[str, Array], boundary_inputs: Mapping[str, Array], dt: float,
                           step: Step) -> dict[str, Array]:
    """:func:`metered_step` for a route-1 receiver: its account recorded by :func:`route_one_recorded`."""
    held = deliveries(node.constants())
    own = unrecorded(state)
    ended, parts = step(p, own, summed(spec, held, boundary_inputs), dt)
    account = node.identity_terms(p, held, own, boundary_inputs, ended, parts, dt)
    return {**ended, **route_one_recorded(state, account, held)}


__all__ = ["CHECK_DEFINITE", "DEFINITE_ULPS", "DELIVERY_KINDS", "DISSIPATIVE", "ENERGY_CLASSES", "FLOW_IN", "HOSTED",
           "IDENTITY_ULPS", "INERTIA", "LEDGER", "LOAD", "POTENTIAL", "RECORDS", "SOLVED", "UNMETERED", "WORKLESS",
           "ROUTE_ONE_FIELDS", "LedgerDelivery", "StepAccount", "balance", "balance_floor", "definite_at_floor", "deliveries",
           "initial_records", "law_input", "laws", "losses", "metered_route_one_step", "metered_step", "own_input", "paired_work", "recorded", "route_one_recorded", "route_one_records", "stored", "summed", "unrecorded",
           "with_law_inputs", "with_records"]
