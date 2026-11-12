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

**Between steps.** The identity's change in storage is each step's own, from the storage at its start to the storage at
its end. Where a receiver's storage reads something the step does not return (a delivered inertia, predicted a step
ahead, which the next step reads afresh; a coupling group's accepted iterate, which its records are one pass after),
the storage at a step's start differs from the one the step before ended at: that jump is recorded apart
(``between``), and the account names whose it is. **A jump is added back to the books only where a producer is
declared for it**: the delivered inertia's exchange (the part of the storage that reads a delivery, ``StepAccount.
own_start`` being the storage without it) and a coupling group (its members' jumps, which the account knows by the
group). **Whatever no declared producer explains is judged exactly as unattributed work is**: the storage of what the
receiver's own state holds at a step's start less the one the step before ended at is rounding where nothing touched
the state between steps, kept as its worst over the step's scale (``between_worst``, over :data:`IDENTITY_ULPS` units
of eps a failure of the step identity) and summed (``between_own``), which the balance leaves in. A state edited
between steps, a restart that resets a field, a node that writes another's state, is seen there.

**The books are read, never stepped on.** No state a step reads depends on a record: the magnitudes the scale is
formed from and the worst steps kept are diagnostics, so their non-smooth operations select no branch of the physics,
and a derivative through the run (of a law's paired work, say) passes through the paired work's own sums."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from jax import Array
from jaxtyping import Float
from typing_extensions import Annotated, NotRequired, ReadOnly, TypedDict

from mime.core._error_free import add_to_pair
from mime.core._field_kinds import Units
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
#: A delivery whose sign is judged in its cluster's sum, not on its receiver's row: a medium's near field's lagged cross
#: loads, which have no sign alone (one body drags another), judged by the joint node at one velocity, the grand
#: resistance's quadratic form at the step's starting twists (its ``cluster_gift_steps``); metered all the same.
CHECK_CLUSTER = "cluster"
#: The constructor argument that attaches the ledger, and the prefix of the fields it adds to a receiver's state.
LEDGER, RECORDS = "ledger", "ledger_"
#: The step identity's rounding floor, in units of the run's eps on the step's scale (the magnitude of the arithmetic
#: its terms are formed by, its solve's included, each product's terms taken in magnitude): measured at 1.5 to 2.7 eps
#: of the terms' own sum on script A's body; and a dissipative operator's, on its symmetric part scaled by its
#: diagonal, the floor it is shifted by before its factorisation (semidefinite at that floor passes).
IDENTITY_ULPS, DEFINITE_ULPS = 16.0, 16.0
#: The running sums kept as a pair (``MIME-ANO-149``): the sums of a signed series, each field with the part below its
#: last place beside it (``<name>_low``). The step scale is a sum of magnitudes and is kept plain.
PAIRED = ("work", "hosted", "terms", "unattributed", "between", "between_own")
#: The suffix of a pair's second field.
LOW = "_low"
#: The columns of a storing law's own books (``StepAccount.stored``): its storage at the step's start and end, its loss
#: and its own numerical term, and the scale its own identity is judged on (J): its terms' magnitudes, and whatever
#: rounding the receiver holding it carries into what the law receives (a flow's face flux, summed over its faces).
LAW_BOOKS = ("storage_start", "storage_end", "loss", "numerical", "scale")
#: The records a receiver adds where a law it holds stores energy of its own (:func:`law_records`).
LAW_RECORDS = ("law_storage_start", "law_storage", "law_loss", f"law_loss{LOW}", "law_numerical",
               f"law_numerical{LOW}", "law_identity_worst")
#: What separates an encoded delivery's fields: no law id, input name or producer holds it.
_SEPARATOR = "|"
#: What joins a law's own input to the input it stands beside and its delivery's index.
_OWN = "__ledger"


class LedgerRecords(TypedDict):
    """The energy ledger's records in a metered receiver's state (:func:`initial_records`), with their units: each
    law's paired work over the run and over the step, each hosted law's two terms and the scheme's own terms, the
    unattributed work, the storage at the run's start and now, what the storage jumped between steps and the part of
    that jump no declared producer explains (``between_own``, of the storage of the receiver's own state,
    ``storage_own`` at the last step's end), the run's summed step scale (J); **each running sum of a signed series is a
    pair** (:data:`PAIRED`: its field, and the part of the sum below that field's last place beside it as
    ``<name>_low``, in the same units; the field alone is the sum rounded to the run's dtype); the step identity's
    residual over the step's scale and its worst, that unexplained jump over the step's scale at its worst
    (``between_worst``), and the counts of indefinite and gift steps per law (1). A receiver whose state a potential
    law's energy reads adds the configuration at the run's start (``ledger_configuration_start``) in its own units,
    beside its own fields, in its typed state."""

    ledger_work: ReadOnly[Annotated[Float[Array, "laws"], Units("J")]]
    ledger_work_low: ReadOnly[Annotated[Float[Array, "laws"], Units("J")]]
    ledger_step_work: ReadOnly[Annotated[Float[Array, "laws"], Units("J")]]
    ledger_hosted: ReadOnly[Annotated[Float[Array, "laws 2"], Units("J")]]  # noqa: F722 - a jaxtyping shape
    ledger_hosted_low: ReadOnly[Annotated[Float[Array, "laws 2"], Units("J")]]  # noqa: F722 - a jaxtyping shape
    ledger_terms: ReadOnly[Annotated[Float[Array, "terms"], Units("J")]]
    ledger_terms_low: ReadOnly[Annotated[Float[Array, "terms"], Units("J")]]
    ledger_step_terms: ReadOnly[Annotated[Float[Array, "terms"], Units("J")]]
    ledger_unattributed: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    ledger_unattributed_low: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    ledger_unattributed_worst: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    ledger_between: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    ledger_between_low: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    ledger_between_own: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    ledger_between_own_low: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    ledger_between_worst: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    ledger_storage_own: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    ledger_storage_start: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    ledger_storage: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    ledger_step_identity: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    ledger_identity_worst: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    ledger_scale: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    ledger_indefinite_steps: ReadOnly[Annotated[Float[Array, "laws"], Units("1")]]
    ledger_gift_steps: ReadOnly[Annotated[Float[Array, "laws"], Units("1")]]


class LawRecords(TypedDict):
    """The records a receiver adds where a law it holds stores energy of its own (:func:`law_records`), held only then:
    each law's storage at the run's start and now, its loss and numerical term summed (pairs), and its own identity's
    worst residual over its own scale."""

    ledger_law_storage_start: NotRequired[ReadOnly[Annotated[Float[Array, "laws"], Units("J")]]]
    ledger_law_storage: NotRequired[ReadOnly[Annotated[Float[Array, "laws"], Units("J")]]]
    ledger_law_loss: NotRequired[ReadOnly[Annotated[Float[Array, "laws"], Units("J")]]]
    ledger_law_loss_low: NotRequired[ReadOnly[Annotated[Float[Array, "laws"], Units("J")]]]
    ledger_law_numerical: NotRequired[ReadOnly[Annotated[Float[Array, "laws"], Units("J")]]]
    ledger_law_numerical_low: NotRequired[ReadOnly[Annotated[Float[Array, "laws"], Units("J")]]]
    ledger_law_identity_worst: NotRequired[ReadOnly[Annotated[Float[Array, ""], Units("1")]]]


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
        if self.check not in ("", CHECK_DEFINITE, CHECK_CLUSTER):
            raise ValueError(f"a delivery's run-time check is {CHECK_DEFINITE!r}, {CHECK_CLUSTER!r} or none, not "
                             f"{self.check!r}")
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
    """What a metered receiver's step reads: each of its own inputs (``spec``, its inputs without the ledger) that an
    edge feeds or a law delivers into, the unattributed value or its default, with every law's own input into it added
    in the order the deliveries are held. An input nothing feeds is left out, as the graph leaves it out of a
    receiver without the ledger, so that a step that reads whether an input is fed reads it alike."""
    delivered = {d.field for d in held if d.kind != HOSTED}
    given = {name: boundary_inputs[name] if name in boundary_inputs else jnp.asarray(entry.default)
             for name, entry in spec.items() if name in boundary_inputs or name in delivered}
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


def magnitude(value: Array) -> Array:
    """``|value|`` elementwise, for a step's scale: the magnitude of the arithmetic its identity's terms are formed by.
    The scale is a record no edge reads and no state the step takes depends on, so no derivative of the physics passes
    through it."""
    return jnp.abs(value)


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
    time. ``own_start`` and ``own_end`` are the storage without what a delivery adds (a delivered inertia's part),
    where the storage has such a part: the jump between steps they leave is the receiver's own state's, which no
    producer declares, and is judged (:func:`recorded`); where it has none they are the storage itself. ``claimed`` and
    ``claimed_scale`` are a route-1 receiver's claimed part, where it claims one (:func:`route_one_recorded`): the
    residual of the part of its step it books by formula and the magnitude that residual is judged on. ``stored``, a
    row per law (``LAW_BOOKS``' columns), is the receiver's laws' own books where some law it holds stores energy of its
    own (a Windkessel's capacitor, held in a flow's step): its storage at the step's start and end, its loss and its own
    numerical term, none of which is the receiver's, so none enters its identity (:func:`recorded`)."""

    work: Array
    hosted: Array
    unattributed: Array
    terms: Array
    storage_start: Array
    storage_end: Array
    scale: Array
    operators: Mapping[int, Array]
    own_start: Array | None = None
    own_end: Array | None = None
    claimed: Array | None = None
    claimed_scale: Array | None = None
    stored: Array | None = None

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
        **{f"{RECORDS}{name}{LOW}": zero(*shape) for name, shape in (
            ("work", (rows,)), ("hosted", (rows, 2)), ("terms", (terms,)), ("unattributed", ()), ("between", ()),
            ("between_own", ()))},
        f"{RECORDS}work": zero(rows), f"{RECORDS}step_work": zero(rows), f"{RECORDS}hosted": zero(rows, 2),
        f"{RECORDS}between": zero(), f"{RECORDS}between_own": zero(), f"{RECORDS}between_worst": zero(),
        f"{RECORDS}storage_own": as_run_array(storage),
        f"{RECORDS}terms": zero(terms), f"{RECORDS}step_terms": zero(terms),
        f"{RECORDS}unattributed": zero(), f"{RECORDS}unattributed_worst": zero(),
        f"{RECORDS}storage_start": as_run_array(storage), f"{RECORDS}storage": as_run_array(storage),
        f"{RECORDS}step_identity": zero(), f"{RECORDS}identity_worst": zero(), f"{RECORDS}scale": zero(),
        f"{RECORDS}indefinite_steps": zero(rows), f"{RECORDS}gift_steps": zero(rows),
    }


def unrecorded(state: Mapping[str, Array]) -> dict[str, Array]:
    """A metered receiver's state without the ledger's fields: what its step takes."""
    return {key: value for key, value in state.items() if not key.startswith(RECORDS)}


def summed_pair(state: Mapping[str, Array], name: str, term: Array) -> dict[str, Array]:
    """The record ``name``'s two fields after ``term`` is added to its running sum: the pair
    (:func:`~mime.core._error_free.add_to_pair`) the run keeps it as, so that the sum of a mixed-sign series does not
    drift with the steps (``MIME-ANO-149``)."""
    total, low = add_to_pair(state[f"{RECORDS}{name}"], state[f"{RECORDS}{name}{LOW}"], term)
    return {f"{RECORDS}{name}": total, f"{RECORDS}{name}{LOW}": low}


def recorded(state: Mapping[str, Array], step: StepAccount, held: Sequence[LedgerDelivery] = ()) -> dict[str, Array]:
    """The ledger's fields after ``step``: running sums, each check's worst over the run, and the step's own values
    beside them (what a recorded history shows per step). The storage at the step's start less the storage the step
    before ended at is kept apart and summed (``between``); its part no declared producer explains, the storage of
    the receiver's own state at the step's start (``StepAccount.own_start``) less the one the step before ended at,
    is kept summed (``between_own``) and as its largest magnitude over the step's scale (``between_worst``; over
    :data:`IDENTITY_ULPS` times eps it has failed, as an unattributed work above zero has). The
    identity's residual is kept relative to the step's scale
    (``step_identity``; over :data:`IDENTITY_ULPS` times eps it has failed), the unattributed work as its largest
    magnitude (above zero it has failed), and per law the count of steps whose operator failed its definiteness
    (``indefinite_steps``) and, per **dissipative** law of ``held``, the count of steps in which its paired work was
    positive beyond :data:`IDENTITY_ULPS` units of eps on the step's scale (but a delivery judged in its cluster's sum,
    :data:`CHECK_CLUSTER`, which its producer judges), or, for a law held in the step, its rule's
    removal at arrival negative beyond it (``gift_steps``: a dissipative law gives no energy, whatever it is
    delivered, neither through the step's solve nor by its rule at arrival; this is its class's sign, read from the
    law's own row, and it is silent on the identity, which books the gift under the law's name)."""
    dtype = step.work.dtype
    scale_at_least = jnp.maximum(step.scale, jnp.finfo(dtype).tiny)
    relative = jnp.abs(step.residual()) / scale_at_least
    indefinite = jnp.zeros_like(state[f"{RECORDS}indefinite_steps"])
    for row, operator in step.operators.items():
        indefinite = indefinite.at[row].set((~definite_at_floor(operator)).astype(dtype))
    gifts = jnp.zeros_like(state[f"{RECORDS}gift_steps"])
    rows = laws(held)
    floor = IDENTITY_ULPS * jnp.finfo(dtype).eps * step.scale
    # A law that stores energy of its own gives back what it stored, which is no gift: its loss is read net of it.
    given = step.work if step.stored is None else step.work + step.stored[:, 1] - step.stored[:, 0]
    for row in sorted({rows.index(d.law) for d in held if d.energy_class == DISSIPATIVE and d.check != CHECK_CLUSTER}):
        gave = (given[row] > floor) | (step.hosted[row, 0] < -floor)
        gifts = gifts.at[row].set(gave.astype(dtype))
    r = RECORDS
    start = {key: state[key] for key in (f"{r}configuration_start",) if key in state}
    jump = step.storage_start - state[f"{r}storage"]
    own_start = step.storage_start if step.own_start is None else step.own_start
    own_end = step.storage_end if step.own_end is None else step.own_end
    unexplained = own_start - state[f"{r}storage_own"]
    unexplained_relative = jnp.abs(unexplained) / scale_at_least
    if step.stored is not None:
        start = {**start, **law_recorded(state, step)}
    return {
        **start,
        **summed_pair(state, "work", step.work), f"{r}step_work": step.work,
        **summed_pair(state, "hosted", step.hosted),
        **summed_pair(state, "between", jump), **summed_pair(state, "between_own", unexplained),
        f"{r}between_worst": jnp.maximum(state[f"{r}between_worst"], unexplained_relative),
        f"{r}storage_own": own_end,
        **summed_pair(state, "terms", step.terms), f"{r}step_terms": step.terms,
        **summed_pair(state, "unattributed", step.unattributed),
        f"{r}unattributed_worst": jnp.maximum(state[f"{r}unattributed_worst"], jnp.abs(step.unattributed)),
        f"{r}storage_start": state[f"{r}storage_start"], f"{r}storage": step.storage_end,
        f"{r}step_identity": relative, f"{r}identity_worst": jnp.maximum(state[f"{r}identity_worst"], relative),
        f"{r}scale": state[f"{r}scale"] + step.scale,
        f"{r}indefinite_steps": state[f"{r}indefinite_steps"] + indefinite,
        f"{r}gift_steps": state[f"{r}gift_steps"] + gifts,
    }


def law_records(storage: Any) -> dict[str, Array]:
    """The records a receiver adds where a law it holds stores energy of its own, at the run's start: each law's
    storage there and now (``storage``, a row per law, zero for a law that stores nothing), its loss and its own
    numerical term summed (pairs, as :data:`PAIRED`'s), and the worst residual of its own identity."""
    held = as_run_array(storage)
    zero = jnp.zeros_like(held)
    r = RECORDS
    return {f"{r}law_storage_start": held, f"{r}law_storage": held, f"{r}law_loss": zero, f"{r}law_loss{LOW}": zero,
            f"{r}law_numerical": zero, f"{r}law_numerical{LOW}": zero,
            f"{r}law_identity_worst": as_run_array(jnp.zeros(()))}


def law_recorded(state: Mapping[str, Array], step: StepAccount) -> dict[str, Array]:
    """The storing laws' records after ``step``: each law's storage now, its loss and numerical term summed, and the
    worst over the run of its own identity's residual over its own scale (``LAW_BOOKS``' last column, never the
    receiver's step: a law's books are judged on the scale they came from), ``-W = dE + loss + N`` (the work it took
    from the receiver, ``-W``, its storage's change, its loss, its numerical term), together with the jump of its
    storage between steps (a law's state edited between steps is seen there, as a receiver's own is)."""
    assert step.stored is not None
    r = RECORDS
    start, end, loss, numerical, scale = (step.stored[:, column] for column in range(len(LAW_BOOKS)))
    held = state[f"{r}law_storage"]
    own = jnp.abs(-step.work - (end - start) - loss - numerical) + jnp.abs(start - held)
    # A law that stores nothing has a row of zeros, its scale among them, and is not judged.
    scale = scale + jnp.abs(held)
    worst = jnp.max(jnp.where(scale > 0.0, own / jnp.where(scale > 0.0, scale, 1.0), 0.0))
    return {f"{r}law_storage_start": state[f"{r}law_storage_start"], f"{r}law_storage": end,
            **summed_pair(state, "law_loss", loss), **summed_pair(state, "law_numerical", numerical),
            f"{r}law_identity_worst": jnp.maximum(state[f"{r}law_identity_worst"], worst)}


def balance(records: Mapping[str, Any], *, grouped: bool = False) -> Any:
    """A metered receiver's books over its run, from its records: the change in its storage less every paired work and
    what its storage jumped between steps where a producer is declared for the jump, its scheme's and its hosted laws'
    terms added back. The sum of its steps' residuals and of the jumps no declared producer explains
    (``between_own``): a state edited between steps is in the balance. ``grouped``, a member of a coupling group,
    whose jump the group owns whole: all of it is added back."""
    r = RECORDS
    declared = records[f"{r}between"] if grouped else records[f"{r}between"] - records[f"{r}between_own"]
    supplied = jnp.sum(records[f"{r}work"]) + records[f"{r}unattributed"] + declared
    numerical = jnp.sum(records[f"{r}terms"]) + jnp.sum(records[f"{r}hosted"])
    return (records[f"{r}storage"] - records[f"{r}storage_start"]) - supplied + numerical \
        + _below_last_place(records, grouped)


def _below_last_place(records: Mapping[str, Any], grouped: bool) -> Any:
    """The balance's terms that the running sums' pairs hold below their fields' last places, added apart from the
    fields' (their sum would round to the fields, in the run's dtype)."""
    r = RECORDS
    low = {name: records[f"{r}{name}{LOW}"] for name in PAIRED}
    declared = low["between"] if grouped else low["between"] - low["between_own"]
    supplied = jnp.sum(low["work"]) + low["unattributed"] + declared
    return jnp.sum(low["terms"]) + jnp.sum(low["hosted"]) - supplied


def losses(records: Mapping[str, Any], held: Sequence[LedgerDelivery]) -> Any:
    """What each law of ``held`` took from its receiver's motion over the run (J, a row per law, as its records keep
    them): a dissipative law's paired work turned to a loss, and, for one held in the step, the velocity rule's
    removal at arrival added (its projection's term is its host's scheme's, shown apart). Zero for any other class.
    Traced, as the records are."""
    rows = laws(held)
    dissipative = jnp.asarray([any(d.law == law and d.energy_class == DISSIPATIVE for d in held) for law in rows])
    taken = (-records[f"{RECORDS}work"] + records[f"{RECORDS}hosted"][:, 0]
             + (-records[f"{RECORDS}work{LOW}"] + records[f"{RECORDS}hosted{LOW}"][:, 0]))
    if f"{RECORDS}law_storage" in records:  # what a storing law holds at the run's end it has not lost
        taken = taken - (records[f"{RECORDS}law_storage"] - records[f"{RECORDS}law_storage_start"])
    return jnp.where(dissipative, taken, jnp.zeros_like(taken))


def between_floors(records: Mapping[str, Any]) -> Any:
    """The worst jump between steps no declared producer explains, in units of the step identity's rounding floor
    (above one it has failed, as the step identity has)."""
    return records[f"{RECORDS}between_worst"] / (IDENTITY_ULPS * jnp.finfo(jnp.asarray(
        records[f"{RECORDS}between_worst"]).dtype).eps)


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
    stored = node.law_storage(p, state)
    return {**state, **initial_records(held, len(terms), node.storage(p, state), node.configuration(p, state)),
            **(law_records(stored) if stored is not None else {})}


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
ROUTE_ONE_FIELDS = ("route_one_step", "route_one_worst", "route_one_negative_steps",
                    f"route_one_negative_steps{LOW}")


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
    (``dH - W + N`` summed, as for any receiver). **Route 1's check is that every step dissipated**: the work every law
    that is not dissipative supplied, the unattributed included, less the change in storage, ``D = W - dH``, is at
    least ``-IDENTITY_ULPS`` units of eps on the step's scale, where a law the receiver holds stores energy of its own
    (``StepAccount.stored``: an open flow's Windkessel) its storage's change counted with the receiver's, so that a
    capacitor giving back what it stored is not read as a gift. A step that gave energy (an energy gift larger than the
    step's own dissipation, by the identity ``D = D_true - G``), or whose books went non-finite, is counted
    (``route_one_negative_steps``, a running sum kept as a pair as :data:`PAIRED`'s are, its ``_low`` beside it);
    ``route_one_step`` is the step's ``D`` over its scale, ``route_one_worst`` its smallest over the run. **A part of
    the step the receiver books by formula** (``StepAccount.claimed``: a flow's drive as its stages applied it less the
    law's delivery) is judged as a step identity is, its residual over its own magnitude in the identity's fields
    (``step_identity``, ``identity_worst``; over :data:`IDENTITY_ULPS` units of eps it has failed): a gift booked
    under the law that delivered less, which the sign of ``D`` cannot see where the step's dissipation is larger. The
    identity's fields read zero where the receiver claims no part."""
    out = recorded(state, step, held)
    dtype = step.work.dtype
    rows = laws(held)
    dissipative = jnp.asarray([any(d.law == law and d.energy_class == DISSIPATIVE for d in held) for law in rows],
                              dtype=bool)
    supplied = jnp.sum(jnp.where(dissipative, 0.0, step.work)) + step.unattributed
    dissipated = supplied - (step.storage_end - step.storage_start)
    if step.stored is not None:  # a law storing energy of its own (a Windkessel) gives back what it stored: no gift
        dissipated = dissipated - jnp.sum(step.stored[:, 1] - step.stored[:, 0])
    relative = dissipated / jnp.maximum(step.scale, jnp.finfo(dtype).tiny)
    floor = IDENTITY_ULPS * jnp.finfo(dtype).eps
    r = RECORDS
    zero = jnp.zeros((), dtype=dtype)
    first = jnp.asarray(state[f"{r}scale"]) == 0.0
    worst = jnp.where(first, relative, jnp.minimum(state[f"{r}route_one_worst"], relative))
    claimed = zero if step.claimed is None or step.claimed_scale is None else \
        jnp.abs(step.claimed) / jnp.maximum(step.claimed_scale, jnp.finfo(dtype).tiny)
    return {**out, f"{r}step_identity": claimed,
            f"{r}identity_worst": jnp.maximum(state[f"{r}identity_worst"], claimed),
            f"{r}route_one_step": relative, f"{r}route_one_worst": worst,
            **summed_pair(state, "route_one_negative_steps", (~(relative >= -floor)).astype(dtype))}


def metered_route_one_step(node: MimeNode, spec: Mapping[str, BoundaryInputSpec], p: Mapping[str, Any],
                           state: Mapping[str, Array], boundary_inputs: Mapping[str, Array], dt: float,
                           step: Step) -> dict[str, Array]:
    """:func:`metered_step` for a route-1 receiver: its account recorded by :func:`route_one_recorded`."""
    held = deliveries(node.constants())
    own = unrecorded(state)
    ended, parts = step(p, own, summed(spec, held, boundary_inputs), dt)
    account = node.identity_terms(p, held, own, boundary_inputs, ended, parts, dt)
    return {**ended, **route_one_recorded(state, account, held)}


# -- what a scheme declares for the ledger harness (``mime.testing``): read by tests, never by a step --------------

@dataclass(frozen=True)
class TermOrder:
    """A numerical term's expected order: ``dt``, the power of the step its size per step falls with (2: summed over a
    fixed time it halves as the step does), and ``tolerance``, the power of the solve's tolerance it falls with;
    ``None`` where it does not fall with that argument, ``reason`` saying why (zero by construction, rounding).
    ``at_least``: the orders are a floor on how fast it falls, not its rate (a residual a solve's stop bounds by its
    tolerance, which the solve may land well under)."""

    dt: int | None = None
    tolerance: int | None = None
    reason: str = ""
    at_least: bool = False


@dataclass(frozen=True)
class FlagEvidence:
    """A flag's planted bug and its measured detection in one dtype: ``caught`` of ``runs`` planted runs read past the
    flag's line, ``false_alarms`` of ``clean_runs`` runs at the same draws without the plant, and ``evidence``, the
    measurement's run (a link)."""

    plant: str
    dtype: str
    caught: int
    runs: int
    false_alarms: int
    clean_runs: int
    evidence: str

    @property
    def every_run(self) -> bool:
        return self.runs > 0 and self.caught == self.runs and self.clean_runs > 0 and self.false_alarms == 0


@dataclass(frozen=True)
class LedgerFlag:
    """A ratio of a metered step's terms that flags a known bug: ``name`` (what a reading past the line names),
    ``ratio`` (the reading, in words), ``reading`` (``(node, p, before, after) -> Array``, the step's reading from the
    states either side of it, traced), ``line`` (by dtype, the largest reading a correct run gives, measured on drawn
    cases that ``clean`` states, the first ``settle`` steps of a run not judged: a state given from outside carries its
    own transient), ``evidence`` (each planted bug with its measured rate; a plant that is not caught every run says
    what the flag does not catch). **A flag is a check only when, in each dtype it has a line in, some plant is caught
    every run with no false alarm**; one that cannot be gated yet is declared ``pending`` with the reason and is never
    counted as a check."""

    name: str
    ratio: str
    reading: Callable[..., Any] | None
    line: Mapping[str, float]
    clean: str
    evidence: tuple[FlagEvidence, ...] = ()
    settle: int = 0
    pending: str = ""

    def shown(self, dtype: str) -> bool:
        """Whether some plant is caught every run with no false alarm in ``dtype``."""
        return any(e.every_run and e.dtype == dtype for e in self.evidence)

    @property
    def active(self) -> bool:
        return not self.pending and bool(self.line) and all(self.shown(dtype) for dtype in self.line)


__all__ = ["CHECK_CLUSTER", "CHECK_DEFINITE", "DEFINITE_ULPS", "DELIVERY_KINDS", "DISSIPATIVE", "ENERGY_CLASSES",
           "FLOW_IN", "HOSTED", "IDENTITY_ULPS", "INERTIA", "LEDGER", "LOAD", "LOW", "PAIRED", "POTENTIAL", "RECORDS",
           "LAW_BOOKS", "LAW_RECORDS", "ROUTE_ONE_FIELDS", "SOLVED", "UNMETERED", "WORKLESS", "FlagEvidence", "LedgerDelivery", "LedgerFlag",
           "LawRecords", "LedgerRecords", "StepAccount", "TermOrder",
           "balance", "balance_floor", "between_floors", "definite_at_floor", "deliveries", "initial_records",
           "law_input", "law_recorded", "law_records", "laws", "losses", "magnitude", "metered_route_one_step", "metered_step", "own_input",
           "paired_work", "recorded", "route_one_recorded", "route_one_records", "stored", "summed", "summed_pair",
           "unrecorded", "with_law_inputs", "with_records"]
