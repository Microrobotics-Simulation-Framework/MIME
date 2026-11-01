"""The saved form's two files as closed TypedDicts, and the run-time check a loader makes of them.

A TypedDict does nothing at run time, so :func:`shape_problems` checks a value read from disk against the same
annotations a type checker reads: an unknown key, a missing key or a value of another type is a problem, named by its
path. The check is driven by the annotations themselves, and refuses a typing form it does not know, so the static
schema and the run-time check cannot drift apart without a failure.
"""


import types
import typing
from collections.abc import Sequence
from typing import Any, Literal, Never, TypeAliasType, cast

from typing_extensions import NotRequired, ReadOnly, Required, TypedDict, get_type_hints, is_typeddict

DOCUMENT_SCHEMA = "mime.document/4"
LOCK_SCHEMA = "mime.lock/4"

type SettingValue = (None | bool | int | float | str | list[SettingValue] | QuantityDoc | AssetDoc | FileDoc
                     | GroupDoc)
type GroupedValue = None | bool | int | float | str | list[GroupedValue] | QuantityDoc | AssetDoc | FileDoc
type Json = None | bool | int | float | str | list[Json] | dict[str, Json]


class PendingDoc(TypedDict, closed=True):
    """A reason left pending (``pending()``): ``{pending: true}``."""

    pending: ReadOnly[Literal[True]]


class ChosenDoc(TypedDict, closed=True):
    """A value the experimenter chose, with why (``chosen(...)``): ``{chosen: <text>}``."""

    chosen: ReadOnly[str]


#: A reason's basis, keyed by its kind: a text as a plain string, ``{pending: true}``, ``{chosen: <text>}``. a1's named
#: bases add a kind of their own key, so a document's shape is kept.
type ReasonDoc = str | PendingDoc | ChosenDoc


class QuantityDoc(TypedDict, closed=True):
    """A quantity as SI value and SI unit symbol (``{value: 0.02, unit: s}``), with its reason where it carries one."""

    value: ReadOnly[float | list[float]]
    unit: ReadOnly[str]
    reason: NotRequired[ReadOnly[ReasonDoc]]


class AssetDoc(TypedDict, closed=True):
    """An asset by name, with the sha256 of the contents the experiment was written against
    (``{asset: dejongh_mlp_v2, sha256: ...}``): a load refuses other contents under that name."""

    asset: ReadOnly[str]
    sha256: ReadOnly[str]


class FileDoc(TypedDict, closed=True):
    """A file an entry reads (a robot's description), by its path below the document's folder, where ``save`` writes
    it, with the SHA-256 of its bytes (``{file: descriptions/<pin prefix>-ar4.urdf, sha256: ...}``): an entry reading
    other bytes is refused."""

    file: ReadOnly[str]
    sha256: ReadOnly[str]


class GroupDoc(TypedDict, closed=True):
    """A group of settings written together (``Lubrication(epsilon=...)``), by its registered group name, with its
    settings written differently from their defaults; a group holds no group."""

    group: ReadOnly[str]
    settings: ReadOnly[dict[str, GroupedValue]]


class EntryDoc(TypedDict, closed=True):
    """A registered entry by name and version, with the settings written differently from its defaults. An alias
    such as ``ToyDrag.Quadratic()`` saves as its family with the rung among the settings."""

    name: ReadOnly[str]
    version: ReadOnly[str]
    settings: NotRequired[dict[str, SettingValue]]


class BoundEntryDoc(TypedDict, closed=True):
    """An interaction or an observation: an entry with its id and the things or fields bound to its roles, by
    name."""

    id: ReadOnly[str]
    name: ReadOnly[str]
    version: ReadOnly[str]
    binds: dict[str, str]
    settings: NotRequired[dict[str, SettingValue]]


class DynamicsDoc(TypedDict, closed=True):
    """A thing's dynamics, one entry per aspect: a body's or a structure's motion, a medium's flow."""

    motion: NotRequired[EntryDoc]
    flow: NotRequired[EntryDoc]


class MagnetDoc(TypedDict, closed=True):
    """A part's role as a magnet: its model, and its direction in the part's frame when the experiment declares it
    rather than the part's prim."""

    model: ReadOnly[Literal["permanent_magnet"]]
    direction: NotRequired[list[float]]


class PartDoc(TypedDict, closed=True):
    """A part: its solid (a geometry kind, by name and version with its settings), where its centroid sits in a body
    written Python-first, or in the frame an apparatus's part is mounted on, the material it is made of, by name, the
    part it lies inside, and an apparatus's part's mount (``<segment>.<frame>``)."""

    solid: EntryDoc
    at: NotRequired[QuantityDoc]
    material: NotRequired[str]
    inside: NotRequired[str]
    mount: NotRequired[str]


class SegmentDoc(TypedDict, closed=True):
    """A segment of an apparatus's chain: its name in the chain, and its dynamics by registered name and version with
    its settings. A chain is a list of them, in chain order."""

    segment: ReadOnly[str]
    name: ReadOnly[str]
    version: ReadOnly[str]
    settings: NotRequired[dict[str, SettingValue]]


class ThingDoc(TypedDict, closed=True):
    """A thing: its kind, its prim (a scene path, an experiment name like the thing's own) and dynamics; a body's
    initial state, its parts and their roles; a structure's geometry kind; the region a medium fills
    (``tube.lumen``) and its material, by name; an apparatus's chain, parts and magnets, its fields and the drive of
    each command."""

    kind: ReadOnly[Literal["structure", "medium", "body", "apparatus"]]
    prim: ReadOnly[str]
    dynamics: NotRequired[DynamicsDoc]
    initial: NotRequired[dict[str, SettingValue]]
    fields: NotRequired[dict[str, EntryDoc]]
    geometry: NotRequired[EntryDoc]
    fills: NotRequired[str]
    material: NotRequired[str]
    commands: NotRequired[dict[str, EntryDoc]]
    magnets: NotRequired[dict[str, MagnetDoc]]
    parts: NotRequired[dict[str, PartDoc]]
    chain: NotRequired[list[SegmentDoc]]


class DeclaredDoc(TypedDict, closed=True):
    """A value declared in the experiment (``umr.mass``): in SI, with its unit and its reason."""

    value: ReadOnly[float | list[float]]
    unit: ReadOnly[str]
    reason: ReadOnly[ReasonDoc]


class ConditionDoc(TypedDict, closed=True):
    """An operating condition every record is read at: its value and tolerance in SI, with their unit, and the
    value's reason where it carries one."""

    value: ReadOnly[float]
    plus_minus: ReadOnly[float]
    unit: ReadOnly[str]
    reason: NotRequired[ReadOnly[ReasonDoc]]


class RecordDoc(TypedDict, closed=True):
    """A record file the experiment names: its file below the document's folder, where ``save`` writes it, the
    SHA-256 of its bytes, and the record's id, version and content hash. A reference: never a value from it."""

    file: ReadOnly[str]
    sha256: ReadOnly[str]
    id: ReadOnly[str]
    version: ReadOnly[str]
    content_hash: ReadOnly[str]


class AcceptanceDoc(TypedDict, closed=True):
    """The experiment's acceptance of a read answered ``unknown``: the reasons it covers, and why."""

    covers: ReadOnly[list[str]]
    reason: ReadOnly[ReasonDoc]


class NominalDoc(TypedDict, closed=True):
    """The value the experiment uses from an interval, in SI with its unit, and why: the experiment's number, never
    the record's."""

    value: ReadOnly[float]
    unit: ReadOnly[str]
    reason: ReadOnly[ReasonDoc]


class ExtrapolationDoc(TypedDict, closed=True):
    """How the experiment extends a table past its readable range, with its basis and reason."""

    method: ReadOnly[Literal["constant", "linear"]]
    basis: ReadOnly[str]
    reason: ReadOnly[str]


class MaterialDoc(TypedDict, closed=True):
    """A material read from a record: the record's reference, the tokens the experiment accepts in every read
    (``require``), and by record field its acceptances, nominal values and extrapolations, each with its reason."""

    record: ReadOnly[RecordDoc]
    require: NotRequired[dict[str, list[str]]]
    accept: NotRequired[dict[str, AcceptanceDoc]]
    nominal: NotRequired[dict[str, NominalDoc]]
    extrapolation: NotRequired[dict[str, ExtrapolationDoc]]


class RunDoc(TypedDict, closed=True):
    dt: QuantityDoc
    seed: int


class ReferenceDoc(TypedDict, closed=True):
    """An asset layer referenced into the scene: the prim it is referenced at, its file below the document's
    folder, and its pin."""

    at: ReadOnly[str]
    layer: ReadOnly[str]
    sha256: ReadOnly[str]


class SceneDoc(TypedDict, closed=True):
    """The scene: up, once, the unit its stage is written in, and the asset layers it references."""

    up: list[float]
    units: str
    references: list[ReferenceDoc]


class ControllerDoc(TypedDict, closed=True):
    """The experiment's controller: its entry by name and version with the settings written differently from their
    defaults; its mode, by word; each read label's path and each write label's command; its rate (a period) and
    latency, when declared."""

    name: ReadOnly[str]
    version: ReadOnly[str]
    settings: NotRequired[dict[str, SettingValue]]
    mode: ReadOnly[Literal["graph", "callback", "client", "fmu"]]
    reads: dict[str, str]
    writes: dict[str, str]
    rate: NotRequired[QuantityDoc]
    latency: NotRequired[QuantityDoc]


class CustomLicenceDoc(TypedDict, closed=True):
    """A custom licence by its ``LicenseRef-`` id: its title, its https link, and what a person says it allows
    (``unknown`` or ``not_permitted``; a licence that permits is a permission)."""

    title: ReadOnly[str]
    url: ReadOnly[str]
    redistribution: ReadOnly[Literal["not_permitted", "unknown"]]


class TermsDoc(TypedDict, closed=True):
    """A third-party file's terms, by its path below the folder: the sha256 of the bytes they were declared for (a
    changed file inherits nothing), its licence as an SPDX expression, the credit its licence asks for, its role, and
    where its exact bytes can be fetched from, primary first."""

    sha256: ReadOnly[str]
    licence: ReadOnly[str]
    credit: ReadOnly[str]
    role: ReadOnly[Literal["results", "presentation"]]
    origin: NotRequired[ReadOnly[list[str]]]


class ScopeDoc(TypedDict, closed=True):
    """What a permission allows."""

    redistribute: ReadOnly[bool]
    modify: ReadOnly[bool]
    commercial: ReadOnly[bool]


class EvidenceDoc(TypedDict, closed=True):
    """A permission's evidence: its kind and its pointer (a public statement's link, or what a written permission is
    and who holds it)."""

    kind: ReadOnly[Literal["public_statement", "written_permission"]]
    pointer: ReadOnly[str]


class PermissionDoc(TypedDict, closed=True):
    """A permission as declared: the rights holder and who was permitted, the files by sha256, the scope, the date
    and the evidence. Its form is checked; its truth never is."""

    grantor: ReadOnly[str]
    grantee: ReadOnly[str]
    files: ReadOnly[list[str]]
    scope: ReadOnly[ScopeDoc]
    date: ReadOnly[str]
    evidence: ReadOnly[EvidenceDoc]


class ExperimentDocument(TypedDict, closed=True):
    """What the experiment declares, written by registered name and version, settings and bindings, and nothing
    else: never code, never a copy of a record's values. ``profile`` is the profile it is built under; ``declared``
    holds the values declared in the experiment, each with its reason; ``sources`` each setting's source given apart
    from its value (``exp.source``); ``gravity`` is its magnitude, along ``-up``, absent where the experiment states
    none (standard gravity, by convention); ``conditions`` the operating conditions every record is read at;
    ``materials`` each material by name, as a reference to its record; ``controller`` the controller, or null.

    Version 4 adds what may be shared, each absent where nothing is declared: ``licence``, the experiment's own (an
    SPDX expression; absent, it declares none, which ``save`` and ``pack`` never refuse and ``push`` will); ``licences``,
    each custom licence named, by its ``LicenseRef-`` id; ``terms``, each third-party file's, by its path below the
    folder; and ``permissions``, each declared permission (``mime.terms``)."""

    schema: ReadOnly[Literal["mime.document/4"]]
    mime: ReadOnly[str]
    experiment: ReadOnly[str]
    profile: ReadOnly[Literal["exploratory", "research", "strict"]]
    run: RunDoc
    scene: SceneDoc
    gravity: NotRequired[QuantityDoc]
    sources: dict[str, ReasonDoc]
    conditions: dict[str, ConditionDoc]
    things: dict[str, ThingDoc]
    interactions: list[BoundEntryDoc]
    observations: list[BoundEntryDoc]
    declared: dict[str, DeclaredDoc]
    materials: dict[str, MaterialDoc]
    controller: ControllerDoc | None
    licence: NotRequired[str]
    licences: NotRequired[dict[str, CustomLicenceDoc]]
    terms: NotRequired[dict[str, TermsDoc]]
    permissions: NotRequired[list[PermissionDoc]]


class GpuRecordDoc(TypedDict, closed=True):
    """The GPU a build ran on (``mime.env.GpuRecord``'s keys and types, which a test holds together): JAX's device
    kind, the driver's release, and the CUDA driver's, CUDA runtime's and cuDNN's versions as JAX reports them."""

    device_kind: ReadOnly[str]
    driver: ReadOnly[str | None]
    cuda_driver: ReadOnly[int | None]
    cuda_runtime: ReadOnly[int | None]
    cudnn: ReadOnly[int | None]


class CpuRecordDoc(TypedDict, closed=True):
    """The CPU a build ran on (``mime.env.CpuRecord``'s keys and types): its model and the hash of its flags."""

    model: ReadOnly[str | None]
    flags_sha256: ReadOnly[str | None]


class EnvironmentRecord(TypedDict, closed=True):
    """What a build ran under. A load compares it and reports every difference; it never refuses on one. Besides the
    versions, the platform, the device and the precision: the GPU (``None`` on a CPU) and the CPU, the matmul
    precision, XLA's flags and whether they turn its deterministic GPU mode on, the runner's own record of strings,
    which MIME stores and compares and never reads, and the runner's information, strings it stores and never compares
    (a workspace's image ID, a client's version)."""

    mime: ReadOnly[str]
    maddening: ReadOnly[str]
    openmaterica: ReadOnly[str]
    usd_core: ReadOnly[str]
    jax: ReadOnly[str]
    jaxlib: ReadOnly[str]
    python: ReadOnly[str]
    platform: ReadOnly[str]
    device: ReadOnly[str]
    x64: ReadOnly[bool]
    prng: ReadOnly[str]
    gpu: ReadOnly[GpuRecordDoc | None]
    cpu: ReadOnly[CpuRecordDoc]
    matmul_precision: ReadOnly[str | None]
    xla_flags: ReadOnly[str]
    deterministic: ReadOnly[bool]
    runner: ReadOnly[dict[str, str]]
    runner_info: ReadOnly[dict[str, str]]


class EntryRecord(TypedDict, closed=True):
    """One entry the experiment uses, and the distribution it came from, so that a lab's entry pins its own
    package."""

    kind: ReadOnly[Literal["interaction", "observation", "dynamics", "field provider", "controller", "drive",
                           "geometry"]]
    name: ReadOnly[str]
    version: ReadOnly[str]
    distribution: ReadOnly[str]
    distribution_version: ReadOnly[str]


class HashRecord(TypedDict, closed=True):
    """``document``: the document in canonical JSON. ``input``: the document with the resolved record, the entries
    and the canonical order, which a load must reproduce exactly. ``derived`` and ``observations``: the graph built,
    physics and observations apart."""

    recipe: ReadOnly[str]
    document: ReadOnly[str]
    input: ReadOnly[str]
    derived: ReadOnly[str]
    observations: ReadOnly[str]


class ResolvedRun(TypedDict, closed=True):
    """The run settings in SI, the world frame with gravity's vector derived, and how a run executes."""

    experiment: ReadOnly[str]
    dt: ReadOnly[float]
    seed: ReadOnly[int]
    up: ReadOnly[list[float]]
    gravity: ReadOnly[list[float]]
    structure: ReadOnly[str]


class ResolvedEntry(TypedDict, closed=True):
    """One declared entry as the build resolved it: every setting, defaults included; a body's initial state in SI;
    the declared gradient status of what it built, its rung's where its rungs declare their own; the nodes it built
    and how many base steps each takes."""

    kind: ReadOnly[str]
    name: ReadOnly[str]
    version: ReadOnly[str]
    differentiability: ReadOnly[str]
    binds: ReadOnly[dict[str, str]]
    settings: ReadOnly[dict[str, SettingValue]]
    initial: ReadOnly[dict[str, SettingValue]]
    nodes: ReadOnly[list[str]]
    every: ReadOnly[int]


class ValueRecord(TypedDict, closed=True):
    """A value of a thing the entries read: in SI, where it came from (``declared in the experiment``, a record, its
    parts) and why, the ids that read it, and one line per input it was read or derived from."""

    value: ReadOnly[float | list[float]]
    unit: ReadOnly[str]
    source: ReadOnly[str]
    reason: ReadOnly[str]
    read_by: ReadOnly[list[str]]
    details: ReadOnly[list[str]]


class ReadRecord(TypedDict, closed=True):
    """One read of a record: the request as sent (the record by its id, version and content hash, the field, the
    operating conditions and the requirement), OpenMaterica's status and reasons as it gave them, how the build used
    the verdict, and the value it took or the bounds it read."""

    material: ReadOnly[str]
    field: ReadOnly[str]
    request: ReadOnly[Json]
    status: ReadOnly[str]
    reasons: ReadOnly[list[str]]
    use: ReadOnly[str]
    value: NotRequired[ReadOnly[float]]
    bounds: NotRequired[ReadOnly[list[float]]]


class GeometryRecord(TypedDict, closed=True):
    """A structure's geometry kind as the build resolved it: every setting, defaults included."""

    name: ReadOnly[str]
    version: ReadOnly[str]
    settings: ReadOnly[dict[str, SettingValue]]


class PartRecord(TypedDict, closed=True):
    """A body's part as resolved: its solid with every setting, its volume, its centroid and its frame's rotation in
    the body's frame, its inertia per unit mass about its centroid in the body's frame, the part it lies inside, its
    material, and whether it is a magnet."""

    solid: ReadOnly[GeometryRecord]
    volume: ReadOnly[float]
    centre: ReadOnly[list[float]]
    rotation: ReadOnly[list[list[float]]]
    inertia_per_mass: ReadOnly[list[list[float]]]
    inside: ReadOnly[str | None]
    material: ReadOnly[str | None]
    magnet: ReadOnly[bool]


class ResolvedMaterials(TypedDict, closed=True):
    """The conditions, each record's pins, every read the values rest on, and each body's parts."""

    conditions: ReadOnly[dict[str, ConditionDoc]]
    records: ReadOnly[dict[str, RecordDoc]]
    reads: ReadOnly[list[ReadRecord]]
    parts: ReadOnly[dict[str, dict[str, PartRecord]]]


class RegistrationRecord(TypedDict, closed=True):
    id: ReadOnly[str]
    kind: ReadOnly[str]
    subject: ReadOnly[str]


class SlotRecord(TypedDict, closed=True):
    name: ReadOnly[str]
    reader: ReadOnly[str]
    interfaces: ReadOnly[list[str]]


class GroupRecord(TypedDict, closed=True):
    nodes: ReadOnly[list[str]]
    tolerance: ReadOnly[float]
    max_iterations: ReadOnly[int]
    requested_by: ReadOnly[list[str]]


class AssetRecord(TypedDict, closed=True):
    """An asset an experiment's entries read: where it is read from and the sha256 of its contents."""

    source: ReadOnly[str]
    sha256: ReadOnly[str]


class RemoteAssetRecord(TypedDict, closed=True):
    """A remote asset as the lock names it, for a file too large to be saved with the experiment: ``mime.assets``'s
    ``AssetRecord`` (a test holds the two together). Locations are tried in order."""

    name: ReadOnly[str]
    sha256: ReadOnly[str]
    size: ReadOnly[int]
    licence: ReadOnly[str]
    locations: ReadOnly[list[str]]


class MagnetRecord(TypedDict, closed=True):
    """A body's magnet as resolved: its part's prim, its model, its direction in the part's frame and in the
    body's, and where the direction came from."""

    part: ReadOnly[str]
    model: ReadOnly[str]
    in_part: ReadOnly[list[float]]
    in_body: ReadOnly[list[float]]
    source: ReadOnly[str]


class ResolvedScene(TypedDict, closed=True):
    """The scene as resolved: up, the stage's unit and metres per unit, every thing's prim, the things bound to a
    prim an asset brings in, and each body's magnets by their key."""

    up: ReadOnly[list[float]]
    units: ReadOnly[str]
    meters_per_unit: ReadOnly[float]
    prims: ReadOnly[dict[str, str]]
    bound: ReadOnly[list[str]]
    magnets: ReadOnly[dict[str, dict[str, MagnetRecord]]]


class DescriptionRecord(TypedDict, closed=True):
    """A chain segment's description as read: the file and its SHA-256, every element by its group (read and used,
    recognised and not used, refused) as ``{kind: [paths]}``, the refused elements read past with their reasons, and
    the links read as massless."""

    file: ReadOnly[str]
    sha256: ReadOnly[str]
    used: ReadOnly[dict[str, list[str]]]
    unused: ReadOnly[dict[str, list[str]]]
    refused: ReadOnly[dict[str, list[str]]]
    acknowledged: ReadOnly[dict[str, str]]
    massless: ReadOnly[list[str]]


class ControllerReadRecord(TypedDict, closed=True):
    """A controller's read as the build resolved it: the path, its class (``state`` or ``observation``) and what
    that class means, and the node field the run reads."""

    path: ReadOnly[str]
    read_class: ReadOnly[Literal["state", "observation"]]
    meaning: ReadOnly[str]
    node: ReadOnly[str]
    field: ReadOnly[str]


class ControllerWriteRecord(TypedDict, closed=True):
    """A controller's write as the build resolved it: the command path, and the node input it feeds as an external
    input of the graph."""

    path: ReadOnly[str]
    node: ReadOnly[str]
    field: ReadOnly[str]


class ResolvedController(TypedDict, closed=True):
    """The controller as the build resolved it: its mode by number, its period and latency in base steps with the rule
    the latency follows, each read with its class and each write with the input it feeds."""

    mode: ReadOnly[int]
    every: ReadOnly[int]
    latency: ReadOnly[int]
    rule: ReadOnly[str]
    reads: ReadOnly[dict[str, ControllerReadRecord]]
    writes: ReadOnly[dict[str, ControllerWriteRecord]]


class WaiverRecord(TypedDict, closed=True):
    """A waiver the build rests on with a pending basis: what it waives, the codes it covers, its basis, and whether
    the profile generated it or the experiment declared it."""

    target: ReadOnly[str]
    reasons: ReadOnly[list[str]]
    basis: ReadOnly[ReasonDoc]
    generated: ReadOnly[bool]


class DefaultRecord(TypedDict, closed=True):
    """A default the build took: its kind, the value, and the entry whose setting it is (``drag 0.3.0``; ``None`` for a
    run setting). Never its rule's or basis's words, so a reworded basis refuses no saved folder."""

    kind: ReadOnly[Literal["determined", "convention", "numerical", "undeclared"]]
    value: ReadOnly[SettingValue]
    entry: ReadOnly[str | None]


class ItemRecord(TypedDict, closed=True):
    """An item the profile left pending or lists: what it is about, and the finding codes it covers."""

    target: ReadOnly[str]
    codes: ReadOnly[list[str]]


class ResolvedProfile(TypedDict, closed=True):
    """What the profile made of the experiment, each in a canonical order (by target, then code): the profile, each
    item left pending (what the stamp names), every waiver with a pending basis, each default taken by path, each
    setting's source by path, what the profile lists, and the values given as the experimenter's own choice, each with
    its declared role."""

    profile: ReadOnly[str]
    pending: ReadOnly[list[ItemRecord]]
    waivers: ReadOnly[list[WaiverRecord]]
    defaults: ReadOnly[dict[str, DefaultRecord]]
    sources: ReadOnly[dict[str, ReasonDoc]]
    listed: ReadOnly[list[ItemRecord]]
    changes: ReadOnly[dict[str, str]]


class ResolvedRecord(TypedDict, closed=True):
    """What the build derived from the document: what a reader needs to see what ran without that version's code,
    and what a load compares item by item."""

    vocabulary: ReadOnly[dict[str, str]]
    run: ReadOnly[ResolvedRun]
    things: ReadOnly[dict[str, str]]
    entries: ReadOnly[dict[str, ResolvedEntry]]
    order: ReadOnly[list[str]]
    registrations: ReadOnly[dict[str, list[RegistrationRecord]]]
    slots: ReadOnly[dict[str, list[SlotRecord]]]
    groups: ReadOnly[list[GroupRecord]]
    values: ReadOnly[dict[str, ValueRecord]]
    geometry: ReadOnly[dict[str, GeometryRecord]]
    fills: ReadOnly[dict[str, str]]
    assets: ReadOnly[dict[str, AssetRecord]]
    scene: ReadOnly[ResolvedScene]
    materials: ReadOnly[ResolvedMaterials]
    descriptions: ReadOnly[dict[str, DescriptionRecord]]
    controller: ReadOnly[ResolvedController | None]
    profile: ReadOnly[ResolvedProfile]


class DerivedRecord(TypedDict, closed=True):
    """The derived graph, saved with its hash: MADDENING's description of it, arrays by content hash."""

    physics: ReadOnly[dict[str, Json]]
    observations: ReadOnly[dict[str, Json]]


class ReferenceRecord(TypedDict, closed=True):
    """The golden outputs ``verify`` reruns against: every field of every node over the first ``steps`` steps, in
    ``file``, with the tolerance used across environments (within one environment they match bit for bit)."""

    file: ReadOnly[str]
    sha256: ReadOnly[str]
    steps: ReadOnly[int]
    rtol: ReadOnly[float]
    atol: ReadOnly[float]


class RecordFileRecord(TypedDict, closed=True):
    """A record file the saved form carries: its file below the folder and the SHA-256 of its bytes."""

    file: ReadOnly[str]
    sha256: ReadOnly[str]


class StageRecord(TypedDict, closed=True):
    """The stage the build wrote: its file below the folder and its SHA-256, and every layer it references, by its
    file below the folder, with its SHA-256. The stage's identity."""

    file: ReadOnly[str]
    sha256: ReadOnly[str]
    layers: ReadOnly[dict[str, str]]


class Lock(TypedDict, closed=True):
    """What a build wrote beside the document: the environment, the entries and their distributions, the derived
    seeds, the hashes, the resolved record, the derived graph, the stage, the record files by material and the
    reference outputs; and, where there are any, the remote assets and the files that travel as references."""

    schema: ReadOnly[Literal["mime.lock/4"]]
    environment: ReadOnly[EnvironmentRecord]
    entries: ReadOnly[list[EntryRecord]]
    seeds: ReadOnly[dict[str, dict[str, int]]]
    hashes: ReadOnly[HashRecord]
    resolved: ReadOnly[ResolvedRecord]
    derived: ReadOnly[DerivedRecord]
    stage: ReadOnly[StageRecord]
    records: ReadOnly[dict[str, RecordFileRecord]]
    reference: ReadOnly[ReferenceRecord]
    assets: NotRequired[ReadOnly[list[RemoteAssetRecord]]]
    #: Each named file whose bytes travel as a reference, by its path below the folder, to the name of the asset in
    #: ``assets`` that holds them (version 4): a file that may not be redistributed, or one over the size a bundle
    #: carries that has an origin. Outside the input hash, as ``assets`` is.
    referenced_files: NotRequired[ReadOnly[dict[str, str]]]


_WRAPPERS = (ReadOnly, NotRequired, Required)


def _unwrap(annotation: Any) -> Any:
    while typing.get_origin(annotation) in _WRAPPERS:
        annotation = typing.get_args(annotation)[0]
    return annotation


def typed_dict_fields(cls: type) -> dict[str, Any]:
    """A TypedDict's keys and their value types, with ``ReadOnly`` and ``NotRequired`` taken off."""
    return {key: _unwrap(value) for key, value in get_type_hints(cls, include_extras=True).items()}


def _name(annotation: Any) -> str:
    return getattr(annotation, "__name__", None) or str(annotation)


def shape_problems(value: object, annotation: Any, path: str = "$") -> list[str]:
    """Every way ``value`` differs from ``annotation``, each with its path; empty when it conforms."""
    annotation = _unwrap(annotation)
    if isinstance(annotation, TypeAliasType):
        return shape_problems(value, annotation.__value__, path)
    if annotation is None or annotation is type(None):
        return [] if value is None else [f"{path}: {value!r} is not null"]
    if annotation is Never:
        return [f"{path}: nothing may be written here yet (the slot is kept empty until it is built)"]
    if annotation is bool:
        return [] if isinstance(value, bool) else [f"{path}: {value!r} is not a boolean"]
    if annotation is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
        return [] if ok else [f"{path}: {value!r} is not an integer"]
    if annotation is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        return [] if ok else [f"{path}: {value!r} is not a number"]
    if annotation is str:
        return [] if isinstance(value, str) else [f"{path}: {value!r} is not a string"]
    origin = typing.get_origin(annotation)
    if origin is Literal:
        allowed = typing.get_args(annotation)
        return [] if any(type(value) is type(a) and value == a for a in allowed) else [
            f"{path}: {value!r} is not one of {', '.join(repr(a) for a in allowed)}"]
    if origin in (types.UnionType, typing.Union):
        members = typing.get_args(annotation)
        found = [shape_problems(value, member, path) for member in members]
        if any(not problems for problems in found):
            return []
        return [f"{path}: {value!r} is none of {' | '.join(_name(m) for m in members)}"]
    if origin is list:
        if not isinstance(value, list):
            return [f"{path}: {value!r} is not a list"]
        (item,) = typing.get_args(annotation)
        return [p for i, element in enumerate(cast(list[object], value)) for p in shape_problems(element, item,
                                                                                               f"{path}[{i}]")]
    if origin is dict:
        if not isinstance(value, dict):
            return [f"{path}: {value!r} is not a mapping"]
        key_type, item = typing.get_args(annotation)
        if key_type is not str:
            raise TypeError(f"the saved form's mappings have string keys, not {key_type!r}")
        problems: list[str] = []
        for key, element in cast(dict[object, object], value).items():
            if not isinstance(key, str):
                problems.append(f"{path}: the key {key!r} is not a string")
            else:
                problems += shape_problems(element, item, f"{path}.{key}")
        return problems
    if isinstance(annotation, type) and is_typeddict(annotation):
        return _typed_dict_problems(value, annotation, path)
    raise TypeError(f"the saved form's check does not know the typing form {annotation!r}")


def _typed_dict_problems(value: object, cls: type, path: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path}: {value!r} is not a mapping"]
    held = cast(dict[object, object], value)
    fields = typed_dict_fields(cls)
    required: frozenset[str] = getattr(cls, "__required_keys__")
    problems = [f"{path}: unknown key {key!r} ({cls.__name__} holds only {', '.join(sorted(fields))})"
                for key in held if key not in fields]
    problems += [f"{path}: missing key {key!r}" for key in sorted(required) if key not in held]
    for key, annotation in fields.items():
        if key in held:
            problems += shape_problems(held[key], annotation, f"{path}.{key}")
    return problems


def reachable_typed_dicts(root: type) -> list[type]:
    """Every TypedDict ``root``'s annotations reach, ``root`` first: what the check and its test walk."""
    found: list[type] = []
    aliases: set[TypeAliasType] = set()

    def visit(annotation: Any) -> None:
        annotation = _unwrap(annotation)
        if isinstance(annotation, TypeAliasType):
            if annotation not in aliases:
                aliases.add(annotation)
                visit(annotation.__value__)
        elif isinstance(annotation, type) and is_typeddict(annotation):
            if annotation not in found:
                found.append(annotation)
                for item in typed_dict_fields(annotation).values():
                    visit(item)
        else:
            for argument in typing.get_args(annotation):
                if not isinstance(argument, (str, int, bool)) and argument is not None:
                    visit(argument)

    visit(root)
    return found


def conforms[T](value: object, schema: type[T]) -> tuple[T | None, Sequence[str]]:
    """``value`` as a ``schema`` when it conforms, and every problem otherwise."""
    problems = shape_problems(value, schema)
    return (cast(T, value) if not problems else None), problems


__all__ = [
    "ChosenDoc", "CustomLicenceDoc", "DefaultRecord", "EvidenceDoc", "ItemRecord", "PendingDoc", "PermissionDoc",
    "ScopeDoc", "TermsDoc",
    "ReasonDoc", "ResolvedProfile", "WaiverRecord",
    "DOCUMENT_SCHEMA", "LOCK_SCHEMA", "AcceptanceDoc", "AssetDoc", "AssetRecord", "BoundEntryDoc", "ConditionDoc",
    "ControllerDoc", "ControllerReadRecord", "ControllerWriteRecord", "CpuRecordDoc", "DeclaredDoc", "DerivedRecord",
    "DescriptionRecord", "DynamicsDoc", "EntryDoc", "EntryRecord", "EnvironmentRecord", "ExperimentDocument",
    "ExtrapolationDoc", "FileDoc", "GeometryRecord", "GpuRecordDoc", "GroupDoc", "GroupRecord", "GroupedValue",
    "HashRecord", "Json",
    "Lock", "MagnetDoc", "MagnetRecord", "MaterialDoc", "NominalDoc", "PartDoc", "PartRecord", "QuantityDoc",
    "ReadRecord", "RecordDoc", "RecordFileRecord", "ReferenceDoc", "ReferenceRecord", "RegistrationRecord",
    "RemoteAssetRecord",
    "ResolvedController", "ResolvedEntry", "ResolvedMaterials", "ResolvedRecord", "ResolvedRun", "ResolvedScene",
    "RunDoc", "SceneDoc", "SegmentDoc", "SettingValue", "SlotRecord", "StageRecord", "ThingDoc", "ValueRecord",
    "conforms", "reachable_typed_dicts", "shape_problems", "typed_dict_fields",
]
