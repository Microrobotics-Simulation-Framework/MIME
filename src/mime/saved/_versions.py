"""The saved form before version 4, read under the current schema and held to its own lock.

**Version 1** (``mime.document/1``, ``mime.lock/1``) is the form before phase 7l-a. A document of it gains what version
2 added, with the values a build before profiles had: no setting's source given apart from its value, and the profile
**exploratory**. That is the profile it was built under in the sense that matters for loading it: exploratory accepts
every build version 1 accepted (version 1 had no pending reason, and exploratory's other rules only waive what research
would ask for, never refuse more), while research would refuse a version 1 build that relied on a default MIME now
classes as numerical (an arm tracker's tuned gains). Its outputs then carry the stamp of whatever is pending, which for
a version 1 document is only those numerical defaults, and ``mime pending`` lists what sealing it needs.

**Version 2** (``mime.document/2``, ``mime.lock/2``) is the form before the drag's rung ``bem`` was renamed ``mrs``
(the method is the method of regularised Stokeslets, not a boundary-element method). Version 3 changed three things a
hash saw: the rung's name in the document and in the resolved record, the drag node's type in the derived graph
(``BEMDragNode``, now ``MRSDragNode``), and the possessive in the sources of the values declared in the experiment,
which a message built from a quoted name had written ``body 'umr''s`` (now ``body umr's``). A version 2 document that
names ``bem`` is read as naming ``mrs``, and the load says so.

**Version 3** (``mime.document/3``, ``mime.lock/3``) is the form before licences and terms (designs #198, #199).
Version 4 adds four keys to the document (``licence``, ``licences``, ``terms``, ``permissions``) and one to the lock
(``referenced_files``), each written only where something is declared; nothing else changed. A document of an earlier
version declares no licence, no terms and no permission, and is read so: every file of it is its author's own, carried
by ``pack``, with no licence declared. One of an earlier version that holds a key version 4 added is refused, never
read as declaring it.

A previous lock's input hash was taken over the previous document and resolved record, and its derived hash over the
previous derived graph, so a load compares the rebuilt experiment in the previous form: the rebuilt document, resolved
record and derived graph are written back into the form the lock was written in (the keys version 4 added taken off;
the keys version 2 added taken off for version 1; the rung, the node type and the possessive written as version 2
wrote them, for versions 1 and 2) and hashed there. Its schema check is the current one, run on the upgraded form, so
the two cannot drift apart. A version 2 source whose name is not an identifier (a material named with spaces or
quotes) was quoted differently, and is not written back: such a document's input hash is refused, and saving it again
from the script writes the current version.

**Version 5** (``mime.document/5``, ``mime.lock/5``) adds one key to the document, ``run.duration``, the length of a run
of the experiment, written only where the experiment declares one. Nothing else changed. A document of an earlier version
declares no duration, and ``mime run`` on its folder needs ``--duration``, as before. One of an earlier version that holds
the key is refused, never read as declaring it. The lock holds no new key; its schema moves with the document's, as at
every version.

**Version 6** (``mime.document/6``, ``mime.lock/6``) adds to the lock alone: an asset record holds, beside its five
keys, each one it states (credit, role, kind and derivation, notice, custom licences, version, citation, the packages
that provide its locations' adapters), and the lock's ``visuals`` holds the manifests of the meshes the stage draws.
Both are outside the input hash. A lock of an earlier version that holds one of them is refused, never read as
declaring it. The document holds no new key; its schema moves with the lock's.

**Version 7** (``mime.document/7``, ``mime.lock/7``) adds one key to the document, ``based_on``, the saved experiment
this one was made from (``Experiment.based_on``): its name, its input hash, the MIME version that saved it and its DOI
where it was published, written only where the experiment was made from one. It is part of the document, and so of the
document and input hashes. Nothing else changed. A document of an earlier version was made from none; one of an earlier
version that holds the key is refused, never read as declaring it. The lock holds no new key; its schema moves with the
document's, as at every version.

**Version 8** (``mime.document/8``, ``mime.lock/8``) adds two keys to the lock, both outside the input hash. The
reference's ``program`` names the program its outputs were run by where it is not the history scan of whole control
periods: ``one-step``, every step its own compiled scan, the program a run chunked by one step runs, from whose graph
the save took them; ``verify`` reruns that program, since two programs of the same steps can differ in their last bits.
The environment's ``usable_cpus`` is how many CPUs the build's process could use, by which XLA's CPU backend partitions
a program's work, and its ``blas_threads`` is how many threads the BLAS under JAX's CPU LAPACK kernels uses (SciPy's
OpenBLAS pool), which splits a routine's sums: a load under another count of either is in another environment. A lock
of an earlier version names no program, its reference outputs being the history scan's, and records neither count, and
a load compares its environment without them, as before; one of an earlier version that holds any of the keys is
refused, never read as declaring it. The document holds
no new key; its schema moves with the lock's.

**Version 9** (``mime.document/9``, ``mime.lock/9``) adds no key. It changes what the resolved record holds for a
body's ``inertia`` that its parts give: the full tensor about its centre of mass in the body's frame, nine numbers
row-major, where an earlier form held its diagonal, three moments (the products of inertia dropped, and in exploratory
stamped pending; ``rigid_inertial`` 0.4.0 carries them). A declared inertia keeps the form it was declared in, and a
declaration in place of the parts' value names the parts' nine numbers in its source, where an earlier form named the
three and said where the products were dropped. A load of an earlier form writes the rebuilt record back so: the
parts' tensor as its diagonal, and such a source as the three numbers it named, with the words where any product
was above what 0.3.0 took as none. The folder's document names ``rigid_inertial`` 0.3.0 or 0.2.0, kept registered,
which builds its node from the diagonal as before; they refuse a tensor whose products they would have dropped, so a
folder that rested on exploratory's waiver of them is refused at load, by the version's own words, and not
reproduced. The document's and the lock's schemas move with the form.

**Version 10** (``mime.document/10``, ``mime.lock/10``) adds two keys below the lock's ``reference``:
``resolution``, each saved field's resolution by block (its largest divergence over the reference steps run again,
by the same program, from a start one unit in the last place off; ``null`` where it was not finite), and
``resolution_factor``, the floor's factor over it. It changes what ``reference.rtol`` and ``reference.atol`` mean where
``verify`` claims no bits (in another environment, or from a reference made on a GPU with XLA's deterministic mode
off): every field is split into blocks by its declared units, and a block is within tolerance when its largest
difference at every step is at most ``max(rtol`` times its largest magnitude, ``resolution_factor`` times its
resolution``)`` plus ``atol`` (``mime.saved._scales``); the load's comparison of the derived graph in another
environment holds each list of numbers and each static array to its own largest magnitude. The tenth form writes
``rtol`` for its precision, 1e-5 in float32 and 2e-14 in float64, and ``atol`` 0.0, and a folder keeps the tolerance
and the factor it was written with. A lock of an earlier form is judged as it was, element by element, ``|a - b| <=
atol + rtol |b|`` with its own numbers (1e-6 and 1e-9): in ``verify``, in ``verify --run`` and in the load's comparison
of the derived graph; one that holds either key is left for the schema check to refuse. Nothing it holds is written
back; its schema moves with the form.

**Version 11** (``mime.document/11``, ``mime.lock/11``) writes a medium's fields under its own key ``fields``, by
role (``fields: {flow: ...}``), as an apparatus's are, where version 10 and earlier wrote a medium's flow as a dynamics
aspect (``dynamics: {flow: ...}``); a medium's ``dynamics`` is no longer written. Nothing else changed. A document of an
earlier version is read with its medium's ``dynamics.flow`` as ``fields.flow``, and written back so for its lock's
hashes; one of an earlier version whose medium holds ``fields`` is left for the schema check to refuse. The lock holds
no new key; its schema moves with the document's.

**Version 12** (``mime.document/12``, ``mime.lock/12``) names the FVM's setting ``solve_report_rows``, which version 11
and earlier named ``report_rows``: in a medium's flow settings, in the flow's resolved entry and its default, and among
the flow node's parameters in the derived graph. Nothing else changed. A document of an earlier version is read with
the old name as the new, and the rebuilt document, resolved record and derived graph are written back with the old name
for its lock's hashes. No key is added; the schemas move with the form."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

from mime.build import BuildRefused, Refusal, RefusalClass
from mime.saved._schema import DOCUMENT_SCHEMA, LOCK_SCHEMA

#: The saved form this MIME writes.
CURRENT = 12
DOCUMENT_SCHEMAS: Mapping[int, str] = {1: "mime.document/1", 2: "mime.document/2", 3: "mime.document/3",
                                       4: "mime.document/4", 5: "mime.document/5", 6: "mime.document/6",
                                       7: "mime.document/7", 8: "mime.document/8", 9: "mime.document/9",
                                       10: "mime.document/10", 11: "mime.document/11", CURRENT: DOCUMENT_SCHEMA}
LOCK_SCHEMAS: Mapping[int, str] = {1: "mime.lock/1", 2: "mime.lock/2", 3: "mime.lock/3", 4: "mime.lock/4",
                                   5: "mime.lock/5", 6: "mime.lock/6", 7: "mime.lock/7", 8: "mime.lock/8",
                                   9: "mime.lock/9", 10: "mime.lock/10", 11: "mime.lock/11", CURRENT: LOCK_SCHEMA}
#: The version that added licences and terms, the version that added the run's duration, the version whose lock
#: holds an asset's full record and the manifests of the meshes its stage draws, the version that added the
#: experiment an experiment was made from, the version whose lock names the program the reference outputs were
#: run by and records the CPUs the build could use, the version whose resolved record holds the full inertia
#: tensor a body's parts give, and the version whose reference tolerance is over each field's blocks by unit, with
#: their measured resolution (``mime.saved._scales.SCALED_FROM``).
TERMS_ADDED = 4
DURATION_ADDED = 5
RECORDS_ADDED = 6
BASED_ON_ADDED = 7
PROGRAM_ADDED = 8
INERTIA_TENSOR_ADDED = 9
SOURCE_SCALE_ADDED = 10
#: The version that writes a medium's fields under its own key, by role.
MEDIUM_FIELDS_ADDED = 11
#: A medium's key version 11 added, and the aspect of its ``dynamics`` an earlier version wrote its flow as.
FIELDS_KEY, FLOW_ASPECT = "fields", "flow"
#: The version that renamed the FVM's setting: a form before it names ``solve_report_rows`` ``report_rows``.
ROWS_RENAMED = 12
OLD_ROWS, NEW_ROWS = "report_rows", "solve_report_rows"
_FVM = "fvm"
#: The version that renamed the drag's rung: a form before it names the rung ``bem``.
RUNG_RENAMED = 3
#: The keys version 4 added, to the document and to the lock: an earlier form holds none of them, and declares no
#: licence, no terms and no permission.
TERMS_KEYS: tuple[str, ...] = ("licence", "licences", "terms", "permissions")
LOCK_TERMS_KEY = "referenced_files"
#: The key version 5 added, below the document's ``run``.
DURATION_KEY = "duration"
#: The lock's key version 6 added (the manifests of the meshes the stage draws), and the keys it added to the lock's
#: asset records (each written only where a record states it).
LOCK_VISUALS_KEY = "visuals"
RECORD_KEYS_ADDED: tuple[str, ...] = ("credit", "role", "kind", "derivation", "notice", "licences", "version",
                                      "citation", "adapters")
#: The document's key version 7 added: the experiment this one was made from.
BASED_ON_KEY = "based_on"
#: The lock's keys version 8 added: the program below its ``reference``, the two counts below its ``environment``.
PROGRAM_KEY = "program"
USABLE_CPUS_KEY = "usable_cpus"
BLAS_THREADS_KEY = "blas_threads"
COUNT_KEYS = (USABLE_CPUS_KEY, BLAS_THREADS_KEY)
#: The lock's keys version 10 added, below its ``reference``: each field's resolution by block, and the floor's factor.
RESOLUTION_KEYS = ("resolution", "resolution_factor")

#: The profile a version 1 document is read under, and the keys version 2 added, with the values a version 1 form holds.
PREVIOUS_PROFILE = "exploratory"
_DOCUMENT_ADDED: Mapping[str, object] = {"profile": PREVIOUS_PROFILE, "sources": {}}
_RESOLVED_ADDED = "profile"
_PREVIOUS_RESOLVED_PROFILE: Mapping[str, object] = {"profile": PREVIOUS_PROFILE, "pending": [], "waivers": [],
                                                    "defaults": {}, "sources": {}, "listed": [], "changes": {}}

#: The rung of the drag as version 2 and earlier named it, and as version 3 on names it; the drag node's type likewise.
OLD_RUNG, NEW_RUNG = "bem", "mrs"
OLD_NODE, NEW_NODE = "BEMDragNode", "MRSDragNode"
_DRAG = "drag"
#: A body's inertia, the value version 9 resolves to the full tensor where its parts give it, and how the resolved
#: record says it came from its parts (``mime.materials.DERIVED_FROM_PARTS``).
_INERTIA = ".inertia"
_FROM_PARTS = "derived from its parts"
#: A declaration's source naming the inertia the parts would give in its place: its numbers, then its unit.
_IN_PLACE_OF_PARTS = re.compile(r"(in place of the value its parts give, )\(([^()]*)\)( kg m\^2)$")
#: The products of inertia 0.3.0's part resolver took as none, as a share of the largest moment
#: (``mime.dynamics.body.PRODUCTS_TAKEN_AS_NONE``): above it, an earlier form's source said they were dropped.
_PRODUCTS_TAKEN_AS_NONE = 1e-12
_PRODUCTS_DROPPED = ", its products of inertia dropped"
#: ``body umr's`` as version 3 writes a possessive in a value's source, and ``body 'umr''s`` as version 2 wrote it. Only
#: a name that is an identifier is written back: ``repr`` quotes any other name differently.
_PLAIN_POSSESSIVE = re.compile(r"\b(body|material) ([A-Za-z_][A-Za-z0-9_]*)'s\b")


def version_of(schema: object, schemas: Mapping[int, str]) -> int | None:
    """The version whose schema string is ``schema``, or ``None``."""
    return next((version for version, text in schemas.items() if text == schema), None)


#: A form's id of one family: ``mime.document/4`` is ``mime.document`` at version 4 (no sign, no leading zero).
_FORM_ID = re.compile(r"mime\.(?P<family>[a-z_]+)/(?P<version>[1-9][0-9]*)")
#: The longest version number compared as a number: a longer one is newer than any form MIME has written.
_COMPARABLE_DIGITS = 9


def newer_form_refusal(found: object, family: str, reads: Sequence[str], where: str) -> Refusal | None:
    """The refusal ``load.newer-form`` when ``found`` (the schema or format id a file states) is of ``family``
    (``document``, for ``mime.document/N``) at a version above the newest of ``reads`` (the ids this MIME reads), or
    ``None``: an id of another family, a version this MIME reads and any other text are for the file's own check to
    judge. ``where`` names the file."""
    from mime import __version__

    match = _FORM_ID.fullmatch(found) if isinstance(found, str) else None
    if match is None or match["family"] != family:
        return None
    versions = [int(text.rpartition("/")[2]) for text in reads]
    digits = match["version"]
    if len(digits) <= _COMPARABLE_DIGITS and int(digits) <= max(versions):
        return None
    return Refusal(
        "load.newer-form", RefusalClass.USER,
        f"{where} was written by a newer MIME: its form is {found}, and this MIME ({__version__}) reads "
        f"{', '.join(reads)}",
        "install the MIME that wrote it, a later one (`pip install -U mime-engine`), then load it again")


#: The refusal of a kept ``rigid_inertial`` (0.3.0 or 0.2.0) given a tensor with products of inertia
#: (``mime.dynamics.body``): at a load it means the folder rested on exploratory's waiver of them.
PRODUCTS_REFUSAL = "value.products-of-inertia"


def products_waiver_refusals(refused: BuildRefused) -> BuildRefused:
    """``refused`` (the build of a loaded folder) with each refusal of products of inertia worded for a folder: the
    kept version's own words say how to build a body, and a folder is not edited as a body. The folder rested on
    exploratory's waiver of the products, which went with ``rigid_inertial`` 0.4.0, so it is re-created from its script
    on the current version."""
    if all(each.code != PRODUCTS_REFUSAL for each in refused.refusals):
        return refused
    found = [replace(
        each,
        message=f"{each.message}. This folder names that version, which dropped the products only under exploratory's "
                f"waiver, and this version no longer gives the waiver: it is refused at load, not reproduced without "
                f"them",
        action="re-create the experiment from its script on this version, whose body carries the full tensor "
               "(Body(..., dynamics=RigidInertial())), and save it again; a folder cannot be edited into one")
        if each.code == PRODUCTS_REFUSAL else each for each in refused.refusals]
    return type(refused)(found, refused.report)


def _renamed_rungs(document: dict[str, object]) -> tuple[dict[str, object], tuple[str, ...]]:
    """``document`` with each drag that names the old rung naming the new one, and the ids of those drags."""
    found: list[str] = []
    interactions = document.get("interactions")
    if not isinstance(interactions, list):
        return document, ()
    changed: list[object] = []
    for item in cast("list[object]", interactions):
        if isinstance(item, dict):
            entry = cast("dict[str, object]", item)
            settings = entry.get("settings")
            if entry.get("name") == _DRAG and isinstance(settings, dict) and \
                    cast("dict[str, object]", settings).get("rung") == OLD_RUNG:
                item = {**entry, "settings": {**cast("dict[str, object]", settings), "rung": NEW_RUNG}}
                found.append(str(entry.get("id")))
        changed.append(cast(object, item))
    return ({**document, "interactions": changed} if found else document), tuple(found)


def terms_keys_held(data: object) -> list[str]:
    """The keys version 4 added that ``data`` (a document, or a lock) holds."""
    if not isinstance(data, dict):
        return []
    held = cast("dict[str, object]", data)
    return [key for key in (*TERMS_KEYS, LOCK_TERMS_KEY) if key in held]


def duration_held(data: object) -> list[str]:
    """``["run.duration"]`` where ``data`` (a document) holds the key version 5 added, and none otherwise."""
    run = cast("dict[str, object]", data).get("run") if isinstance(data, dict) else None
    return [f"run.{DURATION_KEY}"] if isinstance(run, dict) and DURATION_KEY in cast("dict[str, object]", run) else []


def records_held(data: object) -> list[str]:
    """The keys version 6 added that a lock holds: its ``visuals``, and each key of an asset record beyond the five
    every record has (``assets[<i>].<key>``)."""
    if not isinstance(data, dict):
        return []
    held = cast("dict[str, object]", data)
    found = [LOCK_VISUALS_KEY] if LOCK_VISUALS_KEY in held else []
    listed = held.get("assets")
    for index, record in enumerate(cast("list[object]", listed) if isinstance(listed, list) else []):
        if isinstance(record, dict):
            found += [f"assets[{index}].{key}" for key in RECORD_KEYS_ADDED if key in cast("dict[str, object]", record)]
    return found


def based_on_held(data: object) -> list[str]:
    """``["based_on"]`` where ``data`` (a document) holds the key version 7 added, and none otherwise."""
    return [BASED_ON_KEY] if isinstance(data, dict) and BASED_ON_KEY in cast("dict[str, object]", data) else []


def program_keys_held(data: object) -> list[str]:
    """The keys version 8 added that a lock holds: ``reference.program``, ``environment.usable_cpus`` and
    ``environment.blas_threads``."""
    if not isinstance(data, dict):
        return []
    held = cast("dict[str, object]", data)
    found: list[str] = []
    for parent, key in (("reference", PROGRAM_KEY), *(("environment", count) for count in COUNT_KEYS)):
        below = held.get(parent)
        if isinstance(below, dict) and key in cast("dict[str, object]", below):
            found.append(f"{parent}.{key}")
    return found


def resolution_held(data: object) -> list[str]:
    """The keys version 10 added that a lock holds: ``reference.resolution`` and ``reference.resolution_factor``."""
    below = cast("dict[str, object]", data).get("reference") if isinstance(data, dict) else None
    if not isinstance(below, dict):
        return []
    return [f"reference.{key}" for key in RESOLUTION_KEYS if key in cast("dict[str, object]", below)]


def _media(document: Mapping[str, object]) -> list[tuple[str, dict[str, object]]]:
    """The document's media, by name."""
    things = document.get("things")
    if not isinstance(things, dict):
        return []
    return [(str(name), cast("dict[str, object]", doc)) for name, doc in cast("dict[str, object]", things).items()
            if isinstance(doc, dict) and cast("dict[str, object]", doc).get("kind") == "medium"]


def medium_fields_held(data: object) -> list[str]:
    """The keys version 11 added that a document holds: each medium's ``fields`` (``things.<name>.fields``)."""
    if not isinstance(data, dict):
        return []
    return [f"things.{name}.{FIELDS_KEY}" for name, doc in _media(cast("dict[str, object]", data))
            if FIELDS_KEY in doc]


def _with_media(document: dict[str, object], change: Any) -> dict[str, object]:
    """``document`` with each medium's document replaced by ``change(it)``."""
    things = document.get("things")
    if not isinstance(things, dict):
        return document
    held = dict(cast("dict[str, object]", things))
    for name, doc in _media(document):
        held[name] = change(doc)
    return {**document, "things": held}


def _fields_of_flow(medium: dict[str, object]) -> dict[str, object]:
    """A medium's document as version 11 writes it: an earlier version's ``dynamics.flow`` as ``fields.flow``."""
    dynamics = medium.get("dynamics")
    if not isinstance(dynamics, dict) or FLOW_ASPECT not in dynamics:
        return medium
    rest = {key: value for key, value in cast("dict[str, object]", dynamics).items() if key != FLOW_ASPECT}
    held = {key: value for key, value in medium.items() if key != "dynamics"}
    return {**held, **({"dynamics": rest} if rest else {}),
            FIELDS_KEY: {FLOW_ASPECT: cast("dict[str, object]", dynamics)[FLOW_ASPECT]}}


def _flow_of_fields(medium: dict[str, object]) -> dict[str, object]:
    """A medium's document as an earlier version wrote it: its ``fields.flow`` as ``dynamics.flow``."""
    fields = medium.get(FIELDS_KEY)
    if not isinstance(fields, dict):
        return medium
    held = {key: value for key, value in medium.items() if key != FIELDS_KEY}
    return {**held, "dynamics": {**cast("dict[str, object]", held.get("dynamics", {})),
                                 **cast("dict[str, object]", fields)}}


def _with_setting(entry: object, was: str, now: str) -> object:
    """``entry`` (a medium's flow in the document, or the flow's resolved entry) with the FVM's setting ``was`` named
    ``now``."""
    if not isinstance(entry, dict):
        return entry
    held = cast("dict[str, object]", entry)
    settings = held.get("settings")
    if held.get("name") != _FVM or not isinstance(settings, dict) or was not in settings:
        return held
    renamed = {now if key == was else key: value for key, value in cast("dict[str, object]", settings).items()}
    return {**held, "settings": renamed}


def _with_flow_setting(medium: dict[str, object], was: str, now: str) -> dict[str, object]:
    """A medium's document with its flow's FVM setting ``was`` named ``now``."""
    fields = medium.get(FIELDS_KEY)
    if not isinstance(fields, dict):
        return medium
    held = cast("dict[str, object]", fields)
    if FLOW_ASPECT not in held:
        return medium
    return {**medium, FIELDS_KEY: {**held, FLOW_ASPECT: _with_setting(held[FLOW_ASPECT], was, now)}}


def _rows_read(medium: dict[str, object]) -> dict[str, object]:
    """A medium's document of a form before version 12 as version 12 reads it: the FVM's setting by its new name."""
    return _with_flow_setting(medium, OLD_ROWS, NEW_ROWS)


def _rows_written(medium: dict[str, object]) -> dict[str, object]:
    """A medium's document written back into a form before version 12: the FVM's setting by its old name."""
    return _with_flow_setting(medium, NEW_ROWS, OLD_ROWS)


def added_after(data: object, version: int) -> list[str]:
    """The keys of a document (or, for the lock, of a lock) of ``version`` that a later version added: version 4's
    terms keys in one of an earlier version, version 5's duration, version 7's ``based_on`` and version 11's media's
    ``fields`` in one earlier than each."""
    found = terms_keys_held(data) if version < TERMS_ADDED else []
    found += duration_held(data) if version < DURATION_ADDED else []
    found += medium_fields_held(data) if version < MEDIUM_FIELDS_ADDED else []
    return found + (based_on_held(data) if version < BASED_ON_ADDED else [])


def upgraded_document(data: object) -> tuple[object, int, tuple[str, ...]]:
    """``data`` as the current schema reads it, the version it was written in (the current one for a document that is
    not of an earlier version, which the schema check judges), and the ids of the drags that named the old rung. A
    document of an earlier version that holds a key a later one added (version 2's in version 1, version 4's, 5's, 7's
    or 11's in any earlier) is left as it is, for the schema check to refuse. A medium's flow an earlier version wrote
    as ``dynamics.flow`` is read as ``fields.flow``."""
    if not isinstance(data, dict):
        return data, CURRENT, ()
    held = dict(cast("dict[str, object]", data))
    version = version_of(held.get("schema"), DOCUMENT_SCHEMAS)
    if version is None or version == CURRENT:
        return held, CURRENT, ()
    if added_after(held, version):
        return held, version, ()
    if version == 1:
        if any(key in held for key in _DOCUMENT_ADDED):
            return held, version, ()
        held = {**held, **_DOCUMENT_ADDED}
    renamed: tuple[str, ...] = ()
    if version < RUNG_RENAMED:
        held, renamed = _renamed_rungs(held)
    if version < MEDIUM_FIELDS_ADDED:
        held = _with_media(held, _fields_of_flow)
    if version < ROWS_RENAMED:
        held = _with_media(held, _rows_read)
    return {**held, "schema": DOCUMENT_SCHEMA}, version, renamed


def upgraded_lock(data: object) -> tuple[object, int]:
    """``data`` as the current schema reads it, and the version it was written in. The lock's contents stay in the form
    it was written in: a load compares the rebuilt experiment written back into that form. A lock of an earlier
    version that holds a key a later one added (version 4's ``referenced_files``, version 6's ``visuals`` or a key of
    an asset record beyond the five, version 8's program or count, version 10's resolution) is left as it is, for the
    schema check to refuse."""
    if not isinstance(data, dict):
        return data, CURRENT
    held = dict(cast("dict[str, object]", data))
    version = version_of(held.get("schema"), LOCK_SCHEMAS)
    if version is None or version == CURRENT:
        return held, CURRENT
    if LOCK_TERMS_KEY in held and version < TERMS_ADDED:
        return held, version
    if version < RECORDS_ADDED and records_held(held):
        return held, version
    if version < PROGRAM_ADDED and program_keys_held(held):
        return held, version
    if version < SOURCE_SCALE_ADDED and resolution_held(held):
        return held, version
    resolved = held.get("resolved")
    if version == 1 and isinstance(resolved, dict) and _RESOLVED_ADDED not in resolved:
        held["resolved"] = {**cast("dict[str, object]", resolved), _RESOLVED_ADDED: dict(_PREVIOUS_RESOLVED_PROFILE)}
    held["schema"] = LOCK_SCHEMA
    return held, version


def document_in_form(document: Mapping[str, Any], version: int) -> dict[str, Any]:
    """A document written back into the form of ``version``, as its lock hashed it."""
    held = dict(document)
    if version == CURRENT:
        return held
    if version < ROWS_RENAMED:
        held = _with_media(held, _rows_written)
    if version < MEDIUM_FIELDS_ADDED:
        held = _with_media(held, _flow_of_fields)
    if version < BASED_ON_ADDED:
        held = {key: value for key, value in held.items() if key != BASED_ON_KEY}
    if version < TERMS_ADDED:
        held = {key: value for key, value in held.items() if key not in TERMS_KEYS}
    run = held.get("run")
    if version < DURATION_ADDED and isinstance(run, dict):
        held["run"] = {key: value for key, value in cast("dict[str, Any]", run).items() if key != DURATION_KEY}
    interactions = held.get("interactions")
    if version < RUNG_RENAMED and isinstance(interactions, list):
        held["interactions"] = [_with_rung(item, NEW_RUNG, OLD_RUNG) for item in cast("list[Any]", interactions)]
    if version == 1:
        held = {key: value for key, value in held.items() if key not in _DOCUMENT_ADDED}
    held["schema"] = DOCUMENT_SCHEMAS[version]
    return held


def _with_rung(entry: Any, was: str, now: str) -> Any:
    """``entry`` (a bound entry of the document, or a resolved entry) with the drag's rung ``was`` written ``now``."""
    if not isinstance(entry, dict):
        return entry
    held = cast("dict[str, Any]", entry)
    settings = held.get("settings")
    if held.get("name") == _DRAG and isinstance(settings, dict) and cast("dict[str, Any]", settings).get("rung") == was:
        return {**held, "settings": {**cast("dict[str, Any]", settings), "rung": now}}
    return held


def _quoted_possessive(text: object) -> object:
    return _PLAIN_POSSESSIVE.sub(r"\1 '\2''s", text) if isinstance(text, str) else text


def _value_in_form(record: Any) -> Any:
    """A value's record of the resolved record, its source written as version 2 wrote it."""
    if not isinstance(record, dict):
        return record
    held = cast("dict[str, Any]", record)
    return {**held, "source": _quoted_possessive(held.get("source"))}


def _diagonal(numbers: Sequence[Any]) -> list[Any]:
    return [numbers[0], numbers[4], numbers[8]]


def _source_of_parts_in_form(source: str) -> str:
    """A declared inertia's source as a form before version 9 wrote it: the parts' tensor it names, nine numbers as
    the source prints them, written as their diagonal, with the words an earlier form added where a product was above
    what 0.3.0 took as none (judged on the six digits the source prints, as near that line as they resolve). Any other
    source is left as it is."""
    found = _IN_PLACE_OF_PARTS.search(source)
    numbers = found.group(2).split(", ") if found is not None else []
    if found is None or len(numbers) != 9:
        return source
    try:
        values = [float(text) for text in numbers]
    except ValueError:
        return source
    largest = max(abs(values[0]), abs(values[4]), abs(values[8]))
    across = max(abs(c) for i, c in enumerate(values) if i not in (0, 4, 8))
    dropped = _PRODUCTS_DROPPED if across > _PRODUCTS_TAKEN_AS_NONE * largest else ""
    return (f"{source[:found.start()]}{found.group(1)}({', '.join(_diagonal(numbers))}){found.group(3)}"
            f"{dropped}")


def _inertia_in_form(path: str, record: Any) -> Any:
    """A value's record of the resolved record as a form before version 9 wrote it: a body's inertia its parts give
    as its diagonal, and a declared one's source naming the parts' tensor as :func:`_source_of_parts_in_form` writes
    it."""
    if not path.endswith(_INERTIA) or not isinstance(record, dict):
        return record
    held = cast("dict[str, Any]", record)
    value, source = held.get("value"), held.get("source")
    if source == _FROM_PARTS and isinstance(value, list) and len(cast("list[Any]", value)) == 9:
        return {**held, "value": _diagonal(cast("list[Any]", value))}
    if isinstance(source, str):
        return {**held, "source": _source_of_parts_in_form(source)}
    return held


def _default_path_in_form(path: str, record: object) -> str:
    """A default's path in the profile's record as a form before version 12 named it: the FVM's setting by its old
    name."""
    suffix = f".{NEW_ROWS}"
    entry = cast("dict[str, object]", record).get("entry") if isinstance(record, dict) else None
    if path.endswith(suffix) and isinstance(entry, str) and entry.startswith(f"{_FVM} "):
        return f"{path[:-len(suffix)]}.{OLD_ROWS}"
    return path


def _rows_in_form(resolved: dict[str, Any]) -> dict[str, Any]:
    """A resolved record with the FVM's setting named as a form before version 12 named it: in each flow's entry and
    among the profile's defaults (``<path>.solve_report_rows``)."""
    held = dict(resolved)
    entries = held.get("entries")
    if isinstance(entries, dict):
        held["entries"] = {name: _with_setting(entry, NEW_ROWS, OLD_ROWS)
                           for name, entry in cast("dict[str, object]", entries).items()}
    profile = held.get("profile")
    if isinstance(profile, dict):
        record = cast("dict[str, object]", profile)
        defaults = record.get("defaults")
        if isinstance(defaults, dict):
            renamed = {_default_path_in_form(path, value): value
                       for path, value in cast("dict[str, object]", defaults).items()}
            held["profile"] = {**record, "defaults": renamed}
    return held


def resolved_in_form(resolved: Mapping[str, Any], version: int) -> dict[str, Any]:
    """A resolved record written back into the form of ``version``: before version 12, the FVM's setting as
    ``report_rows``; before version 9, a body's inertia from its parts as its diagonal; before version 3, the rung, the
    possessive and the profile as version 2 and 1 wrote them."""
    held = dict(resolved)
    if version < ROWS_RENAMED:
        held = _rows_in_form(held)
    values = held.get("values")
    if version < INERTIA_TENSOR_ADDED and isinstance(values, dict):
        held["values"] = {path: _inertia_in_form(path, record)
                          for path, record in cast("dict[str, Any]", values).items()}
    if version >= RUNG_RENAMED:
        return held
    entries = held.get("entries")
    if isinstance(entries, dict):
        held["entries"] = {name: _with_rung(entry, NEW_RUNG, OLD_RUNG)
                           for name, entry in cast("dict[str, Any]", entries).items()}
    values = held.get("values")
    if isinstance(values, dict):
        held["values"] = {path: _value_in_form(record) for path, record in cast("dict[str, Any]", values).items()}
    if version == 1:
        held = {key: value for key, value in held.items() if key != _RESOLVED_ADDED}
    return held


def _node_rows_in_form(node: object) -> object:
    """A node of the derived graph with the flow node's parameter named as a form before version 12 named it."""
    if not isinstance(node, dict):
        return node
    held = cast("dict[str, object]", node)
    params = held.get("params")
    if not isinstance(params, dict) or NEW_ROWS not in params:
        return held
    renamed = {OLD_ROWS if key == NEW_ROWS else key: value for key, value in cast("dict[str, object]", params).items()}
    return {**held, "params": renamed}


def physics_in_form(physics: Mapping[str, Any], version: int) -> dict[str, Any]:
    """The derived graph's physics written back into the form of ``version``: the flow node's parameter and the drag
    node's type as they were named."""
    held = dict(physics)
    nodes = held.get("nodes")
    if version < ROWS_RENAMED and isinstance(nodes, list):
        held["nodes"] = [_node_rows_in_form(node) for node in cast("list[Any]", nodes)]
    nodes = held.get("nodes")
    if version < RUNG_RENAMED and isinstance(nodes, list):
        held["nodes"] = [{**cast("dict[str, Any]", node), "type": OLD_NODE}
                         if isinstance(node, dict) and cast("dict[str, Any]", node).get("type") == NEW_NODE else node
                         for node in cast("list[Any]", nodes)]
    return held


__all__ = ["BASED_ON_ADDED", "BASED_ON_KEY", "BLAS_THREADS_KEY", "COUNT_KEYS", "CURRENT", "DOCUMENT_SCHEMAS",
           "DURATION_ADDED", "DURATION_KEY", "FIELDS_KEY", "FLOW_ASPECT", "INERTIA_TENSOR_ADDED", "LOCK_VISUALS_KEY",
           "MEDIUM_FIELDS_ADDED", "RECORDS_ADDED", "medium_fields_held",
           "PRODUCTS_REFUSAL", "RECORD_KEYS_ADDED", "records_held", "LOCK_SCHEMAS", "LOCK_TERMS_KEY", "NEW_RUNG",
           "OLD_ROWS", "NEW_ROWS", "OLD_RUNG", "PREVIOUS_PROFILE", "PROGRAM_ADDED", "PROGRAM_KEY", "RESOLUTION_KEYS",
           "ROWS_RENAMED", "RUNG_RENAMED",
           "SOURCE_SCALE_ADDED", "TERMS_ADDED", "TERMS_KEYS", "USABLE_CPUS_KEY", "added_after", "based_on_held",
           "document_in_form", "duration_held", "newer_form_refusal", "physics_in_form", "products_waiver_refusals",
           "program_keys_held", "resolution_held", "resolved_in_form", "terms_keys_held",
           "upgraded_document", "upgraded_lock", "version_of"]
