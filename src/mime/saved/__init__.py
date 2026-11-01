"""The saved form: what makes an experiment reproducible by someone else.

A build computes the document (what the experiment declares, by registered name and version), the resolved record
(what it resolved to), the derived graph and its hash, and the environment. ``save`` writes them with the reference
outputs; a load reads them back without running code, resolves every entry among installed packages, rebuilds, and
makes three checks: the input hash exactly, the environment by report, and the derived values within tolerances.

``pack`` writes a folder as one bundle file to share; ``pull`` gets a bundle, checks it and unpacks it into a new folder.
What may be shared is the document's to say (version 4): the experiment's licence, each third-party file's terms and
each declared permission. A file that may not be redistributed travels as a reference, whatever its size, and
``place_referenced`` puts it in place from the asset cache. ``notice`` and ``card_text`` write the folder's NOTICE and
its card, offline, as ``mime push`` will publish them.
"""

from mime.saved._bundle import BUNDLE_SUFFIX, EXPANSION_LIMIT_BYTES, INLINE_LIMIT_BYTES, pack, unpack
from mime.saved._card import card_text, notice
from mime.saved._document import (
    SCENE_FOLDER,
    UnsavableError,
    document_hash,
    document_of,
    dump_document,
    parse_yaml,
)
from mime.saved._folder import (
    DEFAULT_STEPS,
    DOCUMENT_FILE,
    LOCK_FILE,
    REFERENCE_FILE,
    LoadRefused,
    LoadReport,
    SavedForm,
    VerifyReport,
    check_load,
    environment_differences,
    read_document,
    read_folder,
    scene_file_refusals,
    stage_change_refusals,
    verify_history,
    write_folder,
)
from mime.saved._pull import ASSET_CONFIRM_BYTES, pull
from mime.saved._reconstruct import declarations_from
from mime.saved._record import SavedRecord, environment, record_of
from mime.saved._resolve import ENTRY_POINT_GROUP, namespace_providers
from mime.saved._terms import FileTerms, FolderTerms, Placed, place_referenced
from mime.saved._schema import (
    DOCUMENT_SCHEMA,
    LOCK_SCHEMA,
    BoundEntryDoc,
    EntryDoc,
    EntryRecord,
    EnvironmentRecord,
    ExperimentDocument,
    Lock,
    MagnetDoc,
    QuantityDoc,
    ReferenceDoc,
    ResolvedRecord,
    SceneDoc,
    StageRecord,
    ThingDoc,
    conforms,
    shape_problems,
)

__all__ = [
    "ASSET_CONFIRM_BYTES",
    "BUNDLE_SUFFIX",
    "DEFAULT_STEPS",
    "DOCUMENT_FILE",
    "DOCUMENT_SCHEMA",
    "ENTRY_POINT_GROUP",
    "EXPANSION_LIMIT_BYTES",
    "INLINE_LIMIT_BYTES",
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "REFERENCE_FILE",
    "SCENE_FOLDER",
    "BoundEntryDoc",
    "EntryDoc",
    "EntryRecord",
    "EnvironmentRecord",
    "ExperimentDocument",
    "FileTerms",
    "FolderTerms",
    "LoadRefused",
    "Placed",
    "LoadReport",
    "Lock",
    "MagnetDoc",
    "QuantityDoc",
    "ReferenceDoc",
    "ResolvedRecord",
    "SavedForm",
    "SavedRecord",
    "SceneDoc",
    "StageRecord",
    "ThingDoc",
    "UnsavableError",
    "VerifyReport",
    "card_text",
    "check_load",
    "conforms",
    "declarations_from",
    "document_hash",
    "document_of",
    "dump_document",
    "environment",
    "environment_differences",
    "namespace_providers",
    "notice",
    "pack",
    "place_referenced",
    "parse_yaml",
    "pull",
    "read_document",
    "read_folder",
    "record_of",
    "scene_file_refusals",
    "stage_change_refusals",
    "shape_problems",
    "unpack",
    "verify_history",
    "write_folder",
]
