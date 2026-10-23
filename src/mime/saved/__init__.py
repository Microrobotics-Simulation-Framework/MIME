"""The saved form: what makes an experiment reproducible by someone else.

A build computes the document (what the experiment declares, by registered name and version), the resolved record
(what it resolved to), the derived graph and its hash, and the environment. ``save`` writes them with the reference
outputs; a load reads them back without running code, resolves every entry among installed packages, rebuilds, and
makes three checks: the input hash exactly, the environment by report, and the derived values within tolerances.
"""

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
    verify_history,
    write_folder,
)
from mime.saved._reconstruct import declarations_from
from mime.saved._record import SavedRecord, environment, record_of
from mime.saved._resolve import ENTRY_POINT_GROUP, namespace_providers
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
    "DEFAULT_STEPS",
    "DOCUMENT_FILE",
    "DOCUMENT_SCHEMA",
    "ENTRY_POINT_GROUP",
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "REFERENCE_FILE",
    "SCENE_FOLDER",
    "BoundEntryDoc",
    "EntryDoc",
    "EntryRecord",
    "EnvironmentRecord",
    "ExperimentDocument",
    "LoadRefused",
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
    "check_load",
    "conforms",
    "declarations_from",
    "document_hash",
    "document_of",
    "dump_document",
    "environment",
    "environment_differences",
    "namespace_providers",
    "parse_yaml",
    "read_document",
    "read_folder",
    "record_of",
    "scene_file_refusals",
    "shape_problems",
    "verify_history",
    "write_folder",
]
