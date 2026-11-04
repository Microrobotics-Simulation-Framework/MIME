"""Files referenced by hash, wherever they are kept, fetched on request: the library knows how to fetch and check a
file, never what or from where.

An :class:`AssetRef` is one record type for every asset: its sha256, size, licence and locations, and, where they are
known, its credit, role, kind (with a derived file's derivation record by hash), notice, custom licences, version,
citation and the packages of the adapters its locations need. Its tier is the verdict of its licence. A
:class:`Manifest` is a named group of records (a robot's meshes, beside its description). Records come from the
experiment's pins (:func:`pinned`; a loaded lock), the manifest beside a description, installed packages' ``mime.assets``
entry points and MIME's own data; :func:`reference` finds one by name, and refuses two shared sources that disagree.

:func:`resolve` is the one lookup: the hash-keyed cache, then a file placed in the cache directory, then the
locations, this machine's locations file's first (:func:`read_locations`), each opened by its :class:`Adapter`
(``https``, ``doi:``, ``package:``, ``file://`` mirrors, and any a package or a script registers), the stream held to
the record's size and hash, redirects followed with a credential bound to its host, retries and resuming the
resolver's own. :func:`fetch` is its command's name; :func:`locate`, which a build calls, never downloads.
:func:`retrieve` stores a file whose hash is not yet known (a bundle named by its DOI). :func:`probe` looks at a
location without downloading (``mime assets sources``), and :func:`audit` finds files that may not be redistributed in
an archive or a tree (``mime assets audit``). A derived asset's derivation record (:class:`DerivationRecord`) is
written by :func:`derive`; :func:`stale_inputs` says which of its inputs have since changed, and
:func:`derived_terms` its terms, the strictest of its inputs'. :func:`record_terms` is what every exit judges a record
by: its licence, and for a derived record the strictest of that and its inputs', its derivation record read by hash
(:func:`derivation_by_hash`), failing closed. See the README.
"""

from mime.assets._cache import (
    ASSETS_DIR_VARIABLE,
    OFFLINE_VARIABLE,
    Cached,
    cache_dir,
    check,
    file_path,
    is_cached,
    locate,
    named_file,
    named_path,
    offline,
    size_text,
)
from mime.assets._adapters import (
    ADAPTER_GROUP,
    SOURCE_GROUP,
    Adapter,
    AdapterError,
    Credential,
    LoginRefused,
    NotFound,
    Opened,
    Redirect,
    Request,
    Transient,
    adapter_for,
    adapter_package,
    register_adapter,
    registered_schemes,
    stored_credential,
    with_adapters,
)
from mime.assets._audit import AuditReport, Finding, KnownBytes, audit, container_of, known_bytes, mesh_format
from mime.assets._availability import assets_of, sources_text
from mime.assets._commands import AUDIT, CHECK, FETCH, GROUP, LIST, SOURCES
from mime.assets._derivation import (
    Builder,
    CheckResult,
    DerivationInput,
    DerivationRecord,
    Json,
    Produced,
    by_sha256,
    derivation_by_hash,
    derivation_of,
    derive,
    derived_terms,
    derived_verdicts,
    read_derivation,
    record_terms,
    shipped_derivations,
    stale_inputs,
    working_state,
)
from mime.assets._fetch import (
    RETRIEVE_LIMIT_BYTES,
    Probe,
    announce_downloads,
    credentials_from,
    downloads_doing,
    fetch,
    probe,
    resolve,
    retrieve,
    tried_locations,
    under_test_prefix,
    watching_downloads,
)
from mime.assets._known import known, reference
from mime.assets._locations import CA_BUNDLE_VARIABLE, LOCATIONS_VARIABLE, Locations, locations_path, read_locations
from mime.assets._lock import folder_assets
from mime.assets._manifest import (
    MANIFEST_FORMAT,
    Manifest,
    ManifestFile,
    beside,
    mesh_path,
    read_manifest,
    read_manifest_data,
)
from mime.assets._reference import KINDS, AssetRecord, AssetRef, Kind, location_problem
from mime.assets._sources import active_pins, conflicts, pinned, shared_manifests

__all__ = [
    "ADAPTER_GROUP",
    "ASSETS_DIR_VARIABLE",
    "AUDIT",
    "CA_BUNDLE_VARIABLE",
    "CHECK",
    "FETCH",
    "GROUP",
    "KINDS",
    "LIST",
    "LOCATIONS_VARIABLE",
    "MANIFEST_FORMAT",
    "OFFLINE_VARIABLE",
    "RETRIEVE_LIMIT_BYTES",
    "SOURCES",
    "SOURCE_GROUP",
    "Adapter",
    "AdapterError",
    "AssetRecord",
    "AssetRef",
    "AuditReport",
    "Builder",
    "Cached",
    "CheckResult",
    "Credential",
    "DerivationInput",
    "DerivationRecord",
    "Finding",
    "Json",
    "Kind",
    "KnownBytes",
    "Locations",
    "LoginRefused",
    "Manifest",
    "ManifestFile",
    "NotFound",
    "Opened",
    "Probe",
    "Produced",
    "Redirect",
    "Request",
    "Transient",
    "active_pins",
    "adapter_for",
    "adapter_package",
    "announce_downloads",
    "assets_of",
    "audit",
    "beside",
    "by_sha256",
    "cache_dir",
    "check",
    "conflicts",
    "container_of",
    "credentials_from",
    "derivation_by_hash",
    "derivation_of",
    "derive",
    "derived_terms",
    "derived_verdicts",
    "downloads_doing",
    "fetch",
    "file_path",
    "folder_assets",
    "is_cached",
    "known",
    "known_bytes",
    "locate",
    "location_problem",
    "locations_path",
    "mesh_format",
    "mesh_path",
    "named_file",
    "named_path",
    "offline",
    "pinned",
    "probe",
    "read_derivation",
    "read_locations",
    "read_manifest",
    "read_manifest_data",
    "record_terms",
    "reference",
    "register_adapter",
    "registered_schemes",
    "resolve",
    "retrieve",
    "shared_manifests",
    "shipped_derivations",
    "size_text",
    "sources_text",
    "stale_inputs",
    "stored_credential",
    "tried_locations",
    "under_test_prefix",
    "watching_downloads",
    "with_adapters",
    "working_state",
]
