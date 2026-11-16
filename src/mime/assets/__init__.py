"""Files referenced by hash, wherever they are kept, fetched on request: the library knows how to fetch and check a
file, never what or from where.

**One way to declare a file:** :func:`asset`, its origin decided by its argument's type: a record's name (a string,
dotted like an entry's: MIME's packaged files, a record an installed package ships, or one a package's
:class:`Resolver` answers), a local file (a path), or a whole record (an :class:`AssetRef`). It always gives one
:class:`Asset`, with the record behind it.

An :class:`AssetRef` is one record type for every asset: its sha256, size, licence and locations, and, where they are
known, its credit, role, kind (with a derived file's derivation record by hash), notice, custom licences, version,
citation and the packages of the adapters its locations need. Its tier is the verdict of its licence. A
:class:`Manifest` is a named group of records (a robot's meshes, beside its description). Records come from the
experiment's pins (a loaded lock), the manifest beside a description, installed packages' ``mime.assets`` entry points
(a folder of records, or a resolver) and MIME's own data; :func:`reference` finds one by name, and refuses two shared
sources that disagree.

:func:`resolve` is the one lookup: the hash-keyed cache, then a file placed in the cache directory, then the
locations, this machine's locations file's first, each opened by its :class:`Adapter` (``https``, ``doi:``,
``package:``, ``file://`` mirrors, and any a package or a script registers, :func:`register_adapter`), the stream held
to the record's size and hash, redirects followed with a credential bound to its host, retries and resuming the
resolver's own; :func:`approved` asks a batch's questions once. :func:`check` re-hashes a cached file in full. See the
README.
"""

from mime.assets._acceptances import Acceptance
from mime.assets._adapters import (
    Adapter,
    AdapterError,
    Credential,
    LoginRefused,
    NotFound,
    Opened,
    Redirect,
    Request,
    Transient,
    register_adapter,
)
from mime.assets._asset import Asset, asset
from mime.assets._cache import check
from mime.assets._derivation import CheckResult
from mime.assets._fetch import resolve
from mime.assets._gates import approved
from mime.assets._known import reference
from mime.assets._manifest import MANIFEST_FORMAT, Manifest, read_manifest
from mime.assets._reference import AssetRecord, AssetRef
from mime.assets._sources import Resolver

__all__ = [
    "Acceptance",
    "Adapter",
    "AdapterError",
    "Asset",
    "AssetRecord",
    "AssetRef",
    "CheckResult",
    "Credential",
    "LoginRefused",
    "MANIFEST_FORMAT",
    "Manifest",
    "NotFound",
    "Opened",
    "Redirect",
    "Request",
    "Resolver",
    "Transient",
    "approved",
    "asset",
    "check",
    "read_manifest",
    "reference",
    "register_adapter",
    "resolve",
]
