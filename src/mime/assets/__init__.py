"""Remote assets: files too large for the repository or the wheel, referenced by sha256 and fetched on request.

An :class:`AssetRef` names a file by the sha256 of its bytes, with its size, licence and an ordered tuple of locations.
:func:`resolve` is the one lookup: the hash-keyed cache first, then a file already in the cache directory, then the
primary location through pooch and the backups after it. :func:`fetch` is its command's name; :func:`locate`, which a
build calls, returns the cached file or refuses with ``asset.missing`` and never downloads. :func:`retrieve` stores a
file whose hash is not yet known (a bundle named by its DOI). See the README for the
cache, the check ``locate`` makes and the variables that move or close it. :func:`folder_assets` reads the references
an experiment folder's lock names. A derived asset's derivation record (:class:`DerivationRecord`: its builder, inputs
by hash, parameters, checks and validity domain) is written by :func:`derive`, and :func:`stale_inputs` says which of
its inputs have since changed.
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
    offline,
    size_text,
)
from mime.assets._commands import CHECK, FETCH, GROUP, LIST
from mime.assets._derivation import (
    Builder,
    CheckResult,
    DerivationInput,
    DerivationRecord,
    Json,
    Produced,
    derivation_of,
    derive,
    shipped_derivations,
    stale_inputs,
    working_state,
)
from mime.assets._fetch import RETRIEVE_LIMIT_BYTES, fetch, resolve, retrieve, under_test_prefix
from mime.assets._known import known, reference
from mime.assets._lock import folder_assets
from mime.assets._reference import AssetRecord, AssetRef, location_problem

__all__ = [
    "ASSETS_DIR_VARIABLE",
    "CHECK",
    "FETCH",
    "GROUP",
    "LIST",
    "OFFLINE_VARIABLE",
    "RETRIEVE_LIMIT_BYTES",
    "AssetRecord",
    "AssetRef",
    "Builder",
    "Cached",
    "CheckResult",
    "DerivationInput",
    "DerivationRecord",
    "Json",
    "Produced",
    "cache_dir",
    "check",
    "derivation_of",
    "derive",
    "fetch",
    "file_path",
    "folder_assets",
    "is_cached",
    "known",
    "locate",
    "location_problem",
    "under_test_prefix",
    "offline",
    "reference",
    "resolve",
    "retrieve",
    "shipped_derivations",
    "size_text",
    "stale_inputs",
    "working_state",
]
