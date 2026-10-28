"""Remote assets: files too large for the repository or the wheel, referenced by sha256 and fetched on request.

An :class:`AssetRef` names a file by the sha256 of its bytes, with its size, licence and an ordered tuple of locations.
:func:`resolve` is the one lookup: the hash-keyed cache first, then a file already in the cache directory, then the
primary location through pooch and the backups after it. :func:`fetch` is its command's name; :func:`locate`, which a
build calls, returns the cached file or refuses with ``asset.missing`` and never downloads. See the README for the
cache, the check ``locate`` makes and the variables that move or close it. :func:`folder_assets` reads the references
an experiment folder's lock names.
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
from mime.assets._fetch import fetch, resolve
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
    "AssetRecord",
    "AssetRef",
    "Cached",
    "cache_dir",
    "check",
    "fetch",
    "file_path",
    "folder_assets",
    "is_cached",
    "known",
    "locate",
    "location_problem",
    "offline",
    "reference",
    "size_text",
    "resolve",
]
