"""An NPZ read only as far as the file itself vouches for it.

MIME writes every NPZ with ``np.savez``: each entry stored (not compressed), one ``.npy`` array each. A file handed to MIME
is not always one MIME wrote, and NumPy allocates the array its header names before it reads a byte of it, so a deflated
entry of a few kilobytes can name gigabytes (a nested zip bomb: a 2.5 MB container took 2.3 GB). An entry here is read only
once it is checked to be **stored**, no larger than the whole file, a plain array (never a pickled object), and to hold
exactly the bytes its header declares; anything else is a ``ValueError``, which callers refuse with a code of their own
(an ``OSError``, ``zipfile.BadZipFile`` and ``EOFError`` are the other ways a file is not an NPZ MIME wrote). Where a record
of the file's own hash exists it is checked first (a file that is not the one recorded is never opened); a hash cannot
show that what a file holds is small, which is what the checks here are for. A saved folder's lock records its arrays'
hash, not its file's, so no file hash can be checked before the parse there: the bound is the defence.

The layout of the zip is checked too, from its central directory and before any entry is opened: no entry's local header
and data overlap another's, each lies within the file (before its central directory), and no name is used twice. The
standard library refuses overlapped entries since Python 3.12.2 (CVE-2024-0450); 3.12.0 and 3.12.1, which this package's
``requires-python`` allows, do not, and the check here does not rely on it.

What this does not cover: ``_Npz`` in ``mime.results`` checks every entry once, when it opens a run's NPZ, and reads each
later entry through :func:`stored_entry` (which checks that entry's header against its own size, so the allocation stays
bounded), but it does not check the whole layout again. A file swapped for another after it was opened is read with only
that per-entry check; swapping it needs write access to the folder, which is a local attacker's, and is not defended."""

from __future__ import annotations

import math
import os
import struct
import zipfile
from types import TracebackType
from typing import IO, Any

import numpy as np
from numpy.typing import NDArray


def _declared(file: IO[bytes], info: zipfile.ZipInfo, archive_size: int) -> int:
    """The bytes the array of the entry ``info``, open as ``file`` at its start, declares, once they are checked to be
    those the entry holds (``ValueError`` otherwise)."""
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"the entry {info.filename} is compressed, and an NPZ MIME writes is not")
    if info.file_size > archive_size:
        raise ValueError(f"the entry {info.filename} says it holds {info.file_size} bytes, more than the whole file")
    version = np.lib.format.read_magic(file)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(file)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(file)
    else:
        raise ValueError(f"the entry {info.filename} is an array of NumPy format {version[0]}.{version[1]}")
    count = math.prod(shape)
    if dtype.hasobject or any(extent < 0 for extent in shape) or (dtype.itemsize == 0 and count > 0):
        raise ValueError(f"the entry {info.filename} is not an array of plain numbers or text")
    declared, held = count * dtype.itemsize, info.file_size - file.tell()
    if declared != held:
        raise ValueError(f"the entry {info.filename} declares an array of {declared} bytes and holds {held}")
    return declared


_LOCAL_HEADER = struct.Struct("<4s5H3I2H")  # the fixed 30 bytes of a zip local file header (PK\x03\x04, ...)
_LOCAL_SIGNATURE = b"PK\x03\x04"


def _layout(archive: zipfile.ZipFile, archive_size: int) -> None:
    """The zip's layout from its central directory, checked before any entry is opened (``ValueError``): no name twice,
    each entry's local header where its record says, and each entry's span (its local header, then its data) within the file
    and before the central directory, overlapping no other entry's."""
    names = [info.filename for info in archive.infolist()]
    if len(set(names)) != len(names):
        raise ValueError(f"the entry {sorted(name for name in set(names) if names.count(name) > 1)[0]} is in the file twice")
    file = archive.fp
    if file is None:
        raise ValueError("the file is closed")
    limit = min(archive_size, getattr(archive, "start_dir", archive_size))
    spans: list[tuple[int, int, str]] = []
    for info in archive.infolist():
        file.seek(info.header_offset)
        fixed = file.read(_LOCAL_HEADER.size)
        if len(fixed) != _LOCAL_HEADER.size or fixed[:4] != _LOCAL_SIGNATURE:
            raise ValueError(f"the entry {info.filename} has no local header where the file's directory says it does")
        name_length, extra_length = _LOCAL_HEADER.unpack(fixed)[-2:]
        start = info.header_offset + _LOCAL_HEADER.size + name_length + extra_length
        end = start + info.compress_size
        if end > limit:
            raise ValueError(f"the entry {info.filename} runs past the end of the file's data ({end} of {limit} bytes)")
        spans.append((info.header_offset, end, info.filename))
    spans.sort()
    for (_, earlier_end, earlier), (later_start, _, later) in zip(spans, spans[1:], strict=False):
        if earlier_end > later_start:
            raise ValueError(f"the entries {earlier} and {later} overlap in the file")


def check_entries(archive: zipfile.ZipFile, archive_size: int) -> None:
    """The zip's layout (:func:`_layout`), then every entry of ``archive`` (a file of ``archive_size`` bytes) checked as
    :func:`stored_entry` checks the one it reads, reading no array: ``ValueError`` for one that is not an array MIME wrote."""
    _layout(archive, archive_size)
    for info in archive.infolist():
        if not info.filename.endswith(".npy"):
            raise ValueError(f"the entry {info.filename} is not an array")
        with archive.open(info) as file:
            _declared(file, info, archive_size)


def stored_entry(archive: zipfile.ZipFile, key: str, archive_size: int) -> NDArray[Any]:
    """The array ``key`` of ``archive`` (a file of ``archive_size`` bytes), allocated only after its header is checked
    against the bytes the entry holds (``ValueError`` otherwise, ``KeyError`` for an entry the file does not hold)."""
    info = archive.getinfo(key + ".npy")
    with archive.open(info) as file:
        _declared(file, info, archive_size)
        file.seek(0)
        return np.asarray(np.lib.format.read_array(file, allow_pickle=False))


class StoredNpz:
    """An NPZ file opened as ``np.load`` opens one (``files``, one array by key, a context manager), with every entry
    checked when it is opened (:func:`check_entries`) and each read through :func:`stored_entry`."""

    def __init__(self, path: str | os.PathLike[str], *, check: bool = True) -> None:
        """Opened; with ``check=False`` the entries are not all checked now, only each when it is read (for a file whose
        every entry was checked when it was first opened)."""
        self.path = os.fspath(path)
        self._size = os.stat(self.path).st_size
        self._zip = zipfile.ZipFile(self.path)
        try:
            if check:
                check_entries(self._zip, self._size)
        except BaseException:
            self._zip.close()
            raise
        self.files = [name.removesuffix(".npy") for name in self._zip.namelist()]

    def __getitem__(self, key: str) -> NDArray[Any]:
        return stored_entry(self._zip, key, self._size)

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> StoredNpz:
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None,
                 trace: TracebackType | None) -> None:
        self.close()


__all__ = ["StoredNpz", "check_entries", "stored_entry"]
