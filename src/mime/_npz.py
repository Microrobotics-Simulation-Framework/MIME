"""An NPZ read only as far as the file itself vouches for it.

MIME writes every NPZ with ``np.savez``: each entry stored (not compressed), one ``.npy`` array each. A file handed to MIME
is not always one MIME wrote, and NumPy allocates the array its header names before it reads a byte of it, so a deflated
entry of a few kilobytes can name gigabytes (a nested zip bomb: a 2.5 MB container took 2.3 GB). An entry here is read only
once it is checked to be **stored**, no larger than the whole file, a plain array (never a pickled object), and to hold
exactly the bytes its header declares; anything else is a ``ValueError``, which callers refuse with a code of their own
(an ``OSError``, ``zipfile.BadZipFile`` and ``EOFError`` are the other ways a file is not an NPZ MIME wrote). Where a record
of the file's own hash exists it is checked first (a file that is not the one recorded is never opened); a hash cannot
show that what a file holds is small, which is what the checks here are for."""

from __future__ import annotations

import math
import os
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


def check_entries(archive: zipfile.ZipFile, archive_size: int) -> None:
    """Every entry of ``archive`` (a file of ``archive_size`` bytes) checked as :func:`stored_entry` checks the one it
    reads, reading no array: ``ValueError`` for one that is not an array MIME wrote."""
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
