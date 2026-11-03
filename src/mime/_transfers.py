"""What ``mime push`` and ``mime pull`` tell a terminal's display about the files they move.

A push uploads and a pull downloads while the person waits, and the minutes between one file and the next are not
silent: the service is asked a second apart, a file is committed and its md5 checked, a bundle is hashed and unpacked.
:class:`Transfers` is the seam through which the library says so, in four calls, none of which changes what is sent,
received or checked. The default is no seam at all (``None``): nothing is told, and nothing is drawn.

A display that is told is the terminal's alone (:class:`mime.cli._display.TransferDisplay`), and it must not make the
transfer fail: the library does not catch what a display raises (it is read, inside a request, as a dropped
connection), so a display catches its own."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Transfers(Protocol):
    """Told, in order, what is moving and what is under way. A file is named by its key (a record's file name, an
    asset's name); the size is the bytes it will have where that is known."""

    def files(self, files: Sequence[tuple[str, int | None]]) -> None:
        """The files about to move, with their sizes, before the first of them: how many there are to count to."""

    def sending(self, key: str, done: int, total: int | None) -> None:
        """How much of the file ``key`` has gone (or come): about every 128 kB, and at its start and its end."""

    def sent(self, key: str) -> None:
        """The file ``key`` is whole and checked: the service holds it and states its md5, or the download's hash is the
        one named."""

    def doing(self, words: str) -> None:
        """What is under way when no bytes are moving ("waiting between requests", "checking the md5"), in words a person
        reads; the next call replaces it."""
