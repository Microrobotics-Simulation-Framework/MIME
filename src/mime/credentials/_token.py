"""The one type a credential is held in: its text and its debugging forms are masked, it cannot be pickled, and the
code that must search for it does so without ever returning it."""

from __future__ import annotations

import base64
import json
from typing import NoReturn, override
from urllib.parse import quote, quote_plus

#: A token at least this long shows its last four characters; a shorter one shows nothing of itself, since four
#: characters of a short token are a large part of it.
SHOWN_FROM_LENGTH = 32
_BASE64 = (base64.b64encode, base64.urlsafe_b64encode)


def _base64_runs(value: bytes) -> set[bytes]:
    """The characters of ``value``'s base64 that do not depend on what stands around it, for each of the three places
    it can start within a base64 group, in both alphabets: ``value`` inside a larger encoded blob is found by one of
    them."""
    runs: set[bytes] = set()
    for encode in _BASE64:
        for shift in range(3):
            encoded = encode(bytes(shift) + value).rstrip(b"=")
            first, last = shift, shift + len(value)
            run = bytes(char for position, char in enumerate(encoded)
                        if first <= (6 * position) // 8 and (6 * position + 5) // 8 < last)
            if run:
                runs.add(run)
    return runs


def _forms(value: str) -> tuple[bytes, ...]:
    """The byte strings a token can appear as in a file: as written, percent-encoded, as JSON writes it (with and
    without ``\\u`` escapes), in base64 (either alphabet, at any offset), in hex (either case), and in UTF-16. Not
    every encoding: a token run through a cipher, reversed or split across lines is not found, which is why a bundle
    holds only the files a saved folder names."""
    raw = value.encode("utf-8")
    texts = {value, quote(value, safe=""), quote_plus(value), json.dumps(value)[1:-1],
             json.dumps(value, ensure_ascii=False)[1:-1]}
    found = {text.encode("utf-8") for text in texts}
    found |= {raw.hex().encode("ascii"), raw.hex().upper().encode("ascii")}
    found |= {value.encode("utf-16-le"), value.encode("utf-16-be")}
    found |= _base64_runs(raw)
    return tuple(sorted(found))


class Token:
    """A service's access token. ``repr`` and ``str`` say the service and nothing of the value, so a token that lands
    in a log line, an exception message or a traceback shows no secret; it is built from the text the moment that text
    is read, so that no frame of a traceback holds the text under a name of its own. The value comes out only through
    :meth:`reveal`, which the code that sends it to the service calls. It cannot be pickled, so it cannot be saved
    into a document or a record by accident."""

    __slots__ = ("_forms", "_service", "_value")

    def __init__(self, service: str, value: str) -> None:
        self._service = service
        self._value = value
        self._forms = _forms(value) if value else ()

    @property
    def service(self) -> str:
        return self._service

    @property
    def length(self) -> int:
        return len(self._value)

    @property
    def one_word(self) -> bool:
        """Whether the value is one run of characters with no space, tab or newline in it."""
        return bool(self._value) and len(self._value.split()) == 1 and self._value == self._value.strip()

    def reveal(self) -> str:
        return self._value

    def last_four(self) -> str:
        """The only part of a token a message may show, and only of a token long enough that four characters are a small
        part of it: nothing for a shorter one."""
        return self._value[-4:] if len(self._value) >= SHOWN_FROM_LENGTH else ""

    def occurs_in(self, data: bytes) -> bool:
        """Whether ``data`` holds the token in any form :func:`_forms` lists."""
        return any(form in data for form in self._forms)

    @override
    def __repr__(self) -> str:
        return f"<Token for {self._service}: masked>"

    @override
    def __str__(self) -> str:
        return self.__repr__()

    @override
    def __format__(self, format_spec: str) -> str:
        return format(self.__repr__(), format_spec)

    @override
    def __reduce__(self) -> NoReturn:
        raise TypeError("a token is never pickled or copied into a file; set it again where it is needed")
