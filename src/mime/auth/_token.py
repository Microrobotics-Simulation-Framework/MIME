"""The one type a credential is held in: its text and its debugging forms are masked, and it cannot be pickled."""

from __future__ import annotations

from typing import NoReturn, override


class Token:
    """A service's access token. ``repr`` and ``str`` say the service and nothing of the value, so a token that lands
    in a log line, an exception message or a traceback shows no secret; the value comes out only through
    :meth:`reveal`, which the code that sends it to the service calls. It cannot be pickled, so it cannot be saved
    into a document or a record by accident."""

    __slots__ = ("_service", "_value")

    def __init__(self, service: str, value: str) -> None:
        self._service = service
        self._value = value

    @property
    def service(self) -> str:
        return self._service

    def reveal(self) -> str:
        return self._value

    def last_four(self) -> str:
        """The only part of a token a message may show."""
        return self._value[-4:]

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
        raise TypeError("a token is never pickled or copied into a file; log in again where it is needed")
