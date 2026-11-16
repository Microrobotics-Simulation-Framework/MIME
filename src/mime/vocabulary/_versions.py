"""The versions of entries and namespaces: parsed as versions (PEP 440, the numbers joined by dots a release has), and
ordered as versions, so that ``0.10.0`` is after ``0.9.0``."""

from __future__ import annotations

from packaging.version import InvalidVersion, Version


def parsed(version: object) -> Version | None:
    """``version`` as a version, or ``None`` where it is not a text of one. A text with whitespace around it is none:
    :class:`~packaging.version.Version` strips it, which would make ``' 0.1.0'`` and ``'0.1.0'`` two keys of one
    version."""
    if not isinstance(version, str) or version != version.strip():
        return None
    try:
        return Version(version)
    except InvalidVersion:
        return None


def version_problem(version: object) -> str | None:
    """Why ``version`` is not a version a registration takes, or ``None``: numbers joined by dots, as a release's
    (``0.1.0``), with no whitespace around it, which orders correctly (``0.10.0`` after ``0.9.0``)."""
    return None if parsed(version) is not None else (
        f"version {version!r}: a version is numbers joined by dots, such as '0.1.0' (PEP 440), with no space or newline "
        f"around it, so that it orders as a version: '0.10.0' after '0.9.0'")


def version_order(version: str) -> tuple[bool, Version, str]:
    """The key that sorts versions as versions, the texts that are none after those that are."""
    held = parsed(version)
    return held is None, held if held is not None else Version("0"), version
