"""The templates ``mime scaffold`` fills: package data under ``templates/<kind>/``, filled by :class:`string.Template`, never
fetched. A template names each value it takes as ``${key}``; one it does not get is an error (``substitute``), and a value
is substituted as it is, never read again for a ``$``."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import files
from string import Template


def template(kind: str, name: str) -> str:
    """The template ``name`` of ``kind``, as shipped."""
    return files("mime.scaffold").joinpath("templates", kind, name).read_text(encoding="utf-8")


def fill(kind: str, name: str, values: Mapping[str, str]) -> str:
    """The template ``name`` of ``kind`` with ``values`` in it. Looks the template up through this module's
    :func:`template` each time, so a test can plant a fault in one."""
    return Template(template(kind, name)).substitute(values)


__all__ = ["fill", "template"]
