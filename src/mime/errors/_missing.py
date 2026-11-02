"""A module MIME's own requirements name, missing from the environment: refused by name, with the command that
installs it, never as an error MIME gave no code.

The requirements are read from ``mime.data``'s ``env/declared.json``, which ships with the code that declares them, not
from the installed metadata: an editable install keeps the requirements it was installed with, so the metadata of an
environment that predates a requirement is the one place that does not name it, and that environment is the case here.
``tests/errors/test_missing.py`` holds the file to ``pyproject.toml``, and each module to what its distribution
provides."""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import importlib.util
import json
import shlex
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse

from packaging.requirements import Requirement

from mime import __version__
from mime.build._refusals import BuildRefused, MimeError, Refusal, RefusalClass

DISTRIBUTION = "mime-engine"
DECLARED_FILE = "declared.json"


@dataclass(frozen=True)
class Declared:
    """One requirement of ``mime-engine``: as written, the extra that holds it (``None`` for a core requirement) and
    the top-level module it provides."""

    requirement: str
    extra: str | None
    module: str

    @property
    def distribution(self) -> str:
        return Requirement(self.requirement).name


@cache
def declared_requirements() -> tuple[Declared, ...]:
    """Every requirement ``mime-engine`` declares, core first, each with the module it provides."""
    text = (importlib.resources.files("mime.data") / "env" / DECLARED_FILE).read_text(encoding="utf-8")
    return tuple(Declared(**entry) for entry in json.loads(text))


def declaring(module: str) -> Declared | None:
    """The requirement that provides ``module`` (a top-level name), a core one before one behind an extra."""
    held = [d for d in declared_requirements() if d.module == module]
    return min(held, key=lambda d: d.extra is not None) if held else None


def _checkout() -> Path | None:
    """Where this MIME is installed from in place (an editable install), as the installer recorded it."""
    try:
        text = importlib.metadata.distribution(DISTRIBUTION).read_text("direct_url.json")
        record = cast(dict[str, Any], json.loads(text)) if text else {}
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return None
    if not cast(dict[str, Any], record.get("dir_info", {})).get("editable"):
        return None
    url = urlparse(str(record.get("url", "")))
    return Path(unquote(url.path)) if url.scheme == "file" else None


def reinstall_command(extra: str | None = None) -> str:
    """The command that installs MIME's requirements (with ``extra``'s, where one is named) into this environment:
    ``uv sync`` where the environment is the ``.venv`` of the checkout MIME is installed from, ``pip install -e`` that
    checkout otherwise, and, for an installed release, ``pip install --upgrade mime-engine``."""
    checkout = _checkout()
    if checkout is None:
        return f"pip install --upgrade {shlex.quote(f'{DISTRIBUTION}[{extra}]' if extra else DISTRIBUTION)}"
    if (checkout / "uv.lock").is_file() and (checkout / ".venv").resolve() == Path(sys.prefix).resolve():
        return f"cd {shlex.quote(str(checkout))} && uv sync" + (f" --extra {shlex.quote(extra)}" if extra else "")
    return f"pip install -e {shlex.quote(f'{checkout}[{extra}]' if extra else str(checkout))}"


def missing_dependency(error: BaseException) -> MimeError | None:
    """``error``, if it is a module that ``mime-engine`` declares a requirement for, not found; else ``None``. A
    submodule of a package that is present, and a module no requirement names, are not it."""
    from mime import __version__

    if not isinstance(error, ModuleNotFoundError) or not error.name:
        return None
    module = error.name.split(".", 1)[0]
    declared = declaring(module)
    if declared is None or importlib.util.find_spec(module) is not None:
        return None
    requirer = f"{DISTRIBUTION} {__version__}"
    if declared.extra:
        requirer = f"the {declared.extra!r} extra of {requirer}"
        cause = "the extra was not installed with MIME, or the package was removed"
    else:
        cause = ("this environment's packages predate this MIME (an editable install keeps the requirements it was "
                 "installed with, so one added since is missing), or the package was removed")
    return BuildRefused([Refusal(
        "env.missing", RefusalClass.ENVIRONMENT,
        f"{declared.requirement} is not installed, and {requirer} requires it (MIME could not import {error.name!r})",
        f"install it with MIME's requirements: {reinstall_command(declared.extra)}; or the package alone: "
        f"pip install {shlex.quote(declared.requirement)}", why=cause)])


__all__ = ["DECLARED_FILE", "DISTRIBUTION", "Declared", "declared_requirements", "declaring", "missing_dependency",
           "reinstall_command"]
