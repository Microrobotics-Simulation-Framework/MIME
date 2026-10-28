"""Where a token lives, and how it is kept out of everywhere else.

A token is read from the environment variable of its service, which overrides the file, or from the credentials file
in the user's config directory (``platformdirs``): a JSON object of service to token, the file mode 0600 and its
directory 0700. The file is written through a temporary name created private, so it is never readable by others, not
even briefly. It is never written inside a git working tree, where a ``git add .`` would publish it.

On Windows the modes mean nothing and are not set; the file is in the user's profile.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import platformdirs

from mime.auth._token import Token
from mime.build import BuildRefused, Refusal, RefusalClass

#: The services a token is kept for, each with the variable that overrides the file.
SERVICES: Mapping[str, str] = {"zenodo": "ZENODO_TOKEN", "zenodo-sandbox": "ZENODO_SANDBOX_TOKEN"}

CREDENTIALS_FILE = "credentials"
DIRECTORY_MODE = 0o700
FILE_MODE = 0o600
#: A shorter value is not a token, and a pack that searched its files for it would match nearly everything.
MINIMUM_LENGTH = 8
_POSIX = os.name == "posix"
_TOKEN_TEXT = re.compile(r"\S+")
_LOGIN_AGAIN = "remove it and log in again with `mime auth login <service>`"


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> BuildRefused:
    return BuildRefused([Refusal(code, category, message, action)])


def check_service(service: str) -> None:
    if service not in SERVICES:
        raise _refuse("auth.service", RefusalClass.USER, f"no service is called {service!r}",
                      f"use one of {', '.join(SERVICES)}")


def credentials_path() -> Path:
    return Path(platformdirs.user_config_dir("mime")) / CREDENTIALS_FILE


def inside_git_tree(path: Path) -> Path | None:
    """The working tree ``path`` is in (the nearest ancestor holding a ``.git``), or ``None``."""
    resolved = path.resolve()
    return next((each for each in (resolved, *resolved.parents) if (each / ".git").exists()), None)


def _read(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        held: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:  # not chained: the parser's message can quote the file's text
        raise _refuse("auth.store", RefusalClass.ENVIRONMENT,
                      f"the credentials file {path} cannot be read ({type(error).__name__})", _LOGIN_AGAIN) from None
    if not isinstance(held, dict) or not all(isinstance(key, str) and isinstance(value, str)
                                              for key, value in cast("dict[object, object]", held).items()):
        raise _refuse("auth.store", RefusalClass.ENVIRONMENT,
                      f"the credentials file {path} is not an object of service to token", _LOGIN_AGAIN)
    return cast("dict[str, str]", held)


def file_mode_problem(path: Path) -> str | None:
    """Why the credentials file is not private to its owner, or ``None``."""
    if not (_POSIX and path.exists()):
        return None
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        return f"has mode {mode:04o}, and a token must be readable by its owner alone ({FILE_MODE:04o})"
    return None


def _write(path: Path, held: Mapping[str, str]) -> None:
    path.parent.mkdir(mode=DIRECTORY_MODE, parents=True, exist_ok=True)
    if _POSIX:
        path.parent.chmod(DIRECTORY_MODE)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            if _POSIX:
                os.fchmod(stream.fileno(), FILE_MODE)
            stream.write(json.dumps(dict(held), sort_keys=True) + "\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _from_environment(service: str) -> Token | None:
    value = os.environ.get(SERVICES[service], "").strip()
    return Token(service, value) if value else None


def source(service: str) -> str | None:
    """Where the token of ``service`` comes from: its environment variable, which overrides the file, or the file."""
    check_service(service)
    if _from_environment(service) is not None:
        return f"environment variable {SERVICES[service]}"
    return "the credentials file" if service in _read(credentials_path()) else None


def token(service: str) -> Token | None:
    """The token of ``service``, or ``None``. The environment variable overrides the file; a file other users can read is
    refused (``auth.file-mode``), as ``ssh`` refuses a private key it has left open."""
    check_service(service)
    found = _from_environment(service)
    if found is not None:
        return found
    path = credentials_path()
    held = _read(path)
    if service not in held:
        return None
    problem = file_mode_problem(path)
    if problem is not None:
        raise _refuse("auth.file-mode", RefusalClass.ENVIRONMENT, f"the credentials file {path} {problem}",
                      f"run `chmod 600 {path}`, or log in again")
    return Token(service, held[service])


def login(service: str, value: str) -> Path:
    """Store the token ``value`` of ``service`` in the credentials file and return its path. Refused inside a git working
    tree, and for a value that is not one token of a plausible length; a message names neither the value nor part of
    it."""
    check_service(service)
    path = credentials_path()
    tree = inside_git_tree(path.parent)
    if tree is not None:
        raise _refuse("auth.in-git-tree", RefusalClass.USER,
                      f"the credentials file would be {path}, inside the git working tree {tree}, where a commit "
                      f"could publish it", "move your config directory out of the repository (XDG_CONFIG_HOME on Linux)")
    if len(value) < MINIMUM_LENGTH or not _TOKEN_TEXT.fullmatch(value):
        raise _refuse("auth.token", RefusalClass.USER,
                      f"what was read is not a token: a token is one word of at least {MINIMUM_LENGTH} characters, and "
                      f"this has {len(value)}", "paste the token the service shows, without spaces or a newline")
    _write(path, {**_read(path), service: value})
    return path


def logout(service: str) -> bool:
    """Delete the token of ``service`` from the file (and the file when it holds nothing else). Whether one was
    there. A token in the environment is the environment's to unset."""
    check_service(service)
    path = credentials_path()
    held = _read(path)
    if service not in held:
        return False
    del held[service]
    if held:
        _write(path, held)
    else:
        path.unlink()
    return True


def describe(service: str) -> str:
    """The line for ``mime auth status``: whether there is a token, where from, and its last four characters. A file
    other users can read is reported here, where ``token`` refuses to use it."""
    check_service(service)
    path = credentials_path()
    held = _read(path)
    environment = _from_environment(service)
    stored = Token(service, held[service]) if service in held else None
    found = environment or stored
    if found is None:
        return f"{service}: not logged in"
    line = f"{service}: logged in, from {source(service)}, ending {found.last_four()}"
    if environment is not None and stored is not None:
        line += " (it overrides the credentials file's)"
    problem = file_mode_problem(path) if stored is not None else None
    return line if problem is None else f"{line}\n  warning: the credentials file {problem}: run `chmod 600 {path}`"


def held_values() -> list[str]:
    """Every token MIME holds, as text, for the one use that needs them: a pack searches its files for them. A token
    shorter than :data:`MINIMUM_LENGTH` is left out, and an unreadable file is not an error here."""
    values = [value for service in SERVICES if (found := _from_environment(service)) is not None
              for value in (found.reveal(),)]
    try:
        values += _read(credentials_path()).values()
    except BuildRefused:
        pass
    return sorted({value for value in values if len(value) >= MINIMUM_LENGTH})
