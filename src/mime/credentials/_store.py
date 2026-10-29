"""Where a token lives, and how it is kept out of everywhere else.

A token is read from the environment variable of its service, which overrides the file, or from the credentials file
in the user's config directory (``platformdirs``): a JSON object of service to token, the file mode 0600 and its
directory 0700. The file is written through a temporary name created private, so it is never readable by others, not
even briefly. It is never written inside a git working tree, where a ``git add .`` would publish it.

A token is a :class:`Token` from the moment it is read: the text is wrapped in the expression that reads it and no
frame of a traceback holds it under a name, so a traceback captured with its locals shows none. The one place the text
is taken out again is the line that writes the file.

On Windows the modes mean nothing and are not set; the file is in the user's profile.
"""

from __future__ import annotations

import getpass
import json
import os
import secrets
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import platformdirs

from mime.build import BuildRefused, Refusal, RefusalClass
from mime.credentials._token import Token

#: The services a token is kept for, each with the variable that overrides the file.
SERVICES: Mapping[str, str] = {"zenodo": "ZENODO_TOKEN", "zenodo-sandbox": "ZENODO_SANDBOX_TOKEN"}

CREDENTIALS_FILE = "credentials"
DIRECTORY_MODE = 0o700
FILE_MODE = 0o600
#: A shorter value is not a token, and a pack that searched its files for it would match nearly everything.
MINIMUM_LENGTH = 8
_POSIX = os.name == "posix"
_SET_AGAIN = "remove it and set the token again with `mime credentials set <service>`"
_GIT_SECONDS = 10


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> BuildRefused:
    return BuildRefused([Refusal(code, category, message, action)])


def check_service(service: str) -> None:
    if service not in SERVICES:
        raise _refuse("credentials.service", RefusalClass.USER, f"no service is called {service!r}",
                      f"use one of {', '.join(SERVICES)}")


def credentials_path() -> Path:
    return Path(platformdirs.user_config_dir("mime")) / CREDENTIALS_FILE


def _nearest_existing(path: Path) -> Path:
    resolved = path.resolve()
    return next(each for each in (resolved, *resolved.parents) if each.is_dir())


def _git_says_inside(directory: Path) -> Path | None:
    """The top of the working tree ``git`` itself finds for ``directory``, as the user's own configuration and
    environment lead it (a ``GIT_DIR``, a ``core.worktree``), or ``None``. A missing ``git`` finds none."""
    try:
        inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree", "--show-toplevel"], cwd=directory,
                                capture_output=True, text=True, timeout=_GIT_SECONDS, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = inside.stdout.split("\n")
    return Path(lines[1]) if inside.returncode == 0 and lines[0] == "true" and lines[1] else None


def _named_work_tree(path: Path) -> Path | None:
    """The work tree ``GIT_WORK_TREE`` names, when ``path`` is in it: a dotfiles repository used as
    ``git --git-dir=... --work-tree=$HOME`` leaves no ``.git`` below the work tree to find."""
    named = os.environ.get("GIT_WORK_TREE")
    if not named:
        return None
    tree = Path(named).expanduser().resolve()
    return tree if tree in (path.resolve(), *path.resolve().parents) else None


def inside_git_tree(path: Path) -> Path | None:
    """The working tree ``path`` is in, or ``None``. Found three ways, each catching what the one before cannot: the
    nearest ancestor holding a ``.git``; the ``GIT_WORK_TREE`` the environment names; and what ``git rev-parse
    --is-inside-work-tree`` says from there, which follows ``GIT_DIR`` and ``core.worktree`` as the user has them set.
    A repository reached only through an alias that passes ``--git-dir`` and ``--work-tree`` on each call leaves no
    trace in the environment or the disk for any of them to find."""
    resolved = path.resolve()
    marked = next((each for each in (resolved, *resolved.parents) if (each / ".git").exists()), None)
    return marked or _named_work_tree(path) or _git_says_inside(_nearest_existing(path))


def _tokens_in(path: Path) -> dict[str, Token] | str:
    """The tokens the credentials file holds, or why it cannot be read (never the text it holds, and never
    raising while the text is held, so that no traceback carries it)."""
    try:
        held: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return f"cannot be read ({type(error).__name__})"
    if not isinstance(held, dict) or not all(isinstance(key, str) and isinstance(value, str)
                                              for key, value in cast("dict[object, object]", held).items()):
        return "is not an object of service to token"
    return {key: Token(key, value) for key, value in cast("dict[str, str]", held).items()}


def _read(path: Path) -> dict[str, Token]:
    if not path.exists():
        return {}
    found = _tokens_in(path)
    if isinstance(found, str):
        raise _refuse("credentials.store", RefusalClass.ENVIRONMENT, f"the credentials file {path} {found}", _SET_AGAIN)
    return found


def file_mode_problem(path: Path) -> str | None:
    """Why the credentials file is not private to its owner, or ``None``."""
    if not (_POSIX and path.exists()):
        return None
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        return f"has mode {mode:04o}, and a token must be readable by its owner alone ({FILE_MODE:04o})"
    return None


def _write(path: Path, held: Mapping[str, Token]) -> None:
    path.parent.mkdir(mode=DIRECTORY_MODE, parents=True, exist_ok=True)
    if _POSIX:
        path.parent.chmod(DIRECTORY_MODE)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            if _POSIX:
                os.fchmod(stream.fileno(), FILE_MODE)
            stream.write(json.dumps({service: each.reveal() for service, each in held.items()}, sort_keys=True) + "\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _from_environment(service: str) -> Token | None:
    found = Token(service, os.environ.get(SERVICES[service], "").strip())
    return found if found.length else None


def source(service: str) -> str | None:
    """Where the token of ``service`` comes from: its environment variable, which overrides the file, or the file."""
    check_service(service)
    if _from_environment(service) is not None:
        return f"environment variable {SERVICES[service]}"
    return "the credentials file" if service in _read(credentials_path()) else None


def token(service: str) -> Token | None:
    """The token of ``service``, or ``None``. The environment variable overrides the file; a file other users can read is
    refused (``credentials.file-mode``), as ``ssh`` refuses a private key it has left open."""
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
        raise _refuse("credentials.file-mode", RefusalClass.ENVIRONMENT, f"the credentials file {path} {problem}",
                      f"run `chmod 600 {path}`, or set the token again")
    return held[service]


def prompt(service: str) -> Token:
    """Read the token of ``service`` from the terminal, hidden, and return it wrapped: the text is never in a name of
    this function's, nor of its caller's. An end of input gives an empty token, which :func:`store` refuses."""
    check_service(service)
    try:
        return Token(service, getpass.getpass(f"{service} access token (not shown): ").strip())
    except EOFError:
        return Token(service, "")


def store(service: str, value: Token) -> Path:
    """Store the token ``value`` of ``service`` in the credentials file and return its path. Refused inside a git working
    tree, and for a value that is not one word of a plausible length; a message names neither the value nor part of
    it."""
    check_service(service)
    path = credentials_path()
    tree = inside_git_tree(path.parent)
    if tree is not None:
        raise _refuse("credentials.in-git-tree", RefusalClass.USER,
                      f"the credentials file would be {path}, inside the git working tree {tree}, where a commit "
                      f"could publish it", "move your config directory out of the repository (XDG_CONFIG_HOME on Linux)")
    if value.length < MINIMUM_LENGTH or not value.one_word:
        raise _refuse("credentials.token", RefusalClass.USER,
                      f"what was read is not a token: a token is one word of at least {MINIMUM_LENGTH} characters, and "
                      f"this has {value.length}", "paste the token the service shows, without spaces or a newline")
    _write(path, {**_read(path), service: value})
    return path


def remove(service: str) -> bool:
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
    """The line for ``mime credentials status``: whether there is a token, where from, and the last four characters of
    one of 32 or more. A file other users can read is reported here, where ``token`` refuses to use it."""
    check_service(service)
    path = credentials_path()
    held = _read(path)
    environment = _from_environment(service)
    stored = held.get(service)
    found = environment or stored
    if found is None:
        return f"{service}: no token"
    ending = found.last_four()
    line = f"{service}: a token, from {source(service)}, " + (f"ending {ending}" if ending else "too short to show any of it")
    if environment is not None and stored is not None:
        line += " (it overrides the credentials file's)"
    problem = file_mode_problem(path) if stored is not None else None
    return line if problem is None else f"{line}\n  warning: the credentials file {problem}: run `chmod 600 {path}`"


def held_tokens() -> list[Token]:
    """Every token MIME holds, for the one use that needs them: a pack searches its files for them (with
    :meth:`Token.occurs_in`). A token shorter than :data:`MINIMUM_LENGTH` is left out, and an unreadable file is not an
    error here."""
    found = [each for service in SERVICES if (each := _from_environment(service)) is not None]
    try:
        found += _read(credentials_path()).values()
    except BuildRefused:
        pass
    return [each for each in found if each.length >= MINIMUM_LENGTH]
