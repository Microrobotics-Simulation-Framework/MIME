"""The command words, in one place: the CLI parses them and a refusal quotes them, so renaming a verb is one edit."""

import os
import shlex

GROUP = "assets"
LIST = "list"
FETCH = "fetch"
CHECK = "check"


def folder_argument(folder: str | os.PathLike[str]) -> str:
    """``folder`` as the command takes it: a relative path starts ``./`` (so that it cannot be read as an asset's
    name) and the folder ends in ``/``, quoted where the shell needs it."""
    text = os.fspath(folder)
    if not os.path.isabs(text) and not text.startswith("."):
        text = "./" + text
    return shlex.quote(text if text.endswith("/") else text + "/")


def fetch_command(*names: str, folder: str | os.PathLike[str] | None = None) -> str:
    """The command line that fetches the assets ``names``, or every asset the experiment ``folder`` references, as a
    refusal tells the user to type it."""
    arguments = names if folder is None else (folder_argument(folder),)
    return " ".join(("mime", GROUP, FETCH, *arguments))


def check_command() -> str:
    return " ".join(("mime", GROUP, CHECK))
