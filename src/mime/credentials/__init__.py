"""Credentials, prepared for ``mime push``: a token per service, kept where a commit cannot publish it and shown nowhere.

``mime credentials set | status | remove <service>`` for ``zenodo`` and ``zenodo-sandbox``. MIME has no accounts, so
nothing is "logged into": it stores another service's credential. The token is read with ``getpass``, never from a
command-line argument; it is stored in the user's config directory (the file 0600, its directory 0700, never inside a
git working tree), and its environment variable (``ZENODO_TOKEN``, ``ZENODO_SANDBOX_TOKEN``) overrides the file. A
:class:`Token` prints masked. Nothing here goes online; a check against the service's API is ``push``'s. The operating
system's keyring is optional and not built. See the README.
"""

from mime.credentials._store import (
    CREDENTIALS_FILE,
    DIRECTORY_MODE,
    FILE_MODE,
    MINIMUM_LENGTH,
    SERVICES,
    credentials_path,
    describe,
    held_tokens,
    prompt,
    remove,
    source,
    store,
    token,
)
from mime.credentials._token import SHOWN_FROM_LENGTH, Token

__all__ = [
    "CREDENTIALS_FILE",
    "DIRECTORY_MODE",
    "FILE_MODE",
    "MINIMUM_LENGTH",
    "SERVICES",
    "SHOWN_FROM_LENGTH",
    "Token",
    "credentials_path",
    "describe",
    "held_tokens",
    "prompt",
    "remove",
    "source",
    "store",
    "token",
]
