"""Credentials, prepared for ``mime push``: a token per service, kept where a commit cannot publish it and shown nowhere.

``mime auth login | logout | status <service>`` for ``zenodo`` and ``zenodo-sandbox``. The token is read with
``getpass``, never from a command-line argument; it is stored in the user's config directory (the file 0600, its
directory 0700, never inside a git working tree), and its environment variable (``ZENODO_TOKEN``,
``ZENODO_SANDBOX_TOKEN``) overrides the file. A :class:`Token` prints masked. Nothing here goes online; a check against
the service's API is ``push``'s. The operating system's keyring is optional and not built. See the README.
"""

from mime.auth._store import (
    CREDENTIALS_FILE,
    DIRECTORY_MODE,
    FILE_MODE,
    MINIMUM_LENGTH,
    SERVICES,
    credentials_path,
    describe,
    held_values,
    login,
    logout,
    source,
    token,
)
from mime.auth._token import Token

__all__ = [
    "CREDENTIALS_FILE",
    "DIRECTORY_MODE",
    "FILE_MODE",
    "MINIMUM_LENGTH",
    "SERVICES",
    "Token",
    "credentials_path",
    "describe",
    "held_values",
    "login",
    "logout",
    "source",
    "token",
]
