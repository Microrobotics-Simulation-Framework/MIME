"""Credentials, prepared for ``mime push``: a token per service, kept where a commit cannot publish it and shown nowhere.

``mime credentials set | status | remove <service>`` for ``zenodo`` and ``zenodo-sandbox``, and for a host's own token
(``host:<host name>``), which an asset adapter that needs a credential is handed for that host alone. MIME has no accounts, so
nothing is "logged into": it stores another service's credential. The token is read with ``getpass``, never from a
command-line argument; it is stored in the user's config directory (the file 0600, its directory 0700, never inside a
git working tree), and its environment variable (``ZENODO_TOKEN``, ``ZENODO_SANDBOX_TOKEN``) overrides the file. A
:class:`Token` prints masked. Nothing here goes online; a check against the service's API is ``push``'s. The operating
system's keyring is optional and not built. See the README.
"""

from mime.credentials._token import Token

__all__ = ["Token"]
