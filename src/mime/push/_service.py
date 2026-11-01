"""The two services ``mime push`` publishes to, each fixed in code: its API, its site, the credential it pushes with,
and the words the plan names it in. No variable moves a token's host: a test injects a :class:`Service` of its own
through the Python call, never through the environment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

#: Plain ``http`` is accepted for these literal hosts only (the tests' fake on this machine); any other service is
#: reached over ``https``, so a token never travels in clear.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})
#: DataCite's test prefix, the sandbox's: a DOI under it does not resolve through doi.org, so a production record
#: given one is a crossed wire.
TEST_PREFIX = "10.5072/"


@dataclass(frozen=True)
class Service:
    """A records service: ``name`` (``zenodo`` or ``zenodo-sandbox``, and the state directory's), ``api`` (the records
    API's base, ``https://zenodo.org/api``), ``site`` (where a person reads a draft), ``credential`` (the
    ``mime credentials`` service whose token it pushes with), whether it is ``production``, and ``words``, how the plan
    names it."""

    name: str
    api: str
    site: str
    credential: str
    production: bool
    words: str

    def problem(self) -> str | None:
        """Why the service cannot be sent a token, or ``None``: its API is ``https``, or ``http`` to this machine."""
        parts = urlsplit(self.api)
        host = (parts.hostname or "").lower()
        if parts.scheme == "https" and host:
            return None
        if parts.scheme == "http" and host in _LOOPBACK_HOSTS:
            return None
        return f"the service {self.name}'s API is {self.api!r}: a token is sent over https, or to this machine"

    def draft_page(self, record: str) -> str:
        return f"{self.site}/uploads/{record}"


ZENODO = Service("zenodo", "https://zenodo.org/api", "https://zenodo.org", "zenodo", True,
                 "Zenodo (production: a published record there is permanent)")
ZENODO_SANDBOX = Service("zenodo-sandbox", "https://sandbox.zenodo.org/api", "https://sandbox.zenodo.org",
                         "zenodo-sandbox", False, "the Zenodo sandbox (a test service, which can be cleaned at any "
                                              "time; its DOIs do not resolve)")
#: The services by the name ``--to`` takes. ``--to`` is required: the two are never one flag apart by accident.
SERVICES: Mapping[str, Service] = {ZENODO.name: ZENODO, ZENODO_SANDBOX.name: ZENODO_SANDBOX}

__all__ = ["SERVICES", "TEST_PREFIX", "ZENODO", "ZENODO_SANDBOX", "Service"]
