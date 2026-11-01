"""The two services ``mime push`` publishes to, each fixed in code: its API, its site, the credential it pushes with,
and the words the plan names it in. No variable moves a token's host: a test injects a :class:`Service` of its own
through the Python call, never through the environment, and a service pairs a real service's credential with that
service's own host alone (or a stand-in's on this machine or at a ``.invalid`` address), so no call, however it is
made, sends one service's token to another host."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

#: Plain ``http`` is accepted for these literal hosts only (the tests' fake on this machine), which are reached without
#: the environment's proxy; any other service is reached over ``https``, so a token never travels in clear.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})
#: The domain reserved never to resolve (RFC 2606): a stand-in there can be sent a real credential's token, which never
#: leaves the process that holds the stand-in.
_NEVER_RESOLVES = ".invalid"
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
        """Why the service cannot be sent a token, or ``None``: its API is ``https``, or ``http`` to this machine; and a
        real service's credential goes to that service's host, which takes no other's, as production or not as that
        service is."""
        parts = urlsplit(self.api)
        host = (parts.hostname or "").lower()
        if not ((parts.scheme == "https" and host) or (parts.scheme == "http" and host in _LOOPBACK_HOSTS)):
            return f"the service {self.name}'s API is {self.api!r}: a token is sent over https, or to this machine"
        real = {service.credential: service for service in (ZENODO, ZENODO_SANDBOX)}
        by_host = {_host(service.api): service for service in (ZENODO, ZENODO_SANDBOX)}
        own = real.get(self.credential)
        if own is not None and host != _host(own.api) and not _stand_in(host):
            return (f"the service {self.name} would send the {self.credential} token to {host}: a real service's token "
                    f"goes to its own host, {_host(own.api)}, and to a stand-in on this machine or at a "
                    f"{_NEVER_RESOLVES} address, and nowhere else")
        there = by_host.get(host)
        if there is not None and (self.credential, self.production) != (there.credential, there.production):
            return (f"the service {self.name} is at {host}, {there.name}'s host, with the {self.credential} token "
                    f"{'as' if self.production else 'not as'} production: {host} takes the {there.credential} token "
                    f"alone, {'as' if there.production else 'not as'} production")
        return None

    def draft_page(self, record: str) -> str:
        return f"{self.site}/uploads/{record}"


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _stand_in(host: str) -> bool:
    return host in _LOOPBACK_HOSTS or host.endswith(_NEVER_RESOLVES)


def loopback(url: str) -> bool:
    """Whether ``url`` is on this machine, by its literal host: reached directly, never through a proxy."""
    return _host(url) in _LOOPBACK_HOSTS


ZENODO = Service("zenodo", "https://zenodo.org/api", "https://zenodo.org", "zenodo", True,
                 "Zenodo (production: a published record there is permanent)")
ZENODO_SANDBOX = Service("zenodo-sandbox", "https://sandbox.zenodo.org/api", "https://sandbox.zenodo.org",
                         "zenodo-sandbox", False, "the Zenodo sandbox (a test service, which can be cleaned at any "
                                              "time; its DOIs do not resolve)")
#: The services by the name ``--to`` takes. ``--to`` is required: the two are never one flag apart by accident.
SERVICES: Mapping[str, Service] = {ZENODO.name: ZENODO, ZENODO_SANDBOX.name: ZENODO_SANDBOX}

__all__ = ["SERVICES", "TEST_PREFIX", "ZENODO", "ZENODO_SANDBOX", "Service", "loopback"]
