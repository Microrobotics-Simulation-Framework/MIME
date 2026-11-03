"""A published record as anyone reads it, with no token: what ``mime pull --from <service> <record>`` reads (design
#199 (e)). The sandbox's DOIs are under DataCite's test prefix, which doi.org does not resolve, so a sandbox record is
read by its number, at the sandbox's own records API, and never through doi.org; a production record may be read so
too.

The record is read in InvenioRDM's shape (``Accept: application/vnd.inveniordm.v1+json``): its DOI, its concept's DOI,
its version's number and whether it is the latest, and its files; it must hold one bundle, whose address in the
records API, md5 and size ``pull`` downloads it by. Its other files are not downloaded: the NOTICE the record holds
beside the bundle (``NOTICE.txt``, so that Zenodo previews it) is the bundle's own ``NOTICE``, byte for byte, which the
bundle carries and ``unpack`` holds to its hash. No token is sent, a redirect is never followed, and a service on this
machine (the tests' fake) is reached without the environment's proxy.

**A record that is not public** is read only when the person asks: ``mime pull --from <service> <record> --signed-in``
sends the token ``mime credentials`` stored for that service, and ``mime pull <share link>`` the key a share link
carries (:class:`~mime.push._client.Access`). Either travels as push's token does (:func:`~mime.push._client.fetched`):
to the service's own API host alone, never on a redirect (a file's store is asked with nothing of either,
:func:`~mime.push._client.public_into`), never through a proxy to this machine, and never in a refusal, a traceback, a
log or a printed line: a link is said by its record alone. **That the records API takes a link's key as the parameter
``token`` is a guess** (InvenioRDM's, recalled) until a sandbox session makes a link and pulls it.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import parse_qs, quote, urlsplit

from mime._terminal import plain
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.credentials import Token
from mime.push._client import (
    ANY_TYPE,
    LINK_PARAMETER,
    RDM_TYPE,
    RECORD_FILES_TYPE,
    Access,
    Dropped,
    NoRedirect,
    fetched,
    public_into,
)
from mime.push._service import SERVICES, TEST_PREFIX, Service, loopback

_TIMEOUT_SECONDS = 60.0
#: The answers that redirect a download: read, and never followed with a token or a key.
_REDIRECTS = frozenset({301, 302, 303, 307, 308})
_RECORD = re.compile(r"[0-9]{1,12}")
_RECORD_DOI = re.compile(r"(?:doi:|https?://(?:dx\.)?doi\.org/)?(10\.[0-9]{4,9}/)zenodo\.([0-9]{1,12})", re.IGNORECASE)


@dataclass(frozen=True)
class PublicRecord:
    """A published record: its service and number, DOI, concept's DOI, version (its number, and whether it is the
    latest), and its bundle: the file's name, its address in the records API, its md5 and size; and, where it is not
    the latest version, the latest's number."""

    service: Service
    record: str
    doi: str | None
    concept_doi: str | None
    version: int | None
    bundle: str
    url: str
    md5: str
    size: int
    latest: str | None = None

    def words(self) -> str:
        version = f"version {self.version}, " if self.version is not None else ""
        concept = f"; its concept, every version: {self.concept_doi}" if self.concept_doi else ""
        return f"the record {self.record} on {self.service.name} ({version}{self.doi or 'no DOI'}{concept})"


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> BuildRefused:
    return BuildRefused([Refusal(code, category, message, action)])


@dataclass(frozen=True)
class ShareLink:
    """A record's share link, read: its service, the record's number, and the key it carries (masked wherever it is
    printed). The link itself is never said: a refusal names the record."""

    service: Service
    record: str
    key: Token


#: What a share link's key may be: a run of characters a URL carries unencoded (InvenioRDM's is a signed token).
_LINK_KEY = re.compile(r"[A-Za-z0-9._~+/=-]{8,4096}")
#: How a link held as a secret is labelled: a :class:`~mime.credentials.Token` says this and nothing of its value.
SHARE_LINK = "a share link"
#: The DOI resolver's hosts: a key on an address there is a share link given in a form no service's page gives.
_DOI_HOSTS = ("doi.org", "dx.doi.org")
#: The default port of each scheme a link is read in, which a link may write out.
_DEFAULT_PORTS = {"http": 80, "https": 443}


@dataclass(frozen=True)
class _LinkRead:
    """An address that carries a key, read by its components: the share link it is, or its refusal (``pull.source``),
    which says nothing of the key or of the address's other parts but its scheme, host and port."""

    link: ShareLink | None = None
    refused: BuildRefused | None = None


def _on(host: str) -> Service | None:
    """The service whose host ``host`` is, or a name below it (``www.``), the most particular first; ``None`` for
    another host."""
    found: list[tuple[int, Service]] = []
    for service in SERVICES.values():
        for base in (service.site, service.api):
            own = (urlsplit(base).hostname or "").rstrip(".")
            if own and (host == own or host.endswith("." + own)):
                found.append((len(own), service))
    return max(found, key=lambda each: each[0])[1] if found else None


def _read_link(held: Token) -> _LinkRead | None:
    """``held`` read by its components, where it is an address that carries a key (its :data:`LINK_PARAMETER`) on a
    service's host, a name below it, or the DOI resolver's; ``None`` for anything else, which is pulled as it was. It
    returns rather than raises, so that no frame of a refusal's traceback holds the address under a name."""
    try:
        parts = urlsplit(held.reveal().strip())
        scheme, host, port = parts.scheme, (parts.hostname or "").rstrip("."), parts.port
    except ValueError:
        return None
    if scheme not in ("http", "https") or not parts.query:
        return None
    keys = parse_qs(parts.query, keep_blank_values=True).get(LINK_PARAMETER)
    if not keys:
        return None
    if host in _DOI_HOSTS:
        return _LinkRead(refused=_refuse(
            "pull.source", RefusalClass.USER,
            f"an address at {host} carries an access key (its {LINK_PARAMETER} parameter), which goes to a record's "
            f"own service alone, and the DOI resolver is not one; nothing was sent",
            f"give the link as the record's page gives it, in quotes (a shell reads its ? and &): `mime pull "
            f"'<site>/records/<record>?{LINK_PARAMETER}=...'`, or `mime pull --link -`, which asks for it with no "
            f"echo"))
    on = _on(host)
    if on is None:
        return None
    own = urlsplit(on.site)
    form = f"{on.site}/records/<record>?{LINK_PARAMETER}=..."
    otherwise = [*([f"its scheme is {plain(scheme)}://, not {own.scheme}://"] if scheme != own.scheme else []),
                 *([f"its host is {plain(host)}, not {own.hostname}"] if host != own.hostname else []),
                 *([f"it names the port {port}"] if (port or _DEFAULT_PORTS[scheme]) !=
                   (own.port or _DEFAULT_PORTS.get(own.scheme)) else []),
                 *(["it names a user"] if parts.username is not None or parts.password is not None else [])]
    if otherwise:
        return _LinkRead(refused=_refuse(
            "pull.source", RefusalClass.USER,
            f"an address on {on.name}'s host carries an access key (its {LINK_PARAMETER} parameter), and is not the "
            f"share link {on.name}'s page gives, {form}: {'; '.join(otherwise)}; a link's key goes to its service's "
            f"own host alone, so nothing was sent",
            f"give the link as the record's page gives it, in quotes (a shell reads its ? and &): `mime pull "
            f"'{form}'`, or `mime pull --link -`, which asks for it with no echo"))
    for base in (on.site, on.api):
        found = re.fullmatch(re.escape(urlsplit(base).path) + r"/records/([0-9]{1,12})/?", parts.path)
        if found is not None and len(keys) == 1 and _LINK_KEY.fullmatch(keys[0]):
            return _LinkRead(ShareLink(on, found.group(1), Token(f"{SHARE_LINK} of {on.name}", keys[0])))
    return _LinkRead(refused=_refuse(
        "pull.source", RefusalClass.USER,
        f"{own.scheme}://{own.netloc}{plain(parts.path)} carries an access key (its {LINK_PARAMETER} parameter), and "
        f"it is not the share link of a record on {on.name}, which is {form}; nothing was sent",
        f"give the link as the record's page gives it, in quotes (a shell reads its ? and &): `mime pull '{form}'`, "
        f"or `mime pull --link -`, which asks for it with no echo"))


def share_link(given: str | Token) -> ShareLink | None:
    """The share link ``given`` is (an address, or one held as a secret): an address of a record on one of the two
    services, its page's (``<site>/records/<record>``) or its API's, carrying a key as
    :data:`~mime.push._client.LINK_PARAMETER`, in the scheme, host and port of that service's own site; ``None`` where
    ``given`` is not an address carrying a key on either service's host (an address elsewhere is pulled as it was).

    Read by its components, never by its spelling: an address that carries a key on a service's host, a name below it
    (``www.``) or the DOI resolver's, and is not a record's link in the service's own scheme, host and port (``http://``
    for ``https://``, a user part, another port, another name of the host), is refused (``pull.source``), saying none
    of the key: a link's key goes to its service's own host alone, for a record pull reads."""
    held = given if isinstance(given, Token) else Token(SHARE_LINK, given)
    given = held  # from here on no name of this frame holds the address's text
    read = _read_link(held)
    if read is None:
        return None
    if read.refused is not None:
        raise read.refused
    return read.link


def _get(service: Service, path: str, accept: str = RDM_TYPE,
         access: Access | None = None) -> tuple[int, dict[str, str], bytes]:
    """One GET of ``path`` below the service's API, asking for ``accept``, with no token, or let in by ``access``; the
    status, headers and body."""
    if access is not None:
        try:
            answer = fetched(service, path, access, accept=accept)
        except Dropped as dropped:
            raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                          f"{service.name} could not be read at {service.api + path}: {dropped}",
                          "check the network, then `mime pull --from <service> <record>` again") from None
        return answer.status, dict(answer.headers), answer.body
    url = service.api + path
    opener = urllib.request.build_opener(*([urllib.request.ProxyHandler({})] if loopback(url) else []), NoRedirect())
    request = urllib.request.Request(url, headers={"Accept": accept}, method="GET")
    try:
        with opener.open(request, timeout=_TIMEOUT_SECONDS) as response:
            return response.status, {k.lower(): v for k, v in response.headers.items()}, response.read()
    except urllib.error.HTTPError as error:
        return error.code, {k.lower(): v for k, v in error.headers.items()}, error.read()
    except (OSError, ValueError) as error:
        raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                      f"{service.name} could not be read at {url}: {type(error).__name__}",
                      "check the network, then `mime pull --from <service> <record>` again") from None


def _stored(service: Service) -> bool:
    """Whether a token is stored for ``service`` (in the credentials file or its variable), read without its value."""
    from mime import credentials

    try:
        return credentials.source(service.credential) is not None
    except BuildRefused:
        return False


def restricted(service: Service, what: str, record: str, page: str, status: int,
               access: Access | None, *, files: bool) -> BuildRefused:
    """``pull.restricted``: what was kept back, and the route that works, by how pull asked (with no token, signed in,
    or by a share link); with no token, by what refused: the record's own reading, or its files (``files``)."""
    if access is not None and access.key is not None:
        return _refuse("pull.restricted", RefusalClass.USER,
                       f"{service.name} keeps {what} back from the share link's key ({status}): the link was revoked "
                       f"or has expired, or the records API does not take a link's key as pull sends it (as its "
                       f"{LINK_PARAMETER} parameter, a guess until a sandbox session measures it)",
                       f"ask the record's owner for a new link; or, if it is shared with your account, `mime pull "
                       f"--from {service.name} {record} --signed-in`")
    if access is not None:
        return _refuse("pull.restricted", RefusalClass.USER,
                       f"{service.name} keeps {what} back from the token stored for it ({status}): it is a draft, or "
                       f"it is not shared with that token's account, or the token lacks a scope reading it needs",
                       f"see whose token it is, `mime credentials status {service.credential}`; ask the record's owner "
                       f"to share it with that account, or for a share link, `mime pull --link -`, which asks for it "
                       f"with no echo; or download its bundle from its page signed in, {page}, then `mime pull <the "
                       f".mimebundle file>`")
    signed = (f"if it is shared with your account, pull it signed in, `mime pull --from {service.name} {record} "
              f"--signed-in`, which sends the token `mime credentials` stored for {service.name}" if _stored(service)
              else f"if it is shared with your account, store that account's token, `mime credentials set "
                   f"{service.credential}`, then pull it signed in, `mime pull --from {service.name} {record} "
                   f"--signed-in`")
    return _refuse("pull.restricted", RefusalClass.USER,
                   f"{service.name} keeps {what} from anyone it is not shared with ({status}): "
                   f"{'its files are restricted' if files else 'it is a draft, or a restricted record'}, and pull "
                   f"reads with no token unless asked",
                   f"{signed}; or download its bundle from its page signed in, {page}, then `mime pull <the "
                   f".mimebundle file>`; a share link, `mime pull --link -`, which asks for it with no echo; a draft, "
                   f"publish it first; otherwise ask its owner for access")


def _json(service: Service, path: str, what: str, page: str, accept: str = RDM_TYPE, *, record: str = "<record>",
          access: Access | None = None) -> dict[str, object]:
    status, _, body = _get(service, path, accept, access)
    if status in (401, 403):
        raise restricted(service, what, record, page, status, access, files=path.endswith("/files"))
    if status == 404:
        raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                      f"{service.name} holds no published {what} (404)",
                      f"check the record's number on its page, {service.site}/records/<record> (a draft is not "
                      f"public until it is published), then `mime pull --from {service.name} <record>`")
    try:
        held: object = json.loads(body)
    except ValueError:
        held = None
    if status != 200 or not isinstance(held, dict):
        raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                      f"{service.name} answered {what} with {status}, not the record's JSON",
                      "run the same command again later, `mime pull --from <service> <record>`")
    return cast(dict[str, object], held)


def _map(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _doi(held: object) -> str | None:
    found = _map(_map(_map(held).get("pids")).get("doi")).get("identifier")
    return found if isinstance(found, str) else None


def record_number(service: Service, given: str) -> str:
    """The record's number in ``given``: the number, the record's page on ``service``, or its DOI there (under the test
    prefix on the sandbox, under any other on production). A DOI of the other service is refused, never read as a number
    here: the same number names another record there."""
    found = _RECORD_DOI.fullmatch(given.strip())
    if found is not None:
        if (found.group(1) == TEST_PREFIX) == service.production:
            other = "the sandbox" if service.production else "Zenodo itself"
            raise BuildRefused([Refusal(
                "pull.source", RefusalClass.USER, f"{plain(given)!r} is a DOI on {other}, not on {service.name}: the "
                f"same number names another record there", f"pull it from {other}, `mime pull --from "
                f"{'zenodo-sandbox' if service.production else 'zenodo'} {found.group(2)}`")])
        return found.group(2)
    page = re.fullmatch(re.escape(service.site) + r"/(?:records|uploads)/([0-9]{1,12})/?", given.strip())
    if page is not None:
        return page.group(1)
    if _RECORD.fullmatch(given) is None:
        raise BuildRefused([Refusal(
            "pull.source", RefusalClass.USER, f"{plain(given)!r} is not a record's number on {service.name}",
            f"give the number its page ends with, its page or its DOI, `mime pull --from {service.name} 629525`")])
    return given


def public_record(service: Service, given: str, access: Access | None = None) -> PublicRecord:
    """The published record ``given`` names on ``service`` (:func:`record_number`), read with no token (or let in by
    ``access``), and its one bundle."""
    from mime.saved._bundle import BUNDLE_SUFFIX

    record = record_number(service, given)
    page = f"{service.site}/records/{record}"
    held = _json(service, f"/records/{record}", f"the record {record}", page, record=record, access=access)
    if held.get("is_published") is False:
        raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                      f"the record {record} on {service.name} is a draft, which only its owner reads",
                      f"publish it, `mime push <folder> --to {service.name} --publish`, then pull it")
    listing = _json(service, f"/records/{record}/files", f"the files of the record {record}", page, RECORD_FILES_TYPE,
                    record=record, access=access)
    entries = [_map(each) for each in cast(list[object], listing.get("entries", []))
               if isinstance(listing.get("entries"), list)]
    bundles = [each for each in entries if str(each.get("key", "")).endswith(BUNDLE_SUFFIX)]
    if len(bundles) != 1:
        names = ", ".join(plain(str(each.get("key"))) for each in entries) or "none"
        raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                      f"the record {record} on {service.name} holds {len(bundles)} bundle files (its files: {names})",
                      "pull a record `mime push` made, which holds one bundle")
    bundle = bundles[0]
    key, checksum, size = str(bundle["key"]), str(bundle.get("checksum", "")), bundle.get("size")
    if not checksum.startswith("md5:") or not isinstance(size, int):
        raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                      f"the record {record} on {service.name} states no md5 and size for {plain(key)}",
                      "run the same command again later, `mime pull --from <service> <record>`")
    versions = _map(held.get("versions"))
    index = versions.get("index")
    latest: str | None = None
    if versions.get("is_latest") is False:
        status, headers, _ = _get(service, f"/records/{record}/versions/latest", access=access)
        found = re.fullmatch(re.escape(service.api) + r"/records/(\d+)", headers.get("location", "")) \
            if status in (301, 302, 303) else None
        latest = found.group(1) if found is not None else ""
    address = f"{service.api}/records/{record}/files/{quote(key, safe='')}/content"
    return PublicRecord(service, record, _doi(held), _doi(held.get("parent")),
                        index if isinstance(index, int) else None, key, address, checksum.removeprefix("md5:"), size,
                        latest)


def downloader(service: Service, record: str, access: Access) -> Callable[[str, Any, object], None]:
    """A download of a file of ``record`` for a pull let in by ``access``, as pooch calls a downloader (the address in the
    records API, the file written to, pooch): the address asked through :func:`~mime.push._client.fetched`, with the
    token or the key; a redirect to the service's own API asked so once more, and one anywhere else asked with nothing
    of either (:func:`~mime.push._client.public_into`); refused where the service keeps it back."""

    def download(url: str, output: Any, pooch: object) -> None:
        del pooch
        if not url.startswith(service.api + "/"):
            raise _refuse("pull.source", RefusalClass.USER, f"a download of the record {record} outside "
                                                            f"{service.name}'s API, which pull never sends a token to",
                          "pull the record by its number, `mime pull --from <service> <record>`")
        path = url[len(service.api):]
        for _ in range(2):
            try:
                answer = fetched(service, path, access, accept=ANY_TYPE, into=output.write)
            except Dropped as dropped:
                raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                              f"the bundle of the record {record} on {service.name} could not be downloaded: {dropped}",
                              "check the network, then `mime pull --from <service> <record>` again") from None
            if answer.status == 200:
                return
            if answer.status in (401, 403):
                raise restricted(service, f"the bundle of the record {record}", record,
                                 f"{service.site}/records/{record}", answer.status, access, files=True)
            if answer.status not in _REDIRECTS:
                raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                              f"{service.name} answered the bundle of the record {record} with {answer.status}",
                              "run the same command again later, `mime pull --from <service> <record>`")
            target = answer.headers.get("location", "")
            own, found = urlsplit(service.api), urlsplit(target)
            if (found.scheme, found.netloc) == (own.scheme, own.netloc) and found.path.startswith(own.path + "/"):
                path = target[len(f"{own.scheme}://{own.netloc}{own.path}"):]
                continue
            if access.key is not None and access.key.occurs_in(target.encode("utf-8")):
                raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                              f"{service.name} sent the bundle of the record {record} to another host with the share "
                              f"link's key in its address, which pull sends to the service's own host alone; nothing "
                              f"was asked there", "ask the record's owner for the bundle file, then `mime pull <the "
                                                  ".mimebundle file>`")
            try:
                public_into(target, output.write)
            except Dropped as dropped:
                raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                              f"the bundle of the record {record} could not be downloaded from where {service.name} "
                              f"stores it: {dropped}", "check the network, then `mime pull --from <service> <record>` "
                                                       "again") from None
            return
        raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                      f"{service.name} redirected the bundle of the record {record} within its API twice",
                      "run the same command again later, `mime pull --from <service> <record>`")

    return download


__all__ = ["SHARE_LINK", "PublicRecord", "ShareLink", "downloader", "public_record", "restricted", "share_link"]
