"""A published record as anyone reads it, with no token: what ``mime pull --from <service> <record>`` reads (design
#199 (e)). The sandbox's DOIs are under DataCite's test prefix, which doi.org does not resolve, so a sandbox record is
read by its number, at the sandbox's own records API, and never through doi.org; a production record may be read so
too.

The record is read in InvenioRDM's shape (``Accept: application/vnd.inveniordm.v1+json``): its DOI, its concept's DOI,
its version's number and whether it is the latest, and its files; it must hold one bundle, whose address in the
records API, md5 and size ``pull`` downloads it by. No token is sent, a redirect is never followed, and a service on
this machine (the tests' fake) is reached without the environment's proxy.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import cast
from urllib.parse import quote

from mime._terminal import plain
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._client import RDM_TYPE, RECORD_FILES_TYPE, NoRedirect
from mime.push._service import TEST_PREFIX, Service, loopback

_TIMEOUT_SECONDS = 60.0
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


def _get(service: Service, path: str, accept: str = RDM_TYPE) -> tuple[int, dict[str, str], bytes]:
    """One GET of ``path`` below the service's API, with no token, asking for ``accept``; the status, headers and
    body."""
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


def _json(service: Service, path: str, what: str, page: str, accept: str = RDM_TYPE) -> dict[str, object]:
    status, _, body = _get(service, path, accept)
    if status in (401, 403):
        why = "its files are restricted" if path.endswith("/files") else "it is a draft, or a restricted record"
        raise _refuse("pull.restricted", RefusalClass.USER,
                      f"{service.name} keeps {what} from anyone it is not shared with ({status}): {why}, and pull "
                      f"reads with no token",
                      f"if it is shared with you, download its bundle from its page signed in, {page}, then `mime pull "
                      f"<the .mimebundle file>`; a draft, publish it first; otherwise ask its owner for access")
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


def public_record(service: Service, given: str) -> PublicRecord:
    """The published record ``given`` names on ``service`` (:func:`record_number`), read with no token, and its one
    bundle."""
    from mime.saved._bundle import BUNDLE_SUFFIX

    record = record_number(service, given)
    page = f"{service.site}/records/{record}"
    held = _json(service, f"/records/{record}", f"the record {record}", page)
    if held.get("is_published") is False:
        raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                      f"the record {record} on {service.name} is a draft, which only its owner reads",
                      f"publish it, `mime push <folder> --to {service.name} --publish`, then pull it")
    listing = _json(service, f"/records/{record}/files", f"the files of the record {record}", page, RECORD_FILES_TYPE)
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
        status, headers, _ = _get(service, f"/records/{record}/versions/latest")
        found = re.fullmatch(re.escape(service.api) + r"/records/(\d+)", headers.get("location", "")) \
            if status in (301, 302, 303) else None
        latest = found.group(1) if found is not None else ""
    address = f"{service.api}/records/{record}/files/{quote(key, safe='')}/content"
    return PublicRecord(service, record, _doi(held), _doi(held.get("parent")),
                        index if isinstance(index, int) else None, key, address, checksum.removeprefix("md5:"), size,
                        latest)


__all__ = ["PublicRecord", "public_record"]
