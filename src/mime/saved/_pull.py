"""``pull``: get a bundle, or a record's studies, by DOI, URL or file, unpack it into a new folder, and say what is still
needed.

A record holds one kind of container: one bundle, or one or more studies (``.mimestudy``). One study unpacks into
``./<its name>/`` (or ``--into``); several need ``--into``, refused before anything is downloaded, and each is unpacked
into a folder of its own below it. Their assets are gated once, over every study's locks.

The bundle is fetched through ``mime.assets`` as any content-addressed file is: stored in the cache under its hash, the
network used only because this was asked for, and offline refused. Its assets are fetched the same way afterwards,
unless declined, and each file that travels as a reference (one its licence does not let a bundle carry) is placed at
its path from the cache. Nothing here installs or runs a package: the packages the lock names that are not installed
are listed with the command to install them.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
import shlex
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import unquote

from mime import assets
from mime._lock_json import lock_data
from mime._said import DONE, PLAN_LINE, STOPPED, Said
from mime._shell import quoted
from mime._terminal import plain
from mime.assets._cache import locate_bundle_container
from mime.assets._commands import fetch_command, folder_argument
from mime.assets._reference import carries_key, redacted
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.credentials import Token
from mime.saved._bundle import BUNDLE, BUNDLE_SUFFIX, FOLDER_PREFIX, Unpacked, bundle_name, destination_problem, unpack
from mime.saved._container import collisions, staging_of
from mime.saved._folder import LOCK_FILE
from mime.saved._schema import Lock
from mime.saved._terms import Placed, place_referenced

if TYPE_CHECKING:
    from mime.push._client import Access

_RAISE_LIMIT = "if the bundle is that large on purpose, raise the limit: `mime pull --max-bundle-bytes <bytes>`"
_DOI = re.compile(r"10\.\d{4,9}/\S+")
_DOI_PREFIXES = ("doi:", "https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/")
#: DataCite's test prefix, the Zenodo sandbox's: doi.org does not resolve it, so MIME never asks doi.org for one.
TEST_PREFIX = "10.5072/"


@dataclass(frozen=True)
class Source:
    """Where a bundle comes from: a ``file`` on this machine, a ``url`` to download, or a ``doi`` (with the record's file
    when the source names one)."""

    kind: Literal["file", "url", "doi"]
    value: str
    file: str | None = None


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> BuildRefused:
    return BuildRefused([Refusal(code, category, message, action)])


def parse_source(text: str) -> Source:
    """A DOI (``10.5281/zenodo.1``, ``doi:...`` or ``https://doi.org/...``, optionally ending ``/<file>.mimebundle``), an
    ``https://`` URL, or the path of a file."""
    if Path(text).is_file():
        return Source("file", text)
    bare = next((text.removeprefix(prefix) for prefix in _DOI_PREFIXES if text.startswith(prefix)), text)
    if (_DOI.fullmatch(bare) and bare.startswith(TEST_PREFIX)) or assets.under_test_prefix(text):
        number = re.search(r"zenodo\.(\d+)", unquote(bare))
        raise _refuse("pull.sandbox-doi", RefusalClass.USER,
                      f"{redacted(text)!r} is under DataCite's test prefix 10.5072, the Zenodo sandbox's, which "
                      f"doi.org does not resolve, so MIME does not ask doi.org for it",
                      f"pull the record by its number on the sandbox, `mime pull --from zenodo-sandbox "
                      f"{number.group(1) if number is not None else '<record>'}`")
    if _DOI.fullmatch(bare):
        from mime.saved._studies import kind_of_name

        doi, _, last = bare.rpartition("/")
        return Source("doi", doi, last) if kind_of_name(last) is not None and doi.count("/") >= 1 else Source("doi", bare)
    if text.startswith(("https://", "http://")):
        problem = assets.location_problem(text)
        if problem is not None:
            raise _refuse("pull.source", RefusalClass.USER, f"{redacted(text)!r}: {problem}",
                          "give an https:// address, a DOI, or a file: `mime pull <doi | url | file>`")
        return Source("url", text)
    raise _refuse("pull.source", RefusalClass.USER,
                  f"{redacted(text)!r} is not a file on this machine, a DOI or an https:// address",
                  "give the bundle's DOI, its address, or its file: `mime pull 10.5281/zenodo.1234567`")


def _held(value: object) -> Any:
    return value


def _offline_refusal(what: str) -> BuildRefused:
    return _refuse("asset.offline", RefusalClass.ENVIRONMENT,
                   f"{what} is not on this machine, and {assets.OFFLINE_VARIABLE} is set, so nothing is downloaded",
                   f"unset {assets.OFFLINE_VARIABLE}, or give the bundle as a file: `mime pull <file>`")


def _stated_size(repository: Any, file: str) -> int | None:
    """The size the repository's record states for ``file`` (Zenodo's ``size`` or, in its other layout, ``filesize``),
    or ``None`` where the record states none."""
    try:
        for held in repository.api_response["files"]:
            if file in (held.get("key"), held.get("filename")):
                size = held.get("size", held.get("filesize"))
                return size if isinstance(size, int) and not isinstance(size, bool) and size > 0 else None
    except Exception:  # noqa: BLE001 - a repository other than Zenodo has no such response; a missing size is no failure
        return None
    return None


def _version_words(doi: str, repository: Any) -> str | None:
    """What the record a DOI resolved to is, where the repository says (Zenodo's record: its DOI, its concept's DOI
    and its version): a concept DOI names every version, and resolves to the latest, which this names."""
    try:
        held = repository.api_response
        found, concept = str(held.get("doi") or ""), str(held.get("conceptdoi") or "")
        version = held.get("metadata", {}).get("version")
    except Exception:  # noqa: BLE001 - a repository other than Zenodo has no such response; the words are a courtesy
        return None
    if concept and doi.lower() == concept.lower() and found:
        return (f"{doi} is the concept DOI, which names every version; it resolved to the latest, "
                f"{plain(found)}" + (f" (version {plain(str(version))})" if version else ""))
    if found:
        return f"{plain(found)}" + (f", version {plain(str(version))}" if version else "") + \
            (f", of the concept {plain(concept)}" if concept else "")
    return None


def _record_doi(doi: str, repository: Any) -> str:
    """The DOI of the record ``doi`` resolved to, as the repository states it (Zenodo's record states its own: a concept
    DOI resolves to its latest version, whose DOI this is), or ``doi`` where it states none in a DOI's shape."""
    try:
        found = repository.api_response.get("doi")
    except Exception:  # noqa: BLE001 - a repository other than Zenodo has no such response
        return doi
    return found if isinstance(found, str) and _DOI.fullmatch(found) else doi


def _record_file(doi: str, wanted: str | None) -> tuple[list[tuple[str, str, int | None]], str | None, str]:
    """The containers of the repository record ``doi`` names, each with the repository's checksum of it and the size it
    states (or ``None``); the words naming the version it is; and the record's own DOI (:func:`_record_doi`), through
    pooch's DOI support. The record holds one ``.mimebundle`` file, or one or more ``.mimestudy`` files, or ``wanted``
    names the one to pull."""
    import pooch

    registry: dict[str, str] = {}
    named: str | None = None
    resolved = doi
    repository: Any = None
    try:
        with tempfile.TemporaryDirectory() as scratch:
            library = _held(pooch)
            listing = library.create(path=scratch, base_url=f"doi:{doi}/", registry={})
            downloader = library.downloaders.choose_downloader(f"doi:{doi}/")
            repository = library.downloaders.doi_to_repository(
                doi, headers=downloader.headers, timeout=downloader.timeout, **downloader.kwargs)
            repository.populate_registry(listing)
            registry = dict(listing.registry)
            named = _version_words(doi, repository)
            resolved = _record_doi(doi, repository)
    except Exception as error:  # noqa: BLE001 - pooch and requests raise unrelated types for a record that can't be read
        failed: BuildRefused | None = _refuse(
            "pull.record", RefusalClass.ENVIRONMENT,
            f"the record doi:{redacted(doi)} could not be read: {plain(redacted(str(error)))}",
            "check the DOI and the network, then `mime pull <doi>` again")
        if not carries_key(doi):
            raise failed from error
    else:
        failed = None
    if failed is not None:  # a DOI that carries a key: raised here, chained to no error that quotes it
        raise failed from None
    from mime.saved._studies import STUDY_SUFFIX, kind_of_name

    bundles = sorted(name for name in registry if name.endswith(BUNDLE_SUFFIX))
    studies = sorted(name for name in registry if name.endswith(STUDY_SUFFIX))
    if wanted is not None and wanted in registry and kind_of_name(wanted) is not None:
        chosen = [wanted]
    elif len(bundles) == 1 and not studies:
        chosen = bundles
    elif studies and not bundles:
        chosen = studies
    else:
        held = ", ".join(plain(name) for name in sorted(registry)) or "none"
        raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                      f"the record doi:{doi} lists {len(bundles)} bundle files and {len(studies)} study containers "
                      f"(its files: {held}), and MIME pulls one bundle, or one or more studies",
                      f"name the one to pull: `mime pull doi:{quoted(plain(doi))}/<file>{BUNDLE_SUFFIX}` (or <file>{STUDY_SUFFIX})")
    return ([(file, checksum if ":" in (checksum := registry[file]) else f"md5:{checksum}",
              _stated_size(repository, file)) for file in chosen], named, resolved)


def _limit(stated: int | None, allowed: int) -> int:
    """The size a bundle download is held to: the size the record states, or ``allowed`` when none is stated. A stated size
    over ``allowed`` is refused too, so that the cap holds whichever way the size came."""
    if stated is not None and stated > allowed:
        raise _refuse("asset.size-exceeded", RefusalClass.ENVIRONMENT,
                      f"the record states a bundle of {assets.size_text(stated)}, over the {assets.size_text(allowed)} "
                      f"MIME takes without being told to", _RAISE_LIMIT)
    return stated if stated is not None else allowed


@dataclass(frozen=True)
class Fetched:
    """A container a pull has in hand: its file (in the asset cache, or where the source names it) and the name it came
    by (the record's file name, the address's last part, the file's own), whose suffix must say its kind."""

    path: Path
    named: str


def _several_refused(count: int, where: str) -> BuildRefused:
    return _refuse("pull.destination", RefusalClass.USER,
                   f"{where} holds {count} studies, which unpack into one new folder, a folder each below it; nothing "
                   f"was downloaded", "name the new folder: `mime pull <the record> --into <folder>`")


def _bundle_file(source: Source, say: Callable[[str], None] | None, max_bundle_bytes: int, yes: bool = False,
                 force: bool = False, into_given: bool = True) -> tuple[list[Fetched], str | None]:
    """The containers ``source`` names (a record's one bundle or its studies), and, for a DOI, the DOI of the record
    they were read from (:func:`_record_doi`). A record of several studies without ``into_given`` is refused before
    anything is downloaded."""
    if source.kind == "file":
        return [Fetched(Path(source.value), Path(source.value).name)], None
    if assets.offline():
        raise _offline_refusal(redacted(source.value))
    if source.kind == "url":
        return [Fetched(locate_bundle_container(assets.retrieve(source.value, name="bundle", max_bytes=max_bundle_bytes,
                                                          exceeded_action=_RAISE_LIMIT, report=say, yes=yes,
                                                          force=force)),
                        unquote(source.value.split("?", 1)[0].rstrip("/").rpartition("/")[2]))], None
    assets.downloads_doing(f"reading the file list of the record doi:{source.value}")
    files, named, resolved = _record_file(source.value, source.file)
    if named is not None and say is not None:
        say(f"the record: {named}")
    if len(files) > 1 and not into_given:
        raise _several_refused(len(files), f"the record doi:{source.value}")
    fetched: list[Fetched] = []
    for file, checksum, stated in files:
        limit = _limit(stated, max_bundle_bytes)
        fetched.append(Fetched(locate_bundle_container(assets.retrieve(
            f"doi:{source.value}/{file}", name="bundle" if file.endswith(BUNDLE_SUFFIX) else "study",
            known_hash=checksum, max_bytes=limit, exceeded_action=_RAISE_LIMIT, report=say, size=stated, yes=yes,
            force=force)), file))
    return fetched, resolved


def _from_service(service_name: str, record: str, say: Callable[[str], None] | None, max_bundle_bytes: int,
                  access: Access | None = None, yes: bool = False, force: bool = False,
                  into_given: bool = True) -> list[Fetched]:
    """The containers of the published record ``record`` on a service ``mime push`` publishes to (its one bundle, or its
    studies), read by its number at the service's records API, never through doi.org (the sandbox's DOIs do not resolve
    there); with ``access``, let in by the token stored for the service or a share link's key, which go to the
    service's own host alone. A record of several studies without ``into_given`` is refused before anything is
    downloaded."""
    from mime import push
    from mime.push._records import downloader

    service = push.SERVICES[service_name]
    if assets.offline():
        raise _offline_refusal(f"the record {record} on {service.name}")
    assets.downloads_doing(f"reading the record {record} on {service.name}")
    found = push.public_record(service, record, access)
    if say is not None:
        say(f"the record: {found.words()}")
        if found.latest is not None:
            say(f"! it is not the latest version: `mime pull --from {service.name} {quoted(found.latest)}` pulls the latest"
                if found.latest else f"! it is not the latest version: the record's page, {service.site}/records/"
                                     f"{found.record}, names the latest")
    held = found.containers or (push.RecordContainer(found.bundle, found.url, found.md5, found.size),)
    if len(held) > 1 and not into_given:
        raise _several_refused(len(held), f"the record {found.record} on {service.name}")
    return [Fetched(locate_bundle_container(assets.retrieve(
        each.url, name="bundle" if each.key.endswith(BUNDLE_SUFFIX) else "study", known_hash=f"md5:{each.md5}",
        max_bytes=_limit(each.size, max_bundle_bytes), exceeded_action=_RAISE_LIMIT, report=say, size=each.size,
        yes=yes, force=force, fetcher=downloader(service, found.record, access) if access is not None else None)),
        each.key) for each in held]


def _signed_in(service_name: str) -> Access:
    """The token ``mime credentials`` stored for the service, for ``--signed-in``; refused where none is."""
    from mime import credentials, push
    from mime.push._client import Access

    service = push.SERVICES[service_name]
    found = credentials.token(service.credential)
    if found is None:
        raise _refuse("pull.token", RefusalClass.ENVIRONMENT,
                      f"--signed-in sends the token `mime credentials` stored for {service.name}, and none is stored; "
                      f"nothing was sent", f"make a personal access token of the account the record is shared with "
                                           f"on {service.site} and store it, `mime credentials set "
                                           f"{service.credential}`, then pull again")
    return Access(token=found)


def _kept_secret(source: str | os.PathLike[str] | Token) -> str | Token:
    """``source`` as pull holds it: a text that carries an access key (an address's ``token`` parameter: a share link)
    as a :class:`~mime.credentials.Token` from here on, so that no frame below holds the key under a name; anything else
    as its text. It returns before anything can raise with the text in hand."""
    from mime.push._records import SHARE_LINK

    if isinstance(source, Token):
        return source
    return Token(SHARE_LINK, os.fspath(source)) if carries_key(os.fspath(source)) else os.fspath(source)


def _bundle_from(source: str | Token, say: Callable[[str], None] | None, max_bundle_bytes: int,
                 from_service: str | None, signed_in: bool, yes: bool = False,
                 force: bool = False, into_given: bool = True) -> tuple[list[Fetched], str | None]:
    """The containers ``source`` names (one bundle, or one or more studies): a share link's record (let in by its key), a
    record's number on ``from_service`` (signed in where asked), or a DOI, an address or a file; and, for a DOI, the DOI
    of the record they were read from (:func:`_record_doi`). A record of several studies without ``into_given`` is
    refused before anything is downloaded. A source held as a secret (:func:`_kept_secret`) is read as
    a link first, and no refusal quotes the key it carries; one that is not a service's link is an address elsewhere,
    pulled as it was, its key hidden wherever the address is said."""
    from mime.push._client import Access
    from mime.push._records import share_link

    link = share_link(source) if isinstance(source, Token) else None
    if link is not None:
        if signed_in or (from_service is not None and from_service != link.service.name):
            raise _refuse("pull.source", RefusalClass.USER,
                          f"a share link of the record {link.record} on {link.service.name} carries its own key"
                          + (", and is pulled without --signed-in" if signed_in else
                             f", and is pulled from {link.service.name}, not from {from_service}"),
                          "pull the link alone, `mime pull '<the link>'`")
        return _from_service(link.service.name, link.record, say, max_bundle_bytes, Access(key=link.key), yes,
                             force, into_given), None
    if isinstance(source, Token) and (from_service is not None or signed_in):
        raise _refuse("pull.source", RefusalClass.USER,
                      "with --from or --signed-in, pull takes a record's number, and this is an address that carries "
                      "an access key (its token parameter) and is no share link of either service; nothing was sent",
                      "pull the record by its number, `mime pull --from <service> <record>`, or the address alone")
    if from_service is not None:
        assert not isinstance(source, Token)
        return _from_service(from_service, source, say, max_bundle_bytes,
                             _signed_in(from_service) if signed_in else None, yes, force, into_given), None
    if signed_in:
        raise _refuse("pull.source", RefusalClass.USER,
                      "--signed-in sends the token stored for the service --from names, and no --from is given",
                      "pull the record by its number from its service, `mime pull --from zenodo <record> --signed-in`")
    return _bundle_file(parse_source(source.reveal() if isinstance(source, Token) else source), say, max_bundle_bytes,
                        yes, force, into_given)


def packages_needed(lock: Lock) -> list[tuple[str, str, str | None]]:
    """The distributions the lock's entries came from that are not installed, or not at the version that built them:
    each with the version wanted and the one installed, if any."""
    wanted = {entry["distribution"]: entry["distribution_version"] for entry in lock["entries"]}
    needed: list[tuple[str, str, str | None]] = []
    for name, version in sorted(wanted.items()):
        try:
            installed: str | None = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            needed.append((name, version, installed))
    return needed


def adapters_needed(lock: Lock) -> list[tuple[str, str]]:
    """Each location scheme the lock's assets need an adapter for that is not registered here, with the package their
    records name for it (an empty name for an adapter a script registered): listed by ``pull``, never installed. A file
    placed in the cache by its sha256 is found without the adapter."""
    from mime.assets import registered_schemes

    present = registered_schemes()
    needed: dict[str, str] = {}
    for record in lock.get("assets", []):
        for scheme, package in record.get("adapters", {}).items():
            if scheme not in present:
                needed.setdefault(scheme, package)
    return sorted(needed.items())


def _adapters_line(needed: list[tuple[str, str]]) -> str | None:
    if not needed:
        return None
    named = sorted({package for _, package in needed if package})
    schemes = ", ".join(f"{plain(scheme)}: (from {plain(package) if package else 'a script, no package'})"
                        for scheme, package in needed)
    command = f": pip install {' '.join(shlex.quote(each) for each in named)}" if named else ""
    return (f"Adapters: the assets' locations need {schemes}, not installed here{command} (the bundle's author chose "
            f"these: check them first; a file placed in the asset cache by its sha256 is found without them)")


def _packages_line(needed: list[tuple[str, str, str | None]]) -> str:
    """The line that lists the packages to install, from the lock a stranger wrote: the names and versions were held to
    PEP 503 and PEP 440 when the lock was read, and the command quotes them for a shell besides."""
    if not needed:
        return "Packages: all installed"
    names = ", ".join(f"{plain(name)} {plain(version)} needed" + (f" ({plain(installed)} installed)" if installed else "")
                      for name, version, installed in needed)
    command = "pip install " + " ".join(shlex.quote(f"{name}=={version}") for name, version, _ in needed)
    return f"Packages: {names}: {command} (the bundle's author chose these: check them first)"


@dataclass(frozen=True)
class _AssetOutcome:
    cached: int
    downloaded: int
    downloaded_bytes: int
    not_downloaded: int
    stopped: BuildRefused | None = None


def _assets_line(outcome: _AssetOutcome, folder: Path) -> str:
    parts = [text for text in (
        f"{outcome.cached} cached" if outcome.cached else "",
        f"{outcome.downloaded} downloaded ({assets.size_text(outcome.downloaded_bytes)})" if outcome.downloaded else "",
        f"{outcome.not_downloaded} not downloaded: run `{fetch_command(folder=folder)}`"
        if outcome.not_downloaded else "") if text]
    return "Assets: " + (", ".join(parts) if parts else "none referenced")


def _referenced_line(placed: Placed, folder: Path) -> str | None:
    """What became of the files that travel as references (their terms let no bundle carry them, or they are large):
    placed from the cache, or still to fetch; ``None`` where the bundle carries every file. Their licences and credit
    are in the folder's NOTICE (``mime card <folder> --notice``)."""
    total = len(placed.placed) + len(placed.present) + len(placed.missing)
    if not total:
        return None
    line = (f"Referenced files: {total}, not carried in the bundle (their terms are in `mime card "
            f"{folder_argument(folder)} --notice`)")
    if placed.missing:
        return line + f"; {len(placed.missing)} not placed: run `{fetch_command(folder=folder)}`"
    return line + "; each placed from the asset cache"


def _fetch_assets(folder: Path | Sequence[Path], *, wanted: bool, yes: bool, force: bool,
                  accept_licences: Collection[str], confirm: Callable[[str], bool] | None,
                  say: Callable[[str], None] | None) -> _AssetOutcome:
    """The assets of the saved folder ``folder`` (or of each of several, gated once over all of them) fetched into the
    cache, unless not ``wanted``."""
    folders = [folder] if isinstance(folder, Path) else list(folder)
    refs = list({ref.sha256: ref for each in folders for ref in assets.folder_assets(each)}.values())
    absent = [ref for ref in refs if not assets.is_cached(ref)]
    cached = len(refs) - len(absent)
    if not wanted:
        return _AssetOutcome(cached, 0, 0, len(absent))
    total = sum(ref.size for ref in absent)
    if absent and say is not None:
        say(Said(f"Assets to download: {len(absent)} file{'s' if len(absent) != 1 else ''}, "
                 f"{assets.size_text(total)}", PLAN_LINE))
    try:
        # Every asset the lock names is gated, cached or not: the first use of a cached file asks about its licence too.
        gate = assets.approved(refs, accept_licences=accept_licences, yes=yes, force=force, confirm=confirm)
        with gate:
            if not absent:
                return _AssetOutcome(cached, 0, 0, 0)
            assets.announce_downloads([(ref.name, ref.size) for ref in absent])
            stored = 0
            try:
                for ref in refs:
                    assets.fetch(ref, report=say)
                    stored += ref in absent
            except BuildRefused as refused:
                return _AssetOutcome(cached, stored, sum(ref.size for ref in absent[:stored]), len(absent) - stored,
                                     refused)
    except BuildRefused as refused:
        if refused.codes() == ["asset.download-declined"]:  # a "no" to the size question downloads nothing, as before
            return _AssetOutcome(cached, 0, 0, len(absent))
        return _AssetOutcome(cached, 0, 0, len(absent), refused)
    return _AssetOutcome(cached, len(absent), total, 0)


def pull(source: str | os.PathLike[str] | Token, *, into: str | os.PathLike[str] | None = None,
         fetch_assets: bool = True, yes: bool = False, confirm: Callable[[str], bool] | None = None,
         say: Callable[[str], None] | None = None, max_bundle_bytes: int = assets.RETRIEVE_LIMIT_BYTES,
         from_service: str | None = None, signed_in: bool = False, accept_licences: Collection[str] = (),
         force: bool = False) -> Path:
    """Get the bundle ``source`` names (a DOI, an address or a file), check its manifest's hashes, and unpack it into
    the new folder ``into`` (by default ``./<experiment name>/``); then, unless ``fetch_assets`` is false, fetch its
    assets as ``mime assets fetch <folder>`` does. A destination that cannot be unpacked to is refused before anything is
    downloaded; a bundle download stops once it passes the size its record states, or ``max_bundle_bytes`` where none is
    stated. The assets' downloads are gated (:mod:`mime.assets._gates`): an asset under a licence with conditions beyond
    credit is fetched only once its licence is accepted (``accept_licences`` names the ids; a person at a terminal is
    asked), a download over the size MIME asks about (:data:`mime.assets.CONFIRM_ABOVE_BYTES`, or a tenth of the free
    space) only with ``yes`` or when ``confirm`` (asked with a question) says so, and one that would leave the drive
    almost full only with ``force``; ``yes`` and ``force`` never accept a licence. With none of them the folder is
    complete except for the assets and a ``BuildRefused`` says why and the command that fetches them. ``say`` receives
    progress lines and, last, the summary. Installs nothing. With
    ``from_service`` (``zenodo-sandbox`` or ``zenodo``), ``source`` is a published record's number there, read at that
    service's records API (a sandbox DOI never resolves through doi.org); a DOI under the sandbox's test prefix given
    without it is refused (``pull.sandbox-doi``). ``signed_in`` (with ``from_service``) sends the token ``mime
    credentials`` stored for that service, for a record or files shared with its account; without it, pull sends no
    token. A record's share link (``<site>/records/<record>?token=...``) is pulled by the key it carries, which is
    never printed, logged or written, and goes to its service's own host alone. Either travels as push's token does
    (:mod:`mime.push._records`). ``source`` may be a :class:`~mime.credentials.Token` holding the link (as ``mime pull
    --link`` reads it, with no echo), so that no frame holds it as text; a text that carries a key is held so from the
    moment pull takes it. An address elsewhere that carries a ``token`` parameter is pulled as it was, the parameter's
    value hidden wherever the address is said."""
    from mime.saved._studies import STUDY, container_kind

    source = _kept_secret(source)
    if into is not None:
        refused = destination_problem(into)
        if refused is not None:
            raise refused
    fetched, _ = _bundle_from(source, say, max_bundle_bytes, from_service, signed_in, yes, force, into is not None)
    kinds = [container_kind(each.path, named=each.named) for each in fetched]
    if kinds and all(kind is STUDY for kind in kinds):
        return _pull_studies(fetched, into, fetch_assets=fetch_assets, yes=yes, confirm=confirm, say=say,
                             accept_licences=accept_licences, force=force)
    if len(fetched) != 1 or kinds[0] is not BUNDLE:
        raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                      f"the source holds {len(fetched)} containers of more than one kind "
                      f"({', '.join(plain(each.named) for each in fetched)}), and a record holds one bundle, or one or "
                      f"more studies", "pull each file by its name: `mime pull <its file or address>`")
    bundle = fetched[0].path
    destination = Path(into) if into is not None else Path(bundle_name(bundle))
    assets.downloads_doing("unpacking the bundle, and checking each file in it")
    unpacked: Unpacked = unpack(bundle, destination)
    needed = packages_needed(unpacked.lock)
    outcome = _fetch_assets(destination, wanted=fetch_assets, yes=yes, force=force, accept_licences=accept_licences,
                            confirm=confirm, say=say)
    assets.downloads_doing("placing the files that travel as references")
    placed = place_referenced(destination, accept_licences=accept_licences)
    referenced = _referenced_line(placed, destination)
    name, folder = unpacked.manifest["name"], folder_argument(destination)
    if outcome.stopped is None:
        head, tail = f"\N{CHECK MARK} Bundle {name!r} unpacked to {folder}", f"Next: mime verify {folder}"
    else:  # the refusal follows these lines: the summary does not say a thing is done that it says is not
        head = (f"Bundle {name!r} unpacked to {folder}, but not complete: {outcome.not_downloaded} "
                f"asset{'s' if outcome.not_downloaded != 1 else ''} could not be downloaded (the refusal says why)")
        tail = f"Next: once that is dealt with, `{fetch_command(folder=destination)}`, then mime verify {folder}"
    lines = [head,
             "  - Manifest, scene and records: OK",
             f"  - {_assets_line(outcome, destination)}",
             *([f"  - {referenced}"] if referenced is not None else []),
             f"  - {_packages_line(needed)}",
             *([f"  - {adapters}"] if (adapters := _adapters_line(adapters_needed(unpacked.lock))) is not None else []),
             tail]
    if say is not None:
        say(Said("\n".join(lines), DONE if outcome.stopped is None else STOPPED))
    if outcome.stopped is not None:
        raise outcome.stopped
    return destination


def _pull_studies(fetched: Sequence[Fetched], into: str | os.PathLike[str] | None, *, fetch_assets: bool, yes: bool,
                  confirm: Callable[[str], bool] | None, say: Callable[[str], None] | None,
                  accept_licences: Collection[str], force: bool) -> Path:
    """``pull`` of one or more study containers: one into ``into`` (by default ``./<its name>/``), several into a folder
    each below ``into``, written whole or not at all; then their assets fetched as one, gated once over every study's
    locks, and each referenced file placed. Returns the folder written."""
    from mime.saved._studies import study_container_name, unpack_study

    names = [study_container_name(each.path) for each in fetched]
    if len(fetched) > 1 and into is None:
        raise _several_refused(len(fetched), "the source")
    clashing = collisions(names) + [(name, name) for name in sorted(set(names)) if names.count(name) > 1]
    if clashing:
        raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                      f"the record holds two studies named {plain(clashing[0][0])!r} and {plain(clashing[0][1])!r}, "
                      f"which would unpack into one folder", "pull each file by its name, each --into a folder of its "
                                                             "own: `mime pull doi:<doi>/<file>.mimestudy --into <folder>`")
    assets.downloads_doing("unpacking the studies, and checking each file in them")
    if len(fetched) == 1:
        destination = Path(into) if into is not None else Path(names[0])
        unpacked = [unpack_study(fetched[0].path, destination)]
        studies = [destination]
    else:
        assert into is not None
        destination = Path(into)
        refused = destination_problem(destination)
        if refused is not None:
            raise refused
        staging = staging_of(Path(os.path.abspath(destination)))
        staging.mkdir(parents=True)
        try:
            unpacked = [unpack_study(each.path, staging / name) for each, name in zip(fetched, names, strict=True)]
            if destination.exists():
                destination.rmdir()
            staging.rename(destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        studies = [destination / name for name in names]
    folders = [study / where for study, held in zip(studies, unpacked, strict=True) for where in held.locks]
    outcome = _fetch_assets(folders, wanted=fetch_assets, yes=yes, force=force, accept_licences=accept_licences,
                            confirm=confirm, say=say)
    assets.downloads_doing("placing the files that travel as references")
    placed = [place_referenced(each, accept_licences=accept_licences) for each in folders]
    needed = sorted({row for held in unpacked for lock in held.locks.values() for row in packages_needed(lock)})
    adapters = sorted({row for held in unpacked for lock in held.locks.values() for row in adapters_needed(lock)})
    shown = folder_argument(destination)
    what = (f"Study {unpacked[0].manifest['name']!r}" if len(unpacked) == 1 else
            f"{len(unpacked)} studies ({', '.join(repr(held.manifest['name']) for held in unpacked)})")
    missing = [each for each, held in zip(folders, placed, strict=True) if held.missing]
    if outcome.stopped is None:
        head, tail = f"\N{CHECK MARK} {what} unpacked to {shown}", f"Next: mime verify {shown}"
    else:
        head = (f"{what} unpacked to {shown}, but not complete: {outcome.not_downloaded} "
                f"asset{'s' if outcome.not_downloaded != 1 else ''} could not be downloaded (the refusal says why)")
        tail = f"Next: once that is dealt with, `mime assets fetch` in each saved folder, then mime verify {shown}"
    lines = [head, "  - Manifests, group files, figures and saved experiments: OK",
             f"  - {_assets_line(outcome, folders[0]) if len(folders) == 1 else _studies_assets_line(outcome, folders)}",
             *([f"  - Referenced files: {sum(len(held.missing) for held in placed)} not placed: run `mime assets fetch "
                f"<folder>` in {', '.join(folder_argument(each) for each in missing)}"] if missing else []),
             f"  - {_packages_line(needed)}",
             *([f"  - {line}"] if (line := _adapters_line(adapters)) is not None else []),
             tail]
    if say is not None:
        say(Said("\n".join(lines), DONE if outcome.stopped is None else STOPPED))
    if outcome.stopped is not None:
        raise outcome.stopped
    return destination


def _studies_assets_line(outcome: _AssetOutcome, folders: Sequence[Path]) -> str:
    parts = [text for text in (
        f"{outcome.cached} cached" if outcome.cached else "",
        f"{outcome.downloaded} downloaded ({assets.size_text(outcome.downloaded_bytes)})" if outcome.downloaded else "",
        f"{outcome.not_downloaded} not downloaded: run `mime assets fetch <folder>` in each saved folder "
        f"({', '.join(folder_argument(each) for each in folders)})" if outcome.not_downloaded else "") if text]
    return "Assets: " + (", ".join(parts) if parts else "none referenced")


#: The largest lock read out of a bundle to tell whether a folder already holds its experiment.
_LOCK_LIMIT_BYTES = 16_000_000


def _input_hash_of(lock_text: str | bytes) -> str | None:
    """The input hash a lock's text records, read by the one reader of a lock, or ``None``."""
    try:
        found = cast(Any, lock_data(lock_text))["hashes"]["input"]
    except (ValueError, KeyError, TypeError):
        return None
    return found if isinstance(found, str) else None


def _holds_experiment_of(folder: Path, bundle: Path) -> bool:
    """Whether ``folder`` holds the experiment ``bundle`` packs: its lock's input hash is the one the bundle's lock
    records. A folder with no lock that reads, or a bundle whose lock cannot be read, does not."""
    if not folder.is_dir():
        return False
    try:
        held = _input_hash_of((folder / LOCK_FILE).read_bytes())
        with zipfile.ZipFile(bundle) as archive:
            member = archive.getinfo(FOLDER_PREFIX + LOCK_FILE)
            packed = _input_hash_of(archive.read(member)) if member.file_size <= _LOCK_LIMIT_BYTES else None
    except (OSError, KeyError, zipfile.BadZipFile):
        return False
    return held is not None and held == packed


def pulled_parent(source: str | os.PathLike[str], *, accept_licences: Collection[str] = (), yes: bool = False,
                  force: bool = False) -> tuple[Path, str | None]:
    """For ``Experiment.based_on``: the folder a source that is not a folder (a bundle file, an address, a DOI) is
    read from, and, for a DOI, the DOI of the record its bundle was read from (the record's own, as the repository
    states it: a concept DOI resolves to its latest version's). The bundle is fetched as ``mime pull <source>`` fetches
    it, and unpacked as pull unpacks it into ``./<its name>/``. Where that folder already holds the same experiment (its
    lock's input hash is the bundle's: a notebook's cell run again), it is used as it is, nothing unpacked or written
    in it, and the assets its lock names pass the same gates as pull's and are fetched into the cache; a folder there
    that holds anything else is refused as pull refuses it (``pull.destination``).

    A call run again reads the DOI's record (its metadata, which names the bundle's checksum) and downloads no bundle
    bytes: the asset cache keeps a note of which file that checksum named (:func:`mime.assets.retrieve`), and the bundle
    is read from the cache. The folder is still judged by the input hashes, never by the checksum: a folder that holds
    another experiment is refused as before. With the cache cleared, or a new version of the record, the bundle is
    downloaded."""
    from mime.saved._studies import container_kind

    fetched, doi = _bundle_from(_kept_secret(source), None, assets.RETRIEVE_LIMIT_BYTES, None, False, yes, force)
    if len(fetched) != 1 or container_kind(fetched[0].path, named=fetched[0].named) is not BUNDLE:
        raise _refuse("container.kind", RefusalClass.USER,
                      f"Experiment.based_on reads a saved experiment, and {plain(redacted(str(source)))} holds "
                      f"{'studies' if len(fetched) > 1 else 'another kind of container'}",
                      "pull it, `mime pull <source> --into <folder>`, and base on a saved experiment it holds, "
                      "Experiment.based_on(\"<folder>/experiment\")")
    bundle = fetched[0].path
    destination = Path(bundle_name(bundle))
    if not _holds_experiment_of(destination, bundle):
        return pull(bundle, accept_licences=accept_licences, yes=yes, force=force), doi
    outcome = _fetch_assets(destination, wanted=True, yes=yes, force=force, accept_licences=accept_licences,
                            confirm=None, say=None)
    if outcome.stopped is not None:
        raise outcome.stopped
    return destination, doi


__all__ = ["Source", "adapters_needed", "packages_needed", "parse_source", "pull", "pulled_parent"]
