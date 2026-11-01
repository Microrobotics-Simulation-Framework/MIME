"""What push remembers of a draft, in the user's state directory: never in the folder, never in the bundle.

One record per service and experiment name (design §2.6; this changes #171 item 6, which keyed a draft by its bundle's
hash and so left one orphan draft per edit): ``push/<service>/<experiment>-<hash>/state.json`` under ``platformdirs``'
user state directory, where ``<service>`` is the name of one of the two real services and any other service (a fake, a
dry run) has a namespace of its own (:func:`namespace`), and ``<hash>`` is of the experiment's exact name, so two names
that differ only in case have two records on a file system that folds case (macOS's and Windows' by default). It holds
the draft's id, its concept's id, the reserved DOI, the draft's page, push's mark of the draft (:func:`draft_identifier`
is written into the draft as an alternate identifier, so push knows the draft it reads is the one it made), what the
draft's metadata was when push last wrote it (:func:`metadata_marks`, so an edit made on the site is never overwritten
unasked), and each file push uploaded and verified, with its size, md5 (the one checksum Zenodo states) and sha256
(which only MIME keeps). The record names its service, that service's API and the experiment, and a record read for
another is refused, so a sandbox draft is never taken for a production one, nor one experiment's for another's. Written
whole through a temporary name, so a push stopped at any point leaves the last record or the one before it, never half
of one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import platformdirs

from mime._terminal import plain
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._service import Service

STATE_FORMAT = "mime.push-state/2"
#: The record before push marked its drafts and kept their metadata: read only to refuse it, by name.
OLD_FORMAT = "mime.push-state/1"
STATE_FILE = "state.json"
#: Where the files a push uploads are staged (the bundle and the NOTICE it packs), beside the record.
STAGED = "staged"
#: The scheme of push's mark in a draft's alternate identifiers.
MARK_SCHEME = "urn"
_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
#: The namespace of push's marks (a UUID 5 of MIME's PyPI address): the same service, experiment and mark always give
#: the same identifier, and nothing else does.
_MARKS = uuid.uuid5(uuid.NAMESPACE_URL, "https://pypi.org/project/mime-engine/push")


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> Refusal:
    return Refusal(code, category, message, action)


@dataclass(frozen=True)
class Uploaded:
    """A file push uploaded into the draft and verified there: its size and md5 as the service states them, equal to
    MIME's, and its sha256, MIME's own."""

    size: int
    md5: str
    sha256: str


@dataclass
class DraftState:
    """What push knows of an experiment's draft on one service. ``creating`` is set while a request to make the draft
    is in flight: found set on the next run, it means the service may hold a draft whose answer never arrived. ``mark``
    is the random part of the identifier push writes into the draft (:func:`draft_identifier`); ``written`` is the
    draft's metadata as the service held it after push last wrote it (:func:`metadata_marks`)."""

    service: str
    api: str
    experiment: str
    record: str | None = None
    concept: str | None = None
    doi: str | None = None
    page: str | None = None
    creating: bool = False
    files: dict[str, Uploaded] = field(default_factory=lambda: cast(dict[str, Uploaded], {}))
    mark: str = field(default_factory=lambda: secrets.token_hex(16))
    written: dict[str, str] = field(default_factory=lambda: cast(dict[str, str], {}))


def state_root() -> Path:
    return Path(platformdirs.user_state_dir("mime")) / "push"


def _real(service: Service) -> bool:
    """Whether ``service`` is one of the two MIME fixes in code, in every part (compared by value, so a copy of the
    class made by a reload is still recognised, and a stand-in that differs in any part is not)."""
    from mime.push._service import ZENODO, ZENODO_SANDBOX

    def parts(held: Service) -> tuple[object, ...]:
        return held.name, held.api, held.site, held.credential, held.production, held.words

    return parts(service) in (parts(ZENODO), parts(ZENODO_SANDBOX))


def namespace(service: Service) -> str:
    """Where push keeps the records of ``service``: the service's name for one of the two MIME fixes in code
    (``ZENODO``, ``ZENODO_SANDBOX``, equal in every part); for any other (a test's fake, a dry run, whatever its
    name) a directory of its own under ``other/``, named for its name and its API, so no record made against a
    stand-in is ever read or written for a real service."""
    if _real(service):
        return service.name
    digest = hashlib.sha256(f"{service.name}\n{service.api}".encode()).hexdigest()[:12]
    return f"other/{_NAME_UNSAFE.sub('_', service.name)}-{digest}"


def _folded(experiment: str) -> str:
    return _NAME_UNSAFE.sub("_", experiment) or "experiment"


def state_directory(service: Service, experiment: str) -> Path:
    """The directory of an experiment's record on ``service``: keyed by both, never by the experiment alone, and by the
    experiment's exact name (its hash), so ``Toy`` and ``toy`` have two even where the file system folds case."""
    digest = hashlib.sha256(experiment.encode("utf-8")).hexdigest()[:12]
    return state_root() / namespace(service) / f"{_folded(experiment)}-{digest}"


def draft_identifier(service: Service, experiment: str, mark: str) -> str:
    """The alternate identifier push writes into the draft it makes for ``experiment`` on ``service``: a UUID of the
    three, ``urn:uuid:...``. A record that names another experiment's draft, or another service's, or that was edited,
    finds an identifier in that draft that its own three do not give, and push changes nothing there."""
    return f"urn:uuid:{uuid.uuid5(_MARKS, f'{service.name}\n{service.api}\n{experiment}\n{mark}')}"


def _filled(value: object) -> bool:
    return value not in (None, "", [], {})


def metadata_marks(draft: Mapping[str, object]) -> dict[str, str]:
    """Each field of the draft's metadata as the service holds it, as the sha256 of its canonical JSON: what push
    compares, on its next run, with the draft then, to find what was edited on the site. Taken from the service's own
    answer, not from what push sent, since the service adds to what it is sent (a licence's title, a person's name). A
    field left empty is left out (the site's form may save one empty)."""
    metadata = draft.get("metadata")
    held = cast(dict[str, object], metadata) if isinstance(metadata, dict) else {}
    return {key: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=False).encode("utf-8")).hexdigest()
            for key, value in sorted(held.items()) if _filled(value)}


def _unreadable(path: Path, problem: str) -> BuildRefused:
    return BuildRefused([_refuse(
        "push.state", RefusalClass.ENVIRONMENT, f"push's record {path} {problem}",
        f"look at the draft on the service's site; to start again from no record, delete {path} and run `mime push "
        f"<folder> --to <service>`, which makes a new draft")])


def _old_record(service: Service, experiment: str) -> Path:
    """Where push's first records were kept, by the experiment's folded name alone."""
    return state_root() / namespace(service) / _folded(experiment) / STATE_FILE


def read_state(service: Service, experiment: str) -> DraftState | None:
    """The record of ``experiment``'s draft on ``service``, or ``None``. A record that is not one, that names another
    service, API or experiment (case included), or that was written before push marked its drafts, is refused
    (``push.state``)."""
    path = state_directory(service, experiment) / STATE_FILE
    if not path.is_file():
        old = _old_record(service, experiment)
        if old.is_file():
            raise BuildRefused([_refuse(
                "push.state", RefusalClass.ENVIRONMENT,
                f"push's record {old} is of {OLD_FORMAT}, written before push marked the drafts it makes, so push "
                f"cannot tell whether the draft it names is this experiment's",
                f"look at the draft on the service's site and delete it there if it is not wanted; then delete "
                f"{old.parent} and run `mime push <folder> --to <service>`, which makes a new draft")])
        return None
    try:
        held: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _unreadable(path, f"cannot be read ({type(error).__name__})") from None
    if not isinstance(held, dict) or cast(dict[str, object], held).get("format") != STATE_FORMAT:
        raise _unreadable(path, f"is not a record of {STATE_FORMAT}")
    found = cast(dict[str, object], held)
    if (found.get("service"), found.get("api")) != (service.name, service.api):
        raise _unreadable(path, f"is the record of a draft on {plain(str(found.get('service')))} "
                                f"({plain(str(found.get('api')))}), not on {service.name} ({service.api}): push "
                                f"never takes one service's draft for another's")
    if found.get("experiment") != experiment:
        raise _unreadable(path, f"is the record of the experiment {plain(repr(found.get('experiment')))}, not "
                                f"{plain(repr(experiment))}: push never takes one experiment's draft for another's")
    try:
        files = {str(key): Uploaded(int(cast(int, value["size"])), str(value["md5"]), str(value["sha256"]))
                 for key, value in cast(dict[str, dict[str, object]], found.get("files", {})).items()}
        written = {str(key): str(value) for key, value in cast(dict[str, object], found.get("written", {})).items()}
        return DraftState(str(found["service"]), str(found["api"]), str(found["experiment"]),
                          _text(found.get("record")), _text(found.get("concept")), _text(found.get("doi")),
                          _text(found.get("page")), bool(found.get("creating", False)), files, str(found["mark"]),
                          written)
    except (KeyError, TypeError, ValueError, AttributeError):
        raise _unreadable(path, "is missing a part or holds one of another type") from None


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def write_state(service: Service, state: DraftState) -> Path:
    directory = state_directory(service, state.experiment)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / STATE_FILE
    held = {"format": STATE_FORMAT, **asdict(state)}
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
    try:
        temporary.write_text(json.dumps(held, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def clear_state(service: Service, experiment: str) -> None:
    directory = state_directory(service, experiment)
    (directory / STATE_FILE).unlink(missing_ok=True)
    shutil.rmtree(directory / STAGED, ignore_errors=True)


def staging(service: Service, experiment: str) -> Path:
    """An empty directory for the files this push stages, emptied of an earlier push's."""
    directory = state_directory(service, experiment) / STAGED
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    return directory


__all__ = ["MARK_SCHEME", "OLD_FORMAT", "STAGED", "STATE_FILE", "STATE_FORMAT", "DraftState", "Uploaded", "clear_state",
           "draft_identifier", "metadata_marks", "namespace", "read_state", "staging", "state_directory", "state_root",
           "write_state"]
