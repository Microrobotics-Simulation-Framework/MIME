"""What push remembers of a draft, in the user's state directory: never in the folder, never in the bundle.

One record per service and experiment name (design §2.6; this changes #171 item 6, which keyed a draft by its bundle's
hash and so left one orphan draft per edit): ``push/<service>/<experiment>/state.json`` under ``platformdirs``' user
state directory, where ``<service>`` is the name of one of the two real services and any other service (a fake, a dry
run) has a namespace of its own (:func:`namespace`). It holds the draft's id, its concept's id, the reserved DOI, the
draft's page, and each file push uploaded and verified, with its size, md5 (the one checksum Zenodo states) and sha256
(which only MIME keeps). The record names its service and that service's API, and a record read for another service is
refused, so a sandbox draft is never taken for a production one. Written whole through a temporary name, so a push
stopped at any point leaves the last record or the one before it, never half of one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import platformdirs

from mime._terminal import plain
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._service import Service

STATE_FORMAT = "mime.push-state/1"
STATE_FILE = "state.json"
#: Where the files a push uploads are staged (the bundle and the NOTICE it packs), beside the record.
STAGED = "staged"
_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


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
    is in flight: found set on the next run, it means the service may hold a draft whose answer never arrived."""

    service: str
    api: str
    experiment: str
    record: str | None = None
    concept: str | None = None
    doi: str | None = None
    page: str | None = None
    creating: bool = False
    files: dict[str, Uploaded] = field(default_factory=lambda: cast(dict[str, Uploaded], {}))


def state_root() -> Path:
    return Path(platformdirs.user_state_dir("mime")) / "push"


def namespace(service: Service) -> str:
    """Where push keeps the records of ``service``: the service's name for one of the two MIME fixes in code
    (``ZENODO``, ``ZENODO_SANDBOX``, equal in every part); for any other (a test's fake, a dry run, whatever its
    name) a directory of its own under ``other/``, named for its name and its API, so no record made against a
    stand-in is ever read or written for a real service."""
    from mime.push._service import ZENODO, ZENODO_SANDBOX

    if service in (ZENODO, ZENODO_SANDBOX):
        return service.name
    digest = hashlib.sha256(f"{service.name}\n{service.api}".encode()).hexdigest()[:12]
    return f"other/{_NAME_UNSAFE.sub('_', service.name)}-{digest}"


def state_directory(service: Service, experiment: str) -> Path:
    """The directory of an experiment's record on ``service``: keyed by both, never by the experiment alone."""
    return state_root() / namespace(service) / (_NAME_UNSAFE.sub("_", experiment) or "experiment")


def _unreadable(path: Path, problem: str) -> BuildRefused:
    return BuildRefused([_refuse(
        "push.state", RefusalClass.ENVIRONMENT, f"push's record {path} {problem}",
        f"look at the draft on the service's site; to start again from no record, delete {path} and run `mime push "
        f"<folder> --to <service>`, which makes a new draft")])


def read_state(service: Service, experiment: str) -> DraftState | None:
    """The record of ``experiment``'s draft on ``service``, or ``None``. A record that is not one, or that names
    another service or API, is refused (``push.state``)."""
    path = state_directory(service, experiment) / STATE_FILE
    if not path.is_file():
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
    try:
        files = {str(key): Uploaded(int(cast(int, value["size"])), str(value["md5"]), str(value["sha256"]))
                 for key, value in cast(dict[str, dict[str, object]], found.get("files", {})).items()}
        return DraftState(str(found["service"]), str(found["api"]), str(found["experiment"]),
                          _text(found.get("record")), _text(found.get("concept")), _text(found.get("doi")),
                          _text(found.get("page")), bool(found.get("creating", False)), files)
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


__all__ = ["STAGED", "STATE_FILE", "STATE_FORMAT", "DraftState", "Uploaded", "clear_state", "namespace", "read_state",
           "staging", "state_directory", "state_root", "write_state"]
