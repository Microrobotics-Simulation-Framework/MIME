#!/usr/bin/env python
"""The sandbox lane: ``mime push``'s drafts against the Zenodo sandbox, run by hand, never by CI (CI never holds an
upload token, design #171 item 5). It is the check the offline fake cannot make of itself: that the sandbox answers
what the fake answers.

    uv run python scripts/zenodo_sandbox_lane.py --creator "<family>, <given>" [--orcid <iD>] [--log <dir>]
        [--folder <saved folder> | --demo toy|terms ...] [--discard | --publish] [--dry-run]

It reads the token ``mime credentials set zenodo-sandbox`` stored (or ``ZENODO_SANDBOX_TOKEN``) through
``mime.credentials`` and nothing else, never prints it, and sends it only to ``https://sandbox.zenodo.org``, a
constant of MIME's (``mime.push.ZENODO_SANDBOX``; nothing here or in the environment can name another host). Requests
are paced a second apart. Nothing is published unless ``--publish`` says so, and then on the sandbox only (production
is never this script's).

For each folder (a copy of one you name, so nothing is written into it, or a demo it saves: ``toy``, the toy
experiment under CC-BY-4.0; ``terms``, the toy with an MIT layer, a layer under a licence of its own that a declared
permission lets travel, and one whose licence forbids redistribution, which travels as a reference), it writes
``sharing.yaml`` where the folder has none (``--creator`` the creator and uploader), then:

1. ``mime push <folder> --to zenodo-sandbox --yes``: the plan, the draft, its DOI, its files;
2. ``--status``: the draft as the sandbox holds it;
3. the push again: every file found already uploaded and verified, nothing sent again;
4. the same three steps against the offline fake (``tests/sharing/zenodo_records.py``), and the two compared: each
   answer's status, content type and the type of every field push reads, request by request;
5. with ``--discard``, the draft deleted; without it, kept, and its page printed, for a person to look at on the
   sandbox's site (the uploads page, ``https://sandbox.zenodo.org/me/uploads``);
6. with ``--publish`` (the sandbox's records only): the draft published (``--confirm`` with its DOI; a declared
   permission is asked in the terminal, as ``mime push --publish`` asks it), the record checked as published (a field
   that reads back otherwise than the draft held it is said, kept among the differences, and the lane goes on), pulled
   by its number into a fresh folder (``mime pull --from zenodo-sandbox <record>``) and verified (``mime verify``);
   for the toy, a new version (the toy saved with one reference step more), pushed and published; each answer's
   shape compared with the fake's, as the drafts' are.

The log (``zenodo_sandbox_lane_<date>.jsonl`` in ``--log``, the current directory by default) holds each answer's
method, path, status, content type, rate-limit headers and field types, never a request's headers; it is searched for
the token in every form ``Token.occurs_in`` knows before it is written, and not written if it holds one. It ends with
the comparison's differences: none is the lane's pass.

**A dry run** (``--dry-run``) sends nothing beyond this machine and needs no token: it holds a fresh one of its own in
its process, so the stored token is never read. The sandbox is replaced by the offline fake behind a server on this
machine, reached through the real transport, and compared with the fake in this process, so its pass line compares
two different paths (the real transport's and the in-process one's), never the sandbox: it says so. It writes its log
only where ``--log`` names, and removes what else it wrote (the folders, the stand-ins' records in the state
directory), as every run removes its stand-ins'.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tests" / "sharing")]

from mime import credentials, push  # noqa: E402
from mime.build import BuildRefused  # noqa: E402
from mime.credentials import Token  # noqa: E402
from mime.push import ZENODO_SANDBOX, Answer, HttpTransport  # noqa: E402
from mime.push._service import loopback  # noqa: E402
from mime.push._state import namespace, state_root  # noqa: E402

PAUSE = 1.0
#: The fields push reads, by where they sit in an answer: their JSON types are what the lane compares.
DRAFT_FIELDS = ("id", "parent", "parent.id", "pids", "pids.doi.identifier", "is_published", "status",
                "links.self_html", "metadata.title", "metadata.creators", "metadata.rights", "metadata.identifiers",
                "errors", "conceptrecid")
ENTRY_FIELDS = ("key", "status", "size", "checksum", "errors")
_NUMBER = re.compile(r"/\d+(?=/|$)")


def _kind(value: object) -> str:
    return {dict: "object", list: "array", str: "string", int: "integer", float: "number", bool: "boolean",
            type(None): "null"}[type(value)]


def _at(held: object, path: str) -> object:
    for part in path.split("."):
        if not isinstance(held, dict) or part not in held:
            return KeyError
        held = held[part]
    return held


def shape(method: str, url: str, answer: Answer) -> dict[str, object]:
    """An answer as the lane compares it: no value but the status, the checksum's prefix and the DOI's prefix."""
    path = _NUMBER.sub("/<id>", url.split("/api", 1)[-1])
    found: dict[str, object] = {"method": method, "path": path, "status": answer.status,
                                "content_type": answer.headers.get("content-type", "").split(";")[0],
                                "rate_limit": sorted(h for h in answer.headers if h.startswith("x-ratelimit-")
                                                     or h == "retry-after")}
    try:
        held = json.loads(answer.body) if answer.body.strip() else None
    except ValueError:
        return {**found, "body": "not JSON"}
    if isinstance(held, dict) and "entries" in held:
        found["entries"] = [{name: "absent" if (v := _at(e, name)) is KeyError else _kind(v) for name in ENTRY_FIELDS}
                            for e in held["entries"]]
    elif isinstance(held, dict) and "key" in held:
        found["fields"] = {name: "absent" if (v := _at(held, name)) is KeyError else _kind(v) for name in ENTRY_FIELDS}
        found["checksum_prefix"] = str(held.get("checksum", ""))[:4]
    elif isinstance(held, dict) and answer.status < 300:
        found["fields"] = {name: "absent" if (v := _at(held, name)) is KeyError else _kind(v) for name in DRAFT_FIELDS}
        doi = _at(held, "pids.doi.identifier")
        found["doi_prefix"] = doi[:8] if isinstance(doi, str) else None
    elif isinstance(held, dict):
        found["fields"] = {name: _kind(held.get(name)) for name in ("status", "message")}
    return found


class Recording:
    """A transport that keeps each answer's shape (and the draft's id, which the report names), never a request's
    headers."""

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.shapes: list[dict[str, object]] = []
        #: Each stored file's id, by record and key, from a published record's listing and a version's import: whether
        #: a file linked into a new version is stored once (the same id) is a question only the service answers.
        self.stored: dict[tuple[str, str], object] = {}
        self.imported: dict[tuple[str, str], object] = {}

    def send(self, method: str, url: str, headers: Mapping[str, str], body: bytes | Path | None,
             token: Token) -> Answer:
        answer = self.inner.send(method, url, headers, body, token)  # type: ignore[attr-defined]
        self.shapes.append(shape(method, url, answer))
        found = re.search(r"/records/(\d+)/(files|draft/actions/files-import)$", url)
        if found is not None and answer.status < 300:
            try:
                entries = json.loads(answer.body).get("entries", [])
            except (ValueError, AttributeError):
                entries = []
            held = self.stored if found.group(2) == "files" else self.imported
            for entry in entries if isinstance(entries, list) else []:
                if isinstance(entry, dict) and "key" in entry:
                    held[(found.group(1), str(entry["key"]))] = entry.get("file_id")
        return answer

    def stored_once(self) -> dict[str, bool]:
        """For each file imported into a new version, whether its stored id is the published version's."""
        found: dict[str, bool] = {}
        for (record, key), value in self.imported.items():
            before = [held for (other, name), held in self.stored.items() if name == key and other != record]
            found[key] = value is not None and bool(before) and before[-1] == value
        return found


def _toy_demo(root: Path, steps: int | None = None) -> Path:
    from mime.testing.toys import full_toy

    experiment = full_toy()
    experiment.licence("CC-BY-4.0")
    if steps is None:
        experiment.save(root / "mime_sandbox_toy")
    else:
        experiment.save(root / "mime_sandbox_toy", steps=steps)
    return root / "mime_sandbox_toy"


def _terms_demo(root: Path, uploader: str) -> Path:
    from terms_cases import NO_REDISTRIBUTION, layered, write_mesh

    from mime.terms import CustomLicence, Evidence, Permission, Scope, Terms

    own = CustomLicence("LicenseRef-Example-Maker-1.0", "Example Maker Licence 1.0",
                        "https://example.org/example-maker/LICENSE.txt")
    kinematics = write_mesh(root / "made", "arm_kinematics", 1200)
    granted = write_mesh(root / "made", "arm_shell", 2400)
    visual = write_mesh(root / "made", "arm_visual", 6000)
    layers = [(kinematics, Terms("MIT", "Copyright (c) 2021 A. Driver (synthetic)", "results")),
              (granted, Terms(own, "Example Maker Ltd (synthetic)", "presentation")),
              (visual, Terms(NO_REDISTRIBUTION, "Robot Maker Ltd (synthetic)", "presentation",
                             ("https://example.org/robot-maker/6a3ebb1/arm_visual.usda",)))]
    import hashlib

    permission = Permission("Example Maker Ltd (synthetic)", uploader,
                            (hashlib.sha256(granted.read_bytes()).hexdigest(),), Scope(True, False, False),
                            "2026-11-01", Evidence("public_statement", "https://example.org/example-maker/permission"))
    layered("mime_sandbox_terms", layers, licence="CC-BY-4.0", permissions=[permission]).build().save(
        root / "mime_sandbox_terms")
    return root / "mime_sandbox_terms"


def write_sharing(folder: Path, creator: str, orcid: str | None, title: str) -> None:
    path = folder / push.SHARING_FILE
    if path.exists():
        return
    push.init(folder)
    family, _, given = (part.strip() for part in creator.partition(","))
    person = f"{{family: {json.dumps(family)}" + (f", given: {json.dumps(given)}" if given else "") + \
        (f", orcid: {orcid}" if orcid else "") + "}"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^title: .*$", f"title: {json.dumps(title)}", text)
    text = text.replace("creators: []", f"creators: [{person}]")
    text = text.replace("# uploader: {name: \"<family name>, <given name>\", institution: <your institution>}",
                        f"uploader: {{name: {json.dumps(creator)}}}")
    path.write_text(text, encoding="utf-8")


def _declare(question: str) -> bool:
    """A declared permission asked in the terminal, as ``mime push --publish`` asks it; no terminal declares none."""
    if not sys.stdin.isatty():
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def _published(folder: Path, service: push.Service, doi: str, declare: object, transport: object, pause: float,
               noted: list[str]) -> None:
    """A publish. A field that reads back from the published record otherwise than the draft held it is what this lane
    is here to find: noted, and the lane goes on (the record is published and recorded)."""
    try:
        push.publish(folder, service, confirm=doi, ask=declare, say=print, transport=transport,  # type: ignore[arg-type]
                     pause=pause)
    except BuildRefused as refused:
        if refused.codes() != ["push.metadata-read-back"]:
            raise
        print(f"read back otherwise: {refused.refusals[0].message}")
        noted.append(refused.refusals[0].message)


def cycle(folder: Path, service: push.Service, transport: object, discard: bool, publish: bool = False,
          declare: object = _declare, noted: list[str] | None = None) -> push.Drafted | None:
    """Push, read the draft, push again (nothing sent again), and discard or publish where asked: a second apart on
    the sandbox, at once against a fake. Publishing, the toy's next version is pushed and published too. What reads back
    otherwise from a published record is added to ``noted``."""
    read_back = noted if noted is not None else []
    remote = isinstance(getattr(transport, "inner", transport), HttpTransport) and not loopback(service.api)
    pause = PAUSE if remote else 0.0
    first = push.push(folder, service, yes=True, say=print, transport=transport, pause=pause)  # type: ignore[arg-type]
    push.status(folder, service, say=print, transport=transport, pause=pause)  # type: ignore[arg-type]
    again = push.push(folder, service, yes=True, say=print, transport=transport, pause=pause)  # type: ignore[arg-type]
    assert again is not None and not again.uploaded, f"the second push sent {again and again.uploaded} again"
    if discard:
        push.discard(folder, service, yes=True, say=print, transport=transport, pause=pause)  # type: ignore[arg-type]
    if publish:
        assert service.name != push.ZENODO.name and not service.production, "the lane publishes on the sandbox only"
        _published(folder, service, first.doi, declare, transport, pause, read_back)
        if folder.name == "mime_sandbox_toy":
            _toy_demo(folder.parent, steps=9)
            later = push.push(folder, service, say=print, transport=transport, pause=pause)  # type: ignore[arg-type]
            _published(folder, service, later.doi, declare, transport, pause, read_back)
    return first


def pulled(record: str, work: Path) -> int:
    """The published sandbox record pulled by its number into a fresh folder, and verified: ``mime verify``'s exit."""
    from mime.cli import main as mime

    into = work / "pulled" / record
    code = mime(["pull", "--from", ZENODO_SANDBOX.name, record, "--into", str(into)])
    return code if code else mime(["verify", str(into)])


def differences(sandbox: list[dict[str, object]], fake: list[dict[str, object]]) -> list[str]:
    found: list[str] = []
    if len(sandbox) != len(fake):
        found.append(f"{len(sandbox)} requests to the sandbox, {len(fake)} to the fake")
    for index, (real, made) in enumerate(zip(sandbox, fake, strict=False)):
        for key in sorted(set(real) | set(made)):
            if key == "rate_limit" and real.get("status") != 204:
                continue
            if real.get(key) != made.get(key):
                found.append(f"request {index + 1} ({real.get('method')} {real.get('path')}): {key} is "
                             f"{real.get(key)!r} on the sandbox and {made.get(key)!r} on the fake")
    return found


@contextlib.contextmanager
def sandbox_side(dry_run: bool, token: Token) -> Iterator[tuple[push.Service, object]]:
    """What the lane pushes to: the sandbox itself, over HTTP; or, for a dry run, the fake behind a server on this
    machine, through the same transport, with an address, a name and so a namespace of push's records of its own
    (``mime.push._state.namespace``), which never reads or writes the sandbox's records."""
    if not dry_run:
        yield ZENODO_SANDBOX, HttpTransport()
        return
    from zenodo_records import served

    with served(token.reveal()) as fake:
        yield fake.service("zenodo-sandbox-dry-run", credential=ZENODO_SANDBOX.credential), HttpTransport()


@contextlib.contextmanager
def dry_token(dry_run: bool) -> Iterator[None]:
    """For a dry run, a fresh token of the lane's own as this process's sandbox token, so the stored one is never
    read; the variable as it was afterwards."""
    if not dry_run:
        yield
        return
    variable = credentials.SERVICES[ZENODO_SANDBOX.credential]
    held = os.environ.get(variable)
    os.environ[variable] = secrets.token_urlsafe(30)[:40]
    try:
        yield
    finally:
        if held is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = held


def remove_records(service: push.Service) -> None:
    """A stand-in's records in the state directory, and the directories they leave empty; never a real service's."""
    if namespace(service) in (ZENODO_SANDBOX.name, push.ZENODO.name):
        return
    shutil.rmtree(state_root() / namespace(service), ignore_errors=True)
    for directory in ((state_root() / namespace(service)).parent, state_root(), state_root().parent):
        with contextlib.suppress(OSError):
            directory.rmdir()


def write_log(lines: list[dict[str, object]], token: Token, directory: Path) -> Path | None:
    """The log, searched for the token in every form ``Token.occurs_in`` knows first, and not written if it holds one."""
    text = "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines).encode("utf-8")
    if token.occurs_in(text):
        print("the log would hold the token: not written", file=sys.stderr)
        return None
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / f"zenodo_sandbox_lane_{datetime.date.today().isoformat()}.jsonl"
    log.write_bytes(text)
    print(f"wrote {log}")
    return log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--creator", required=True, help='the creator and uploader, "<family>, <given>"')
    parser.add_argument("--orcid", help="the creator's ORCID iD")
    parser.add_argument("--folder", type=Path, action="append", default=[], help="a saved folder to push (repeat)")
    parser.add_argument("--demo", choices=("toy", "terms"), action="append", default=[], help="a demo folder to save "
                                                                                            "and push (repeat)")
    parser.add_argument("--log", type=Path, help="the directory to write the log to (the current directory by "
                                                 "default; a dry run writes one only where this names)")
    ending = parser.add_mutually_exclusive_group()
    ending.add_argument("--discard", action="store_true", help="delete each draft at the end")
    ending.add_argument("--publish", action="store_true", help="publish each draft on the sandbox, pull and verify it, "
                                                               "and publish the toy's next version")
    parser.add_argument("--dry-run", action="store_true", help="the sandbox replaced by the fake behind a server on "
                                                               "this machine, through the real transport: no request "
                                                               "leaves the machine, no token is read, and no record "
                                                               "of the sandbox's is touched")
    arguments = parser.parse_args(argv)
    with dry_token(arguments.dry_run):
        return lane(arguments)


def lane(arguments: argparse.Namespace) -> int:
    token = credentials.token(ZENODO_SANDBOX.credential)
    if token is None:
        print("no zenodo-sandbox token: `mime credentials set zenodo-sandbox`", file=sys.stderr)
        return 2
    log = arguments.log if arguments.log is not None else None if arguments.dry_run else Path.cwd()
    work = Path(tempfile.mkdtemp(prefix="mime-sandbox-lane-"))
    stand_ins: list[push.Service] = []
    try:
        folders = [Path(shutil.copytree(folder, work / "given" / f"{index}" / folder.name))
                   for index, folder in enumerate(arguments.folder)]
        for kind in arguments.demo or ([] if folders else ["toy"]):
            folders.append(_toy_demo(work) if kind == "toy" else _terms_demo(work, arguments.creator))
        found = compared(arguments, folders, token, log, stand_ins)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        for service in stand_ins:
            remove_records(service)
    if found is None:
        return 3
    if not found:
        print("the fake behind the real transport answers as the fake in this process does (a dry run: the sandbox "
              "was not asked, and this compared no answer of it)" if arguments.dry_run else
              "the fake answers as the sandbox does")
        return 0
    print("differences:\n  " + "\n  ".join(found))
    return 1


def compared(arguments: argparse.Namespace, folders: list[Path], token: Token, log: Path | None,
             stand_ins: list[push.Service]) -> list[str] | None:
    """Each folder's cycle on the lane's side and on the fake in this process, and their differences; ``None`` when
    the log would have held the token."""
    from zenodo_records import FakeRecords, FakeTransport

    lines: list[dict[str, object]] = []
    found: list[str] = []
    try:
        for folder in folders:
            write_sharing(folder, arguments.creator, arguments.orcid,
                          f"{folder.name} (MIME push sandbox lane, {datetime.date.today().isoformat()})")
            with sandbox_side(arguments.dry_run, token) as (service, transport):
                stand_ins.append(service)
                real = Recording(transport)
                start = folder.parent / f"{folder.name}.as-saved"
                shutil.copytree(folder, start)
                read_back: list[str] = []
                try:
                    drafted = cycle(folder, service, real, arguments.discard, arguments.publish,
                                    declare=(lambda question: True) if arguments.dry_run else _declare,
                                    noted=read_back)
                finally:
                    found += [f"{folder.name}: published, and read back otherwise: {each}" for each in read_back]
                    lines += [{"folder": folder.name, "service": service.name, **each} for each in real.shapes]
                    if real.imported:
                        lines.append({"folder": folder.name, "service": service.name,
                                      "linked_file_stored_once": real.stored_once()})
                        print(f"files linked into the new version, each stored once (its file id the published "
                              f"version's): {real.stored_once()}")
                if arguments.publish and not arguments.dry_run and drafted is not None:
                    code = pulled(drafted.record, folder.parent)
                    lines.append({"folder": folder.name, "pulled": drafted.record, "verify_exit": code})
                    if code:
                        found.append(f"{folder.name}: the record {drafted.record}, pulled, did not verify ({code})")
            shutil.rmtree(folder)
            shutil.move(start, folder)
            fake = FakeRecords(token)
            made = Recording(FakeTransport(fake))
            stand_in = fake.service("zenodo-sandbox-fake", credential=ZENODO_SANDBOX.credential)
            stand_ins.append(stand_in)
            cycle(folder, stand_in, made, arguments.discard, arguments.publish, declare=lambda question: True)
            if not arguments.discard and not arguments.publish:
                push.discard(folder, stand_in, yes=True, transport=FakeTransport(fake), pause=0)
            found += [f"{folder.name}: {each}" for each in differences(real.shapes, made.shapes)]
            lines += [{"folder": folder.name, "service": "fake", **each} for each in made.shapes]
            if drafted is not None:
                lines.append({"folder": folder.name, "service": service.name, "draft": drafted.record,
                              "page": drafted.page, "doi": drafted.doi, "kept": not arguments.discard})
    except BuildRefused as refused:
        lines.append({"stopped": [f"{r.code} [{r.category.value}]: {r.message}" for r in refused.refusals]})
        if log is not None:
            write_log(lines, token, log)
        raise
    lines.append({"differences": found})
    if log is not None and write_log(lines, token, log) is None:
        return None
    return found


if __name__ == "__main__":
    sys.exit(main())
