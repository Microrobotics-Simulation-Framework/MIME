#!/usr/bin/env python
"""The sandbox lane: ``mime push``'s drafts against the Zenodo sandbox, run by hand, never by CI (CI never holds an
upload token, design #171 item 5). It is the check the offline fake cannot make of itself: that the sandbox answers
what the fake answers.

    uv run python scripts/zenodo_sandbox_lane.py --creator "<family>, <given>" [--orcid <iD>] [--log <dir>]
        [--folder <saved folder> | --demo toy|terms ...] [--discard]

It reads the token ``mime credentials set zenodo-sandbox`` stored (or ``ZENODO_SANDBOX_TOKEN``) through
``mime.credentials`` and nothing else, never prints it, and sends it only to ``https://sandbox.zenodo.org``, a
constant of MIME's (``mime.push.ZENODO_SANDBOX``; nothing here or in the environment can name another host). Requests
are paced a second apart. Nothing is published: push has no publish.

For each folder (one you name, or a demo it saves: ``toy``, the toy experiment under CC-BY-4.0; ``terms``, the toy with
an MIT layer, a layer under a licence of its own that a declared permission lets travel, and one whose licence forbids
redistribution, which travels as a reference), it writes ``sharing.yaml`` where the folder has none (``--creator`` the
creator and uploader), then:

1. ``mime push <folder> --to zenodo-sandbox --yes``: the plan, the draft, its DOI, its files;
2. ``--status``: the draft as the sandbox holds it;
3. the push again: every file found already uploaded and verified, nothing sent again;
4. the same three steps against the offline fake (``tests/sharing/zenodo_records.py``), and the two compared: each
   answer's status, content type and the type of every field push reads, request by request;
5. with ``--discard``, the draft deleted; without it, kept, and its page printed, for a person to look at on the
   sandbox's site (the uploads page, ``https://sandbox.zenodo.org/me/uploads``).

The log (``zenodo_sandbox_lane_<date>.jsonl`` in ``--log``, the current directory by default) holds each answer's
method, path, status, content type, rate-limit headers and field types, never a request's headers; it is searched for
the token in every form ``Token.occurs_in`` knows before it is written, and not written if it holds one. It ends with
the comparison's differences: none is the lane's pass.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tests" / "sharing")]

from mime import credentials, push  # noqa: E402
from mime.build import BuildRefused  # noqa: E402
from mime.credentials import Token  # noqa: E402
from mime.push import ZENODO_SANDBOX, Answer, HttpTransport  # noqa: E402

PAUSE = 1.0
#: The fields push reads, by where they sit in an answer: their JSON types are what the lane compares.
DRAFT_FIELDS = ("id", "parent", "parent.id", "pids", "pids.doi.identifier", "is_published", "status",
                "links.self_html", "metadata.title", "metadata.creators", "metadata.rights", "errors", "conceptrecid")
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

    def send(self, method: str, url: str, headers: Mapping[str, str], body: bytes | Path | None,
             token: Token) -> Answer:
        answer = self.inner.send(method, url, headers, body, token)  # type: ignore[attr-defined]
        self.shapes.append(shape(method, url, answer))
        return answer


def _toy_demo(root: Path) -> Path:
    from mime.testing.toys import full_toy

    experiment = full_toy()
    experiment.licence("CC-BY-4.0")
    experiment.save(root / "mime_sandbox_toy")
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


def cycle(folder: Path, service: push.Service, transport: object, discard: bool) -> push.Drafted | None:
    """Push, read the draft, push again (nothing sent again), and discard where asked: a second apart on the sandbox,
    at once against a fake."""
    pause = PAUSE if isinstance(getattr(transport, "inner", transport), HttpTransport) else 0.0
    first = push.push(folder, service, yes=True, say=print, transport=transport, pause=pause)  # type: ignore[arg-type]
    push.status(folder, service, say=print, transport=transport, pause=pause)  # type: ignore[arg-type]
    again = push.push(folder, service, yes=True, say=print, transport=transport, pause=pause)  # type: ignore[arg-type]
    assert again is not None and not again.uploaded, f"the second push sent {again and again.uploaded} again"
    if discard:
        push.discard(folder, service, yes=True, say=print, transport=transport, pause=pause)  # type: ignore[arg-type]
    return first


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


def sandbox_side(dry_run: bool, token: Token) -> tuple[push.Service, object]:
    """What the lane pushes to: the sandbox itself, over HTTP; or, for a dry run, a fake with an address, a name and so
    a namespace of push's records of its own (``mime.push._state.namespace``), which never reads or writes the
    sandbox's records."""
    if not dry_run:
        return ZENODO_SANDBOX, HttpTransport()
    from zenodo_records import FakeRecords, FakeTransport

    fake = FakeRecords(token)
    return fake.service("zenodo-sandbox-dry-run", credential=ZENODO_SANDBOX.credential), FakeTransport(fake)


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
    parser.add_argument("--log", type=Path, default=Path.cwd(), help="the directory to write the log to")
    parser.add_argument("--discard", action="store_true", help="delete each draft at the end")
    parser.add_argument("--dry-run", action="store_true", help="the sandbox replaced by a fake of its own name and "
                                                               "address, in this process: no request leaves the "
                                                               "machine, and no record of the sandbox's is touched")
    arguments = parser.parse_args(argv)
    token = credentials.token(ZENODO_SANDBOX.credential)
    if token is None:
        print("no zenodo-sandbox token: `mime credentials set zenodo-sandbox`", file=sys.stderr)
        return 2
    work = Path(tempfile.mkdtemp(prefix="mime-sandbox-lane-"))
    folders = list(arguments.folder)
    for kind in arguments.demo or ([] if folders else ["toy"]):
        folders.append(_toy_demo(work) if kind == "toy" else _terms_demo(work, arguments.creator))
    from zenodo_records import FakeRecords, FakeTransport

    lines: list[dict[str, object]] = []
    found: list[str] = []
    try:
        for folder in folders:
            write_sharing(folder, arguments.creator, arguments.orcid,
                          f"{folder.name} (MIME push sandbox lane, {datetime.date.today().isoformat()})")
            service, transport = sandbox_side(arguments.dry_run, token)
            real = Recording(transport)
            try:
                drafted = cycle(folder, service, real, arguments.discard)
            finally:
                lines += [{"folder": folder.name, "service": service.name, **each} for each in real.shapes]
            fake = FakeRecords(token)
            made = Recording(FakeTransport(fake))
            stand_in = fake.service("zenodo-sandbox-fake", credential=ZENODO_SANDBOX.credential)
            cycle(folder, stand_in, made, arguments.discard)
            if not arguments.discard:
                push.discard(folder, stand_in, yes=True, transport=FakeTransport(fake), pause=0)
            found += [f"{folder.name}: {each}" for each in differences(real.shapes, made.shapes)]
            lines += [{"folder": folder.name, "service": "fake", **each} for each in made.shapes]
            if drafted is not None:
                lines.append({"folder": folder.name, "service": service.name, "draft": drafted.record,
                              "page": drafted.page, "doi": drafted.doi, "kept": not arguments.discard})
    except BuildRefused as refused:
        lines.append({"stopped": [f"{r.code} [{r.category.value}]: {r.message}" for r in refused.refusals]})
        write_log(lines, token, arguments.log)
        raise
    lines.append({"differences": found})
    if write_log(lines, token, arguments.log) is None:
        return 3
    print("the fake answers as the sandbox does" if not found else "differences:\n  " + "\n  ".join(found))
    return 0 if not found else 1


if __name__ == "__main__":
    sys.exit(main())
