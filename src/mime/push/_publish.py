"""Publishing: the one step that cannot be undone (design §2.5), and what push checks and records afterwards (§2.9).

``mime push <folder> --to <service> --publish [--confirm <DOI>]`` publishes the draft an earlier push made, and only
that: the folder packed again must make the files the draft holds (``push.folder-changed`` otherwise: push the folder,
look at the draft, then publish), and the plan shown is the one published. The metadata is written once more where
it differs (``sharing.yaml`` changed, the publication date a day's), each edit made on the site settled as a push
settles it.

**The confirmation**, in words, before the request: what cannot be undone, the files published as the uploader's own
work under the experiment's licence, the consent of everyone named; then **each declared permission asked on its own,
apt-style** (``y`` to declare it), worded as the public statement it becomes, in a terminal only: ``--yes`` never
answers one, and a publish without a terminal goes ahead only for declarations confirmed in a terminal for exactly
these bytes (``push.permission-unconfirmed`` otherwise). **The publish itself is behind the record number, typed**
(never ``y``, which a hand types by habit), or ``--confirm <DOI>`` from a script, which must be the draft's reserved
DOI (``push.not-confirmed``), so a script cannot publish another draft than the one it was shown.

**Never sent twice.** An answer lost is followed by reading the record: published, push goes on to the checks; not, it
stops, and the next run reads the record again before it does anything. A run that finds the draft already published
(an earlier run's answer lost) goes straight to the checks.

**The checks afterwards**: the record read again as published, its DOI the reserved one, each file's size and md5 the
uploaded one's and no other file, its metadata what push wrote, and the bundle downloaded and its sha256 the one
packed. Only then is the folder's lock replaced by the staged lock (production only: a sandbox record is never written
into the folder, the sandbox being one that can be cleaned at any time), when it is still the lock the bundle was
packed from (``push.lock-changed`` otherwise), so ``mime pack <folder>`` makes the published bundle again, byte for
byte. Push's record keeps the published version, from which the next push of a changed folder makes a new version.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

from mime._terminal import plain
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._client import Client, Dropped, Json, Transport
from mime.push._draft import (
    Ask,
    Choices,
    Drafting,
    Say,
    concept_doi_of,
    doi_of,
    entry_verified,
    experiment_name,
    foreign,
    metadata_of,
    ours,
    planned,
    record_page,
    token_of,
    unchanged,
    unconfirmed,
    usable,
)
from mime.push._edits import differs, field_value
from mime.push._plan import Plan, declaring
from mime.push._service import Service
from mime.push._stage import rewrite_lock
from mime.push._state import (
    CUSTOM,
    DraftState,
    Published,
    Uploaded,
    declaration_identity,
    metadata_marks,
    read_state,
    write_state,
)

type Typed = Callable[[str], str]

#: What publishing means on production, in Zenodo's own terms (its API and help pages, read 2026-11-01).
PRODUCTION_FACTS = ("This cannot be undone. The DOI is registered for good. Zenodo lets the record's owner delete it "
                    "within 30 days, which leaves a tombstone page with its citation; after that only Zenodo can "
                    "withdraw it. The files cannot be changed: a change is a new version.")
#: What it means on the sandbox, a test service.
SANDBOX_FACTS = ("This is the sandbox, a test service: its DOI does not resolve, and the sandbox can be cleaned at any "
                 "time. On Zenodo itself this step cannot be undone, and the same words are asked there.")


def _quiet(line: str) -> None:
    pass


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> Refusal:
    return Refusal(code, category, message, action)


def _no_draft(service: Service, experiment: str, why: str) -> BuildRefused:
    return BuildRefused([_refuse(
        "push.no-draft", RefusalClass.USER,
        f"there is no draft of {plain(experiment)!r} on {service.name} to publish: {why}",
        f"make the draft, `mime push <folder> --to {service.name}`, look at it on its page, then `mime push <folder> "
        f"--to {service.name} --publish`")])


def statement(plan: Plan, state: DraftState) -> list[str]:
    """The words shown before the record number is asked: what is published, as what, and what cannot be undone."""
    published = state.published
    what = (f"version {state.version}, a new version of {published.doi}" if published is not None else
            "version 1 of a new concept")
    yours = [file for file in plan.judged.files if file.owner == "yours" and file.verdict.carried
             and file.referenced is None]
    return [f"Publishing {plain(plan.experiment)!r} on {plan.service.name} as {state.doi} ({what}).",
            PRODUCTION_FACTS if plan.service.production else SANDBOX_FACTS,
            f"  You are publishing, under your name, {len(yours)} file{'s' if len(yours) != 1 else ''} as your own "
            f"work (or your lab's), licensed {plain(plan.licence)}.",
            "  You state that everyone named as a creator or contributor agreed to be named."]


def _permissions(plan: Plan, state: DraftState, ask: Ask | None, say: Say) -> bool:
    """Each declared permission asked on its own, in a terminal (``ask``); without one, each must have been
    confirmed in a terminal for exactly these bytes. Whether every one is declared."""
    declared = plan.judged.permissions
    if not declared:
        return True
    if ask is None:
        missing = [index for index, held in enumerate(declared, start=1)
                   if declaration_identity(cast(dict[str, object], held)) not in state.confirmed]
        if missing:
            raise BuildRefused([_refuse(
                "push.permission-unconfirmed", RefusalClass.USER,
                f"permission{'s' if len(missing) != 1 else ''} {', '.join(map(str, missing))} of {len(declared)} "
                f"{'were' if len(missing) != 1 else 'was'} never confirmed in a terminal for these files: a "
                f"declaration under your name is your own act, and --yes or --confirm never makes it; nothing was "
                f"published", f"publish in a terminal, `mime push <folder> --to {plan.service.name} --publish`, "
                              f"which asks you to declare each")])
        say(f"each of the {len(declared)} declared permissions was confirmed in a terminal for these files")
        return True
    for index, held in enumerate(declared, start=1):
        words = declaring(index, held, plan.judged)
        say(f"  {words[0]}")
        for line in words[1:]:
            say(f"       {line.strip()}")
        if not ask(f"Declare permission {index} publicly, under your name?"):
            say(f"Not published: permission {index} was not declared.")
            return False
        identity = declaration_identity(cast(dict[str, object], held))
        if identity not in state.confirmed:
            state.confirmed.append(identity)
    write_state(plan.service, state)
    return True


def _number(state: DraftState, confirm: str | None, typed: Typed | None, say: Say, service: Service) -> bool:
    """The record number typed, or ``confirm`` from a script, equal to the draft's; whether to publish."""
    assert state.record is not None and state.doi is not None
    if confirm is not None:
        return True  # held to the draft's DOI before any request
    if typed is None:
        raise unconfirmed(f"publishing the draft {state.record}", f"type its number in a terminal, or pass its DOI "
                                                                  f"from a script, `mime push <folder> --to "
                                                                  f"{service.name} --publish --confirm {state.doi}`")
    answer = typed(f"Type the record number to publish ({state.record}): ").strip()
    if answer != state.record:
        say(f"Not published: {plain(answer)!r} is not the record number {state.record}." if answer else
            "Not published.")
        return False
    return True


def _held_to(state: DraftState, confirm: str | None, service: Service) -> None:
    if confirm is not None and confirm.strip() not in (state.doi, f"https://doi.org/{state.doi}"):
        raise BuildRefused([_refuse(
            "push.not-confirmed", RefusalClass.USER,
            f"--confirm names {plain(confirm)}, and the draft push would publish on {service.name} reserved "
            f"{state.doi}: a script publishes only the draft it was shown; nothing was sent",
            f"pass the draft's DOI, `--confirm {state.doi}`, after reading its plan, `mime push <folder> --to "
            f"{service.name} --status`")])


def publish(folder: str | os.PathLike[str], service: Service, *, confirm: str | None = None, ask: Ask | None = None,
            typed: Typed | None = None, say: Say = _quiet, transport: Transport | None = None,
            pause: float | None = None, sleep: Callable[[float], None] = time.sleep, overwrite_site_edits: bool = False,
            keep_site_edits: bool = False, full: bool = False) -> Published | None:
    """Publish the draft of the saved experiment in ``folder`` on ``service``, after the plan, the confirmation in
    words, each declared permission asked (``ask``, a terminal's), and the record number typed (``typed``) or its DOI
    passed (``confirm``); then check the record as published and, on production, write the staged lock into the
    folder. The published version, or ``None`` where a question was declined."""
    usable(service)
    root = Path(os.path.abspath(folder))
    experiment = experiment_name(root)
    state = read_state(service, experiment)
    if state is None or (state.record is None and state.published is None):
        raise _no_draft(service, experiment, "push has made none")
    if state.record is None:
        return _again(root, service, state, transport, pause, sleep, say)
    if state.doi is None:
        raise _no_draft(service, experiment, f"the draft {state.record} has no DOI yet, its push having stopped")
    _held_to(state, confirm, service)
    plan = planned(root, service, state)
    client = Client(service, token_of(service), transport, pause=pause, sleep=sleep)
    read = client.read_draft(state.record)
    if read is None or read[0].get("is_published") is True:
        found = client.read_record(state.record)
        if found is None or found.get("is_published") is not True:
            raise _no_draft(service, experiment, f"the draft {state.record} is no longer on {service.name} (discarded "
                                                 f"on the site?)")
        say(f"the draft {state.record} is published already (an earlier publish whose answer was lost, or the site's "
            f"button): checking it, and publishing nothing")
        return finish(plan, state, client, say)
    if not ours(read[0], service, state):
        raise foreign(service, state)
    held = client.files(state.record)
    differing = [file.key for file in plan.files if not entry_verified(held.get(file.key), file)] + \
        sorted(set(held) - {file.key for file in plan.files})
    if differing:
        raise BuildRefused([_refuse(
            "push.folder-changed", RefusalClass.USER,
            f"the folder, packed again, does not make the files the draft {state.record} holds "
            f"({', '.join(plain(key) for key in differing)} differ): it changed since the draft was made, or the "
            f"draft's files did, and push publishes only what you could look at; nothing was published",
            f"update the draft, `mime push <folder> --to {service.name}`, look at it on its page, then publish it")])
    shown: list[Plan] = []

    def show() -> None:
        current = drafting.plan
        shown.append(current)
        for line in (current.lines(state) if full else current.summary(state, publishing=True)):
            say(line)

    drafting = Drafting(client, plan, state, say, Choices(ask, overwrite_site_edits, keep_site_edits), show, full)
    drafting.update(read[0], dated=True)
    if not shown or shown[-1] is not drafting.plan:
        show()  # the plan published: shown last, after any edit kept from the site
    plan = drafting.plan
    if drafting.wrote:
        say(f"wrote the draft's metadata again, as the plan shows it: {state.page}")
    for line in statement(plan, state):
        say(line)
    if not _permissions(plan, state, ask, say) or not _number(state, confirm, typed, say, service):
        say(str(state.page))
        return None
    try:
        client.publish(state.record)
    except Dropped:
        found = client.read_record(state.record)
        if found is None or found.get("is_published") is not True:
            raise client.lost("POST", f"/records/{state.record}/draft/actions/publish",
                              "the connection dropped, and the record, read again, is not published") from None
        say("the answer to the publish was lost, and the record, read again, is published: checking it")
    return finish(plan, state, client, say)


def _again(root: Path, service: Service, state: DraftState, transport: Transport | None, pause: float | None,
           sleep: Callable[[float], None], say: Say) -> Published:
    """No draft to publish: the version last published checked again, and its lock written into the folder where it
    is not yet (an earlier publish whose checks or rewrite stopped)."""
    published = state.published
    assert published is not None
    plan = planned(root, service, state)
    if not unchanged(plan, published):
        raise _no_draft(service, state.experiment, f"version {published.version} is published, and the folder "
                                                   f"changed since: its next version is made by a push")
    client = Client(service, token_of(service), transport, pause=pause, sleep=sleep)
    draft_state = replace(state, record=published.record, doi=published.doi, concept=published.concept,
                          version=published.version, page=published.page)
    say(f"nothing to publish: version {published.version} is published ({published.doi}); checking it again")
    return finish(plan, draft_state, client, say, recorded=True)


def _differences(plan: Plan, state: DraftState, record: Json, files: dict[str, Json], bundle: bytes) -> list[str]:
    """What the record, as published, holds otherwise than push sent: the DOI, a file, a field of the metadata, the
    bundle's bytes."""
    import hashlib

    found: list[str] = []
    if record.get("is_published") is not True:
        found.append("it does not read as published")
    if doi_of(record) != state.doi:
        found.append(f"its DOI is {doi_of(record)}, not the reserved {state.doi}")
    for file in plan.files:
        if not entry_verified(files.get(file.key), file):
            entry = files.get(file.key) or {}
            found.append(f"{plain(file.key)}: size {entry.get('size')} and {plain(str(entry.get('checksum')))}, not "
                         f"{file.size} bytes of md5 {file.md5}")
    found += [f"{plain(key)}: a file push did not upload" for key in sorted(set(files) - {f.key for f in plan.files})]
    sent = plan.payload
    names = [*metadata_of(sent), *(f"{CUSTOM}{key}" for key in cast(dict[str, object], sent.get("custom_fields", {})))]
    found += [f"its metadata's {name} is {plain(str(field_value(record, name)))[:120]}, not what push wrote"
              for name in names if name not in ("identifiers", "publication_date") and differs(name, record, sent)]
    if hashlib.sha256(bundle).hexdigest() != plan.files[0].sha256:
        found.append(f"the bundle downloaded from it is not the one packed (sha256 {plan.files[0].sha256})")
    return found


def finish(plan: Plan, state: DraftState, client: Client, say: Say, *, recorded: bool = False) -> Published:
    """The checks after publishing (the record read again; its DOI, files, metadata and bundle against what push sent),
    push's record of the published version, and, on production, the staged lock written into the folder."""
    service = plan.service
    record = state.record
    assert record is not None and state.doi is not None and state.concept is not None
    answer = client.read_record(record)
    if answer is None:
        raise BuildRefused([_refuse(
            "push.service", RefusalClass.ENVIRONMENT, f"the record {record} on {service.name}, published, cannot be "
                                                      f"read back", f"run the same command again, `mime push "
                                                                    f"<folder> --to {service.name} --publish`: it "
                                                                    f"checks the record and publishes nothing again")])
    page = record_page(service, record)
    published = Published(record, state.doi, page, state.concept, state.version, concept_doi_of(answer),
                          {file.key: Uploaded(file.size, file.md5, file.sha256) for file in plan.files},
                          metadata_marks(answer), state.sent, dict(plan.publication or {}))
    if not recorded and answer.get("is_published") is True:
        # recorded first: whatever the checks below find, the version is published, and --publish checks it again
        write_state(service, replace(state, record=None, doi=None, page=None, files={}, written={}, sent="",
                                     confirmed=[], creating=False, version=state.version + 1, published=published))
    concept = f"; its concept, every version: {published.concept_doi}" if published.concept_doi else ""
    say(f"published {plain(plan.experiment)!r} as {state.doi} (version {state.version}{concept})")
    files = client.record_files(record)
    bundle = client.download(record, plan.files[0].key)
    problems = _differences(plan, state, answer, files, bundle)
    if problems:
        say(page)
        raise BuildRefused([_refuse(
            "push.after-publish", RefusalClass.ENVIRONMENT,
            f"the record {record} on {service.name} is published, and read back it differs from what push sent: "
            f"{'; '.join(problems)}; so push did not write its lock into the folder",
            f"look at the record, {page}; run `mime push <folder> --to {service.name} --publish` again, which checks "
            f"it again and publishes nothing")])
    say("checked against the record as published: its DOI, each file's size and md5, its metadata, and the bundle's "
        "sha256")
    refusal: Refusal | None = None
    if service.production and plan.staged is not None:
        refusal = rewrite_lock(plan.folder, plan.lock, plan.staged)
        if refusal is None:
            say("the folder's lock now names the record: `mime pack <folder>` makes the published bundle again")
    else:
        say(f"the folder's lock is left as it is: a {service.name} record is never written into the folder, the "
            f"sandbox being one that can be cleaned at any time")
    say(page)
    if refusal is not None:
        raise BuildRefused([refusal])
    return published


__all__ = ["PRODUCTION_FACTS", "SANDBOX_FACTS", "finish", "publish", "statement"]
