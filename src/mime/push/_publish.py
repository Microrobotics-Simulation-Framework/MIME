"""Publishing: the one step that cannot be undone (design §2.5), and what push checks and records afterwards (§2.9).

``mime push <folder> --to <service> --publish [--confirm <DOI>]`` publishes the draft an earlier push made, and only
that: the folder packed again must make the files the draft holds (``push.folder-changed`` otherwise: push the folder,
look at the draft, then publish), and the plan shown is the one published. Each edit made on the site is settled as a
push settles it.

**What is published is what was seen on the draft.** The title and the contributors are on the plan always. A script
(``--confirm``, or no terminal) publishes what the draft holds, the publication date aside, or nothing
(``push.metadata-changed``); in a terminal, each field ``sharing.yaml`` changed since the draft was written is shown by
field, with its new value, before the number is asked, and written to the draft. **Immediately before the request**, the
draft is read again (the prompt can wait for hours, and the plan's own advice is to look at its page): its DOI, its
metadata and its files must be what was shown, or nothing is published (``push.draft-moved``).

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

**The checks afterwards**: the record read again as published and recorded, each file as the record's own listing holds
it (so "unchanged" is true of the record, whatever the folder became); its DOI the reserved one, each file's size and
md5 the uploaded one's and no other file, and the bundle downloaded and its sha256 the one packed. A difference in any
of those stops here (``push.after-publish``), the folder's lock left, and a later ``--publish`` checks again and finishes.
Then the folder's lock is replaced by the staged lock (production only: a sandbox record is never written into the
folder, the sandbox being one that can be cleaned at any time), when it is still the lock the bundle was packed from
(``push.lock-changed`` otherwise), so ``mime pack <folder>`` makes the published bundle again, byte for byte.

**The metadata is held to the draft as last read, and never holds back the lock**, which rests on the DOI and the bytes
alone: that the service keeps a field as the draft held it is a guess (a sandbox session's to measure), and a service
that normalises one must not leave a published record with no way to finish. A field that reads back otherwise is named
(``push.metadata-read-back``) once the version is recorded and the lock written; a later ``--publish`` names it again,
and what was edited on the record's page since, and refuses nothing.
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
from mime.push._client import Client, Dropped, Errors, Json, Transport
from mime.push._draft import (
    Ask,
    Choices,
    Drafting,
    Say,
    concept_doi_of,
    doi_of,
    edited_fields,
    entry_verified,
    errors_refusal,
    experiment_name,
    foreign,
    label,
    marked,
    metadata_of,
    ours,
    planned,
    record_page,
    token_of,
    unchanged,
    unconfirmed,
    usable,
)
from mime.push._edits import SHARING_KEYS, apart, differs, field_value, words_of
from mime.push._plan import Plan, declaring
from mime.push._service import Service
from mime.push._stage import lock_unwritten, publication_of, rewrite_lock
from mime.push._state import (
    ACCESS,
    CUSTOM,
    DraftState,
    Published,
    Uploaded,
    declaration_identity,
    metadata_marks,
    payload_identity,
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


def _named(names: list[str], found: Json, body: Json) -> list[str]:
    """Each of ``names`` with its value on the draft and the value to be published."""
    lines: list[str] = []
    for name in names:
        held, then = apart(words_of(field_value(found, name)), words_of(field_value(body, name)))
        lines += [f"    {label(name)} on the draft: {held}", f"    {label(name)} to publish: {then}"]
    return lines


def changed_since_draft(found: Json, body: Json, state: DraftState) -> list[str]:
    """The fields a publish would write otherwise than the draft ``found`` holds them, ``body`` being what it would
    write: ``sharing.yaml`` changed since the draft was written. The publication date aside, the one write a publish
    makes of itself, and a field the service alone added."""
    if payload_identity(body) == state.sent:
        return []
    custom = body.get("custom_fields")
    names = {*metadata_of(body), ACCESS, *SHARING_KEYS, *(
        f"{CUSTOM}{key}" for key in (cast(dict[str, object], custom) if isinstance(custom, dict) else {}))}
    return sorted(name for name in names - {"publication_date", "identifiers"} if differs(name, found, body))


def _refuse_changes(service: Service, record: str, names: list[str], found: Json, body: Json) -> BuildRefused:
    return BuildRefused([_refuse(
        "push.metadata-changed", RefusalClass.USER,
        f"a script publishes only what the draft {record} on {service.name} holds, and this publish would write "
        f"otherwise: {'; '.join(line.strip() for line in _named(names, found, body))}; nothing was written or published",
        f"update the draft, `mime push <folder> --to {service.name}`, look at it on its page, then publish it; or "
        f"publish in a terminal, which shows each change first")])


def _no_longer(service: Service, experiment: str, record: str) -> BuildRefused:
    return _no_draft(service, experiment, f"the record {record} reads neither as a draft nor as published on "
                                          f"{service.name}: discarded on the site, or published a moment ago and not "
                                          f"readable yet")


def _moved(service: Service, record: str, what: list[str]) -> BuildRefused:
    return BuildRefused([_refuse(
        "push.draft-moved", RefusalClass.USER,
        f"the draft {record} on {service.name} is not what was shown: {'; '.join(what)}; nothing was published",
        f"look at the draft, then `mime push <folder> --to {service.name}`, which shows each edit and packs the bundle "
        f"naming the draft's DOI; then publish it")])


def _held_at_post(plan: Plan, state: DraftState, client: Client) -> bool:
    """Read immediately before the publish: whether the record is still a draft, and it holds what was shown (its DOI,
    its metadata as push last wrote it, exactly the plan's files); otherwise nothing is published (``push.draft-moved``).
    Found published (the site's button while the question waited), it is not a draft, and no request is to be made."""
    service, record = plan.service, state.record
    assert record is not None
    read = client.read_draft(record)
    if read is None or read[0].get("is_published") is True:
        found = client.read_record(record)
        if found is None or found.get("is_published") is not True:
            raise _no_longer(service, state.experiment, record)
        if not ours(found, service, state):
            raise foreign(service, state)
        return False
    draft = read[0]
    if not ours(draft, service, state):
        raise foreign(service, state)
    moved: list[str] = []
    if doi_of(draft) != state.doi:
        moved.append(f"its DOI is {plain(str(doi_of(draft)))}, and push reserved {state.doi}")
    body = marked(plan.payload, service, state)
    edits = [name for name in edited_fields(draft, state) if differs(name, draft, body)]
    moved += [line.strip() for line in _named(edits, draft, body)]
    held = client.files(record)
    moved += [f"{plain(file.key)} is not the file push uploaded" for file in plan.files
              if not entry_verified(held.get(file.key), file)]
    moved += [f"it holds {plain(key)}, which push did not upload" for key in sorted(set(held) -
                                                                                    {file.key for file in plan.files})]
    if moved:
        raise _moved(service, record, moved)
    return True


def _second_concept(plan: Plan, state: DraftState) -> None:
    """A folder whose lock names a published record on this service, of which push has no record here, is not published
    as a second concept without a word: until ``--version-of`` comes, it is refused."""
    named = publication_of(plan.lock)
    if named is None or named["service"] != plan.service.name or plan.second_concept(state) is None:
        return
    raise BuildRefused([_refuse(
        "push.second-concept", RefusalClass.USER,
        f"the folder's lock names the published record {plain(named['doi'])} on {plain(named['service'])}, and push has "
        f"no record of it here, so publishing the draft {state.record} would make a second concept, a second DOI for "
        f"this experiment, and that cannot be undone; nothing was published",
        f"to make a new version of that record, which `--version-of` will do later, press New version on its page; to "
        f"make a new concept on purpose, save the experiment again from its script, built.save(folder), whose lock "
        f"names no record, then `mime push <folder> --to {plan.service.name}`")])


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
            raise _no_longer(service, experiment, state.record)
        if not ours(found, service, state):
            raise foreign(service, state)
        say(f"the draft {state.record} is published already (an earlier publish whose answer was lost, or the site's "
            f"button): checking it, and publishing nothing")
        return finish(plan, state, client, say)
    if not ours(read[0], service, state):
        raise foreign(service, state)
    if doi_of(read[0]) != state.doi:
        raise _moved(service, state.record, [f"its DOI is {plain(str(doi_of(read[0])))}, and push reserved {state.doi} "
                                             f"and packed the bundle naming that"])
    _second_concept(plan, state)
    if service.production and plan.staged is not None and not os.access(plan.folder, os.W_OK):
        raise BuildRefused([lock_unwritten(plan.folder, "it is read-only")])
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
    over = drafting.settle(read[0], state.record)
    changes = changed_since_draft(read[0], drafting.payload(state), state)
    if (changes or over) and (confirm is not None or typed is None):
        raise _refuse_changes(service, state.record, sorted({*changes, *over}), read[0], drafting.payload(state))
    if unasked := [name for name in changes if name not in over]:
        say("! changed in sharing.yaml since the draft was written, and written to it now:")
        for line in _named(unasked, read[0], drafting.payload(state)):
            say(line)
    drafting.written_where_needed(read[0], over, dated=True)
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
    if not _held_at_post(plan, state, client):
        say(f"the draft {state.record} was published while the question waited (the site's button): checking it, and "
            f"publishing nothing")
        return finish(plan, state, client, say)
    try:
        client.publish(state.record)
    except Dropped:
        found = client.read_record(state.record)
        if found is None or found.get("is_published") is not True:
            raise client.lost("POST", f"/records/{state.record}/draft/actions/publish",
                              "the connection dropped, and the record, read again, is not published") from None
        say("the answer to the publish was lost, and the record, read again, is published: checking it")
    except Errors as answered:
        found = client.read_record(state.record)
        if found is None or found.get("is_published") is not True:
            raise errors_refusal(service, state.record, "the publish", answered, plan.owner) from None
        say(f"{service.name} answered the publish listing errors ({answered.words()}), and the record, read again, is "
            f"published: checking it")
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
                          version=published.version, page=published.page, written=published.written,
                          sent=published.sent)
    say(f"nothing to publish: version {published.version} is published ({published.doi}); checking it again")
    return finish(plan, draft_state, client, say, recorded=published)


def _differences(plan: Plan, state: DraftState, record: Json, files: dict[str, Json], bundle: bytes) -> list[str]:
    """What the record, as published, holds otherwise than push sent, and on which the folder's lock rests: the DOI, a
    file, the bundle's bytes."""
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
    if hashlib.sha256(bundle).hexdigest() != plan.files[0].sha256:
        found.append(f"the bundle downloaded from it is not the one packed (sha256 {plan.files[0].sha256})")
    return found


def listed(files: dict[str, Json], plan: Plan) -> dict[str, Uploaded]:
    """Each file as the record's own listing holds it; MIME's sha256 where it is the file packed."""
    packed = {file.key: file for file in plan.files}
    found: dict[str, Uploaded] = {}
    for key, entry in files.items():
        size, mine = entry.get("size"), packed.get(key)
        found[key] = Uploaded(size if isinstance(size, int) else -1, str(entry.get("checksum", "")).removeprefix("md5:"),
                              mine.sha256 if mine is not None and entry_verified(entry, mine) else "")
    return found


def _read_otherwise(record: Json, plan: Plan, state: DraftState) -> list[str]:
    """The fields of the record, as it reads now, that are not what ``state`` last held of it (by the service's own
    answers, so what it adds as it stores a draft is no difference). Its publication date and push's mark aside. Where the
    plan is what was sent, a field the service only filled in is no difference either; where it is not (``sharing.yaml``
    changed since), the plan says nothing of what was sent."""
    now = metadata_marks(record)
    names = sorted(name for name in set(now) | set(state.written) if now.get(name) != state.written.get(name)
                   and name not in ("publication_date", "identifiers"))
    body = marked(plan.payload, plan.service, state)
    sent = payload_identity(body) == state.sent
    return [name for name in names if not sent or differs(name, record, body)]


def _version_words(record: Json, state: DraftState) -> list[str]:
    versions = record.get("versions")
    index = cast(dict[str, object], versions).get("index") if isinstance(versions, dict) else None
    return [f"the service counts it as version {index}, and push wrote {state.version}"] \
        if isinstance(index, int) and index != state.version else []


def finish(plan: Plan, state: DraftState, client: Client, say: Say, *, recorded: Published | None = None) -> Published:
    """The checks after publishing and push's record of the published version, its files as the record's own listing
    holds them; on production, the staged lock written into the folder, once the DOI, the files and the bundle hold. A
    field of the metadata that reads back otherwise is named at the end (``push.metadata-read-back``), holding back
    nothing. With ``recorded``, a version already recorded, checked again: what differed then and what was edited on the
    record's page since are said, and no refusal rests on them."""
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
    if answer.get("is_published") is not True:
        raise BuildRefused([_refuse(
            "push.after-publish", RefusalClass.ENVIRONMENT, f"the record {record} on {service.name} does not read as "
                                                            f"published yet, so push recorded nothing and wrote no lock",
            f"run the same command again in a moment, `mime push <folder> --to {service.name} --publish`: it reads the "
            f"record first")])
    page = record_page(service, record)
    files = client.record_files(record)
    otherwise = _read_otherwise(answer, plan, state)
    words = [f"its {label(name)} reads {plain(words_of(field_value(answer, name)))[:120]}" for name in otherwise]
    if recorded is None:
        words += _version_words(answer, state)
        published = Published(record, state.doi, page, state.concept, state.version, concept_doi_of(answer),
                              listed(files, plan), metadata_marks(answer), state.sent, dict(plan.publication or {}),
                              plan.card, [*otherwise])
        # recorded first: whatever the checks below find, the version is published, and --publish checks it again
        write_state(service, replace(state, record=None, doi=None, page=None, files={}, written={}, sent="",
                                     confirmed=[], creating=False, version=state.version + 1, published=published))
    else:
        published = recorded
    concept = f"; its concept, every version: {published.concept_doi}" if published.concept_doi else ""
    say(f"published {plain(plan.experiment)!r} as {state.doi} (version {state.version}{concept})")
    bundle = client.download(record, plan.files[0].key)
    problems = _differences(plan, state, answer, files, bundle)
    if problems:
        say(page)
        raise BuildRefused([_refuse(
            "push.after-publish", RefusalClass.ENVIRONMENT,
            f"the record {record} on {service.name} is published, and read back it differs from what push sent: "
            f"{'; '.join([*problems, *words])}; so push did not write its lock into the folder",
            f"look at the record, {page}; run `mime push <folder> --to {service.name} --publish` again, which checks "
            f"it again and publishes nothing; if the folder changed since the draft, `mime push <folder> --to "
            f"{service.name}` makes its next version")])
    say("checked against the record as published: its DOI, each file's size and md5, and the bundle's sha256" if words
        else "checked against the record as published: its DOI, each file's size and md5, its metadata, and the "
             "bundle's sha256")
    refusal: Refusal | None = None
    if service.production and plan.staged is not None:
        refusal = rewrite_lock(plan.folder, plan.lock, plan.staged)
        if refusal is None:
            say("the folder's lock now names the record: `mime pack <folder>` makes the published bundle again")
    else:
        say(f"the folder's lock is left as it is: a {service.name} record is never written into the folder, the "
            f"sandbox being one that can be cleaned at any time")
    if recorded is not None and recorded.differed:
        say(f"! read back when it was published, its metadata differed from what the draft held: "
            f"{', '.join(label(name) for name in recorded.differed)}")
    if recorded is not None and otherwise:
        say(f"! edited on the record's page since it was published: {', '.join(label(name) for name in otherwise)}")
    say(page)
    if refusal is not None:
        raise BuildRefused([refusal])
    if recorded is None and words:
        raise BuildRefused([_refuse(
            "push.metadata-read-back", RefusalClass.ENVIRONMENT,
            f"the record {record} on {service.name} is published and recorded, and its DOI, files and bundle are what "
            f"push sent{'; the folder names it' if service.production and plan.staged is not None else ''}. Read back, "
            f"its metadata is not what the draft held: {'; '.join(words)}",
            f"look at the record, {page}: the service may have normalised a field as it published, and a correction is "
            f"made on its page. Nothing is owed: `mime push <folder> --to {service.name} --publish` checks it again and "
            f"publishes nothing")])
    return published


__all__ = ["PRODUCTION_FACTS", "SANDBOX_FACTS", "changed_since_draft", "finish", "listed", "publish", "statement"]
