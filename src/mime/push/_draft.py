"""Drafts: make or update one, reserve its DOI, upload its files; read it as the service holds it; discard it. Never
publish: that is a later piece, with its own confirmation.

Every step is check-then-act against the service's state, read again on every run, so a push stopped at any request
is resumed by running the same command (design §2.6), at the level of whole files: a file in the draft whose size and
md5 are the planned file's was uploaded and verified, and is not sent again; any other is deleted and sent whole, into
an entry started for it (Zenodo keeps no partial upload, ignores ``Content-Range``, and deletes a completed file that
is sent again). After each upload the service's md5 and size must equal MIME's; MIME's sha256 is kept in push's record,
since Zenodo states md5 only. A file in the draft that this plan does not hold is deleted where push uploaded it
earlier, and refused where it did not (``push.draft-changed``): one added on the site is the author's to remove.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mime._terminal import plain
from mime.assets import size_text
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.credentials import Token
from mime.push._client import Client, Dropped, Errors, Json, Transport
from mime.push._plan import Plan, RecordFile, field_owner, make_plan
from mime.push._service import TEST_PREFIX, Service
from mime.push._sharing import read_sharing, scaffold, sharing_path
from mime.push._state import DraftState, Uploaded, clear_state, read_state, staging, state_directory, write_state

#: How often one file is started again from the beginning after its upload failed, before push stops.
UPLOAD_ATTEMPTS = 3

type Say = Callable[[str], None]
type Ask = Callable[[str], bool]


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> Refusal:
    return Refusal(code, category, message, action)


def _quiet(line: str) -> None:
    pass


@dataclass(frozen=True)
class Drafted:
    """What a push left on the service: the draft's id, its page, its reserved DOI, its concept's id, the files it
    uploaded this run and those it found already uploaded and verified."""

    record: str
    page: str
    doi: str
    concept: str | None
    uploaded: tuple[str, ...]
    kept: tuple[str, ...]


def experiment_name(folder: Path) -> str:
    from mime.saved import read_folder

    return read_folder(folder, absent_referenced=True).document["experiment"]


def _token(service: Service) -> Token:
    """The token push sends ``service``, and only where MIME may go online (``MIME_OFFLINE`` closes push too)."""
    from mime import credentials
    from mime.assets import offline

    if offline():
        raise BuildRefused([_refuse(
            "push.service", RefusalClass.ENVIRONMENT,
            f"MIME_OFFLINE is set, so push sends nothing to {service.name}",
            "unset MIME_OFFLINE to let push go online, then `mime push <folder> --to <service>` again")])
    found = credentials.token(service.credential)
    if found is None:
        raise BuildRefused([_refuse(
            "push.token", RefusalClass.ENVIRONMENT,
            f"no token for {service.name}: push uploads with the token of the account the draft belongs to; nothing "
            f"was sent", f"make a personal access token with the scope deposit:write on the service's site and store "
                         f"it, `mime credentials set {service.credential}`")])
    return found


def _unconfirmed(what: str) -> BuildRefused:
    return BuildRefused([_refuse(
        "push.unconfirmed", RefusalClass.USER,
        f"{what} was shown and not confirmed, and there is no terminal to ask in: nothing was sent",
        "read it, then run the same command again with --yes")])


def init(folder: str | os.PathLike[str]) -> tuple[Path, bool, str | None]:
    """Write ``sharing.yaml`` beside the saved form from what the folder knows: its path, whether it was written (an
    existing one is never overwritten), and the licence the experiment declares."""
    from mime.saved import read_folder

    root = Path(os.path.abspath(folder))
    document = read_folder(root, absent_referenced=True).document
    path = sharing_path(root)
    licence = document.get("licence")
    if path.exists():
        return path, False, licence
    path.write_text(scaffold(document["experiment"], licence), encoding="utf-8", newline="\n")
    return path, True, licence


def _entry_verified(entry: Json, file: RecordFile) -> bool:
    """Whether the service holds ``file`` complete: its stated size and md5 are MIME's."""
    return entry.get("status") == "completed" and entry.get("size") == file.size and \
        entry.get("checksum") == f"md5:{file.md5}"


def _errors_refusal(service: Service, record: str, what: str, errors: Errors | list[object]) -> BuildRefused:
    """A refusal for each field the service names in ``errors``, by who owns it (``FIELD_OWNERS``): the author's in
    sharing.yaml or the experiment, to correct there; MIME's, its defect, said as one."""
    found: list[Refusal] = []
    for each in errors.errors if isinstance(errors, Errors) else errors:
        held = cast(dict[str, object], each) if isinstance(each, dict) else {}
        field = str(held.get("field", ""))
        said = Errors([each]).words()
        owner = field_owner(field) if field else "mime"
        if owner == "sharing":
            found.append(_refuse(
                "push.rejected", RefusalClass.USER,
                f"{service.name} kept the draft {record} and refused part of {what}: {said}",
                f"correct it in sharing.yaml, then `mime push <folder> --to {service.name}`, which writes the draft's "
                f"metadata again"))
        elif owner == "experiment":
            found.append(_refuse(
                "push.rejected", RefusalClass.USER,
                f"{service.name} kept the draft {record} and refused part of {what}: {said}",
                f"correct the licence or terms where the experiment is built, exp.licence(...) or Terms(...), save it "
                f"again, built.save(folder), then `mime push <folder> --to {service.name}`"))
        else:
            found.append(_refuse(
                "push.rejected", RefusalClass.INTERNAL,
                f"{service.name} kept the draft {record} and refused part of {what}, in a field MIME builds, not one "
                f"you write: {said}. This is MIME's defect, not yours", "report it with this message"))
    return BuildRefused(found or [_refuse(
        "push.rejected", RefusalClass.INTERNAL,
        f"{service.name} kept the draft {record} and listed errors naming no field in its answer to {what}. This is "
        f"MIME's defect, not yours", "report it with this message")])


def _published(service: Service, record: str) -> BuildRefused:
    return BuildRefused([_refuse(
        "push.published", RefusalClass.USER,
        f"the record {record} on {service.name} is published: a published record cannot be deleted or changed by "
        f"push, and a new version of it comes with a later MIME", f"look at it, `mime push <folder> --to "
                                                                  f"{service.name} --status`")])


class _Drafting:
    """One push's requests, its record kept current after every step that changes the service."""

    def __init__(self, client: Client, plan: Plan, state: DraftState | None, say: Say) -> None:
        self.client = client
        self.plan = plan
        self.service = plan.service
        self.state = state
        self.say = say

    def save(self) -> None:
        assert self.state is not None
        write_state(self.service, self.state)

    def draft(self) -> tuple[DraftState, Json]:
        """Make the draft, or write the metadata of the one push made earlier, and reserve its DOI."""
        service, state = self.service, self.state
        found: Json | None = None
        if state is not None and state.record is not None:
            read = self.client.read_draft(state.record)
            if read is None:
                if self.client.read_record(state.record) is not None:
                    raise _published(service, state.record)
                self.say(f"the draft {state.record} MIME made earlier is no longer on {service.name} (discarded on "
                         f"the site?): a new one is made")
                state = None
            elif read[0].get("is_published") is True:
                raise _published(service, state.record)
            else:
                found = read[0]
        if state is None or state.record is None or found is None:
            if state is not None and state.creating:
                self.say(f"warning: an earlier push stopped while it asked {service.name} to make the draft, before "
                         f"its answer came, so {service.name} may hold a draft MIME has no record of: look at "
                         f"{service.site}/me/uploads, and delete one there that this push does not name")
            self.state = state = DraftState(service.name, service.api, self.plan.experiment, creating=True)
            self.save()
            draft, errors = self.client.create_draft(self.plan.payload)
            record = str(draft["id"])
            self.state = state = DraftState(service.name, service.api, self.plan.experiment, record,
                                            _parent_id(draft), None, _page(draft, service, record))
            self.save()
            self.say(f"made the draft {record}: {state.page}")
            if errors:
                raise _errors_refusal(service, record, "its metadata", errors)
        else:
            record = state.record
            assert record is not None
            body = dict(self.plan.payload)
            body["pids"] = found.get("pids", {})
            draft, errors = self.client.update_draft(record, body)
            self.say(f"wrote the metadata of the draft {record} again, from sharing.yaml and the folder")
            if errors:
                raise _errors_refusal(service, record, "its metadata", errors)
        doi = _doi_of(draft)
        if doi is None:
            try:
                draft = self.client.reserve_doi(record)
            except Errors as answered:
                raise _errors_refusal(service, record, "the DOI's reservation", answered) from None
            doi = _doi_of(draft)
            if doi is None:
                raise BuildRefused([_refuse(
                    "push.service", RefusalClass.ENVIRONMENT,
                    f"{service.name} answered the DOI's reservation for the draft {record} with no DOI",
                    "run the same command again, `mime push <folder> --to <service>`")])
            self.say(f"reserved the DOI {doi} for the draft {record} (registered only when it is published)")
        if service.production and doi.startswith(TEST_PREFIX):
            raise BuildRefused([_refuse(
                "push.service", RefusalClass.ENVIRONMENT,
                f"{service.name}, a production service, reserved {doi}, under DataCite's test prefix: push stops "
                f"rather than give a production record a DOI that does not resolve",
                "look at the draft, `mime push <folder> --to zenodo --status`, and discard it, `mime push <folder> "
                "--to zenodo --discard`")])
        state.doi = doi
        state.concept = _parent_id(draft) or state.concept
        self.save()
        return state, draft

    def upload(self) -> tuple[list[str], list[str]]:
        """Each planned file into the draft, unless it is there already, verified: the files sent and those kept."""
        state = self.state
        assert state is not None and state.record is not None
        record = state.record
        held = self.client.files(record)
        planned = {file.key for file in self.plan.files}
        strays = sorted(key for key in held if key not in planned)
        unknown = [key for key in strays if key not in state.files]
        if unknown:
            raise BuildRefused([_refuse(
                "push.draft-changed", RefusalClass.USER,
                f"the draft {record} on {self.service.name} holds {', '.join(plain(key) for key in unknown)}, which "
                f"this plan does not and push did not upload (added on the site?)",
                f"remove it on the draft's page, {state.page}, or start again from no draft, `mime push <folder> --to "
                f"{self.service.name} --discard`")])
        for key in strays:
            self.client.delete_file(record, key)
            del state.files[key]
            self.save()
            self.say(f"{key}: deleted from the draft (an earlier push uploaded it, and this plan does not hold it)")
        sent: list[str] = []
        kept: list[str] = []
        for file in self.plan.files:
            entry = held.get(file.key)
            mine = Uploaded(file.size, file.md5, file.sha256)
            if entry is not None and _entry_verified(entry, file):
                if state.files.get(file.key) != mine:
                    state.files[file.key] = mine
                    self.save()
                kept.append(file.key)
                self.say(f"{file.key}: already in the draft, its md5 {file.md5} and size verified, not sent again")
                continue
            if state.files.pop(file.key, None) is not None:
                self.save()
            if entry is not None:
                self.client.delete_file(record, file.key)
            self._send(record, file)
            state.files[file.key] = mine
            self.save()
            sent.append(file.key)
            self.say(f"{file.key}: uploaded, {size_text(file.size)}, its md5 {file.md5} equal to the one "
                     f"{self.service.name} states")
        return sent, kept

    def _send(self, record: str, file: RecordFile) -> None:
        """One file, whole, into an entry started for it; after a failure the entry is cleared and the file sent again
        from its start, at most :data:`UPLOAD_ATTEMPTS` times."""
        problem = ""
        mismatch = False
        for _ in range(UPLOAD_ATTEMPTS):
            self.client.start_file(record, file.key)
            entry: Json | None = None
            try:
                entry = self.client.send_content(record, file.key, file.path)
                entry = self.client.commit_file(record, file.key)
            except Dropped:
                problem, mismatch, entry = "its connection dropped", False, None
            except Errors as answered:
                problem, mismatch, entry = f"{self.service.name} answered {answered.words()}", False, None
            if entry is not None and _entry_verified(entry, file):
                return
            if entry is not None:
                problem = (f"{self.service.name} states size {entry.get('size')} and "
                           f"{plain(str(entry.get('checksum')))}, and MIME sent {file.size} bytes of md5 {file.md5}")
                mismatch = True
            now = self.client.files(record).get(file.key)
            if now is not None and _entry_verified(now, file):
                return
            if now is not None:
                self.client.delete_file(record, file.key)
            self.say(f"{file.key}: {problem}; its entry cleared, and the file sent again from its start (Zenodo keeps "
                     f"no partial upload)")
        if mismatch:
            raise BuildRefused([_refuse(
                "push.checksum", RefusalClass.ENVIRONMENT,
                f"{file.key}: after {UPLOAD_ATTEMPTS} uploads, {problem}: the draft does not hold the file push sent, "
                f"and its entry was deleted", "run the same command again, `mime push <folder> --to <service>`; what "
                                              "was uploaded and verified is not sent again")])
        raise BuildRefused([_refuse(
            "push.service", RefusalClass.ENVIRONMENT, f"{file.key}: {UPLOAD_ATTEMPTS} uploads failed: {problem}",
            "run the same command again later, `mime push <folder> --to <service>`: what was uploaded and verified is "
            "not sent again")])

    def check(self) -> None:
        """The draft holds exactly the planned files, each verified."""
        state = self.state
        assert state is not None and state.record is not None
        held = self.client.files(state.record)
        wrong = [file.key for file in self.plan.files if file.key not in held or not _entry_verified(held[file.key],
                                                                                                     file)]
        extra = sorted(set(held) - {file.key for file in self.plan.files})
        if wrong or extra:
            raise BuildRefused([_refuse(
                "push.service", RefusalClass.ENVIRONMENT,
                f"read again after the uploads, the draft {state.record} does not hold the plan's files: "
                f"{', '.join(plain(key) for key in wrong + extra)} differ", "run the same command again, `mime push "
                                                                             "<folder> --to <service>`")])


def _parent_id(draft: Json) -> str | None:
    parent = draft.get("parent")
    found = cast(dict[str, object], parent).get("id") if isinstance(parent, dict) else None
    return str(found) if found is not None else None


def _doi_of(draft: Json) -> str | None:
    pids = draft.get("pids")
    doi = cast(dict[str, object], pids).get("doi") if isinstance(pids, dict) else None
    identifier = cast(dict[str, object], doi).get("identifier") if isinstance(doi, dict) else None
    return identifier if isinstance(identifier, str) and identifier.startswith("10.") else None


def _page(draft: Json, service: Service, record: str) -> str:
    links = draft.get("links")
    page = cast(dict[str, object], links).get("self_html") if isinstance(links, dict) else None
    return page if isinstance(page, str) and page.startswith(service.site + "/") else service.draft_page(record)


def push(folder: str | os.PathLike[str], service: Service, *, yes: bool = False, confirm: Ask | None = None,
         say: Say = _quiet, transport: Transport | None = None, pause: float | None = None,
         sleep: Callable[[float], None] = time.sleep) -> Drafted | None:
    """Show the plan, then make or update the draft of the saved experiment in ``folder`` on ``service``, reserve its
    DOI and upload its files; never publish. Nothing is sent before the plan is shown and confirmed (``yes``, or
    ``confirm`` answering true); without either, refused (``push.unconfirmed``). ``None`` when the plan is declined."""
    root = Path(os.path.abspath(folder))
    sharing = read_sharing(root)
    experiment = experiment_name(root)
    state = read_state(service, experiment)
    plan = make_plan(root, service, sharing, staging(service, experiment))
    for line in plan.lines(state):
        say(line)
    token = _token(service)
    if not yes:
        if confirm is None:
            raise _unconfirmed("the plan")
        if not confirm(f"Make or update the draft on {service.name}?"):
            say("Not made: nothing was sent.")
            return None
    return send(plan, state, Client(service, token, transport, pause=pause, sleep=sleep), say)


def send(plan: Plan, state: DraftState | None, client: Client, say: Say = _quiet) -> Drafted:
    """What ``push`` does once its plan is shown and confirmed: make or update the draft, reserve its DOI, upload each
    file that differs, and read the draft again to check it holds exactly the plan's files, verified."""
    drafting = _Drafting(client, plan, state, say)
    state, _ = drafting.draft()
    sent, kept = drafting.upload()
    drafting.check()
    assert state.record is not None and state.page is not None and state.doi is not None
    say(f"The draft {state.record} on {plan.service.name} holds the {len(plan.files)} record files, each md5 "
        f"verified; its DOI {state.doi} is reserved, and nothing is published. Look at it: {state.page}")
    return Drafted(state.record, state.page, state.doi, state.concept, tuple(sent), tuple(kept))


def status(folder: str | os.PathLike[str], service: Service, *, say: Say = _quiet, transport: Transport | None = None,
           pause: float | None = None) -> DraftState | None:
    """Say what push knows of the experiment's draft on ``service`` and the draft as the service holds it: its page,
    reserved DOI, concept, title, creators and licences, and each file with its size and md5, against what push
    uploaded and verified. Changes nothing."""
    root = Path(os.path.abspath(folder))
    experiment = experiment_name(root)
    state = read_state(service, experiment)
    if state is None or state.record is None:
        say(f"no draft of {plain(experiment)!r} on {service.name} that MIME knows of (push keeps its record in "
            f"{state_directory(service, experiment)})")
        return state
    client = Client(service, _token(service), transport, pause=pause)
    read = client.read_draft(state.record)
    if read is None:
        if client.read_record(state.record) is not None:
            say(f"the record {state.record} on {service.name} is published: {service.site}/records/{state.record}")
        else:
            say(f"the draft {state.record} MIME made is no longer on {service.name} (discarded on the site?); the next "
                f"push makes a new one")
        return state
    draft, errors = read
    metadata = draft.get("metadata")
    held = cast(dict[str, object], metadata) if isinstance(metadata, dict) else {}
    say(f"The draft {state.record} on {service.name}: {_page(draft, service, state.record)}")
    say(f"  status: {plain(str(draft.get('status', 'draft')))}, published: {draft.get('is_published') is True}; "
        f"reserved DOI: {_doi_of(draft) or 'none'}; concept: {_parent_id(draft) or 'none'}")
    say(f"  title: {plain(str(held.get('title', '')))}")
    say(f"  creators: {'; '.join(_names(held.get('creators')))}")
    say(f"  licences: {'; '.join(_rights(held.get('rights')))}")
    if errors:
        say(f"  {service.name} holds it with errors: {Errors(errors).words()}")
    files = client.files(state.record)
    say(f"  files ({len(files)}):")
    for key, entry in sorted(files.items()):
        mine = state.files.get(key)
        checksum = str(entry.get("checksum", ""))
        verdict = ("not complete" if entry.get("status") != "completed" else
                   "not uploaded by push" if mine is None else
                   "as push uploaded and verified it" if (entry.get("size"), checksum) == (mine.size, f"md5:{mine.md5}")
                   else "differs from what push uploaded")
        say(f"    {plain(key)}  {entry.get('size')} bytes  {plain(checksum)}  {verdict}")
    for key in sorted(set(state.files) - set(files)):
        say(f"    {plain(key)}: push uploaded it, and the draft no longer holds it")
    return state


def discard(folder: str | os.PathLike[str], service: Service, *, yes: bool = False, confirm: Ask | None = None,
            say: Say = _quiet, transport: Transport | None = None, pause: float | None = None) -> bool:
    """Delete the experiment's draft on ``service`` (only a draft can be deleted) and push's record of it: whether a
    draft was deleted."""
    root = Path(os.path.abspath(folder))
    experiment = experiment_name(root)
    state = read_state(service, experiment)
    if state is None or state.record is None:
        say(f"no draft of {plain(experiment)!r} on {service.name} that MIME knows of: nothing to discard")
        return False
    client = Client(service, _token(service), transport, pause=pause)
    read = client.read_draft(state.record)
    if read is None:
        if client.read_record(state.record) is not None:
            raise _published(service, state.record)
        clear_state(service, experiment)
        say(f"the draft {state.record} is no longer on {service.name}; push's record of it is cleared")
        return False
    if read[0].get("is_published") is True:
        raise _published(service, state.record)
    question = f"Delete the draft {state.record} on {service.name} and its files ({state.page})?"
    if not yes:
        if confirm is None:
            raise _unconfirmed(f"the deletion of the draft {state.record}")
        if not confirm(question):
            say("Not deleted.")
            return False
    client.delete_draft(state.record)
    clear_state(service, experiment)
    say(f"deleted the draft {state.record} on {service.name}, and push's record of it")
    return True


def _names(value: object) -> list[str]:
    found: list[str] = []
    for each in cast(list[object], value) if isinstance(value, list) else []:
        person = cast(dict[str, object], each).get("person_or_org") if isinstance(each, dict) else None
        name = cast(dict[str, object], person).get("name") if isinstance(person, dict) else None
        family = cast(dict[str, object], person).get("family_name") if isinstance(person, dict) else None
        found.append(plain(str(name or family or "?")))
    return found or ["none"]


def _rights(value: object) -> list[str]:
    found: list[str] = []
    for each in cast(list[object], value) if isinstance(value, list) else []:
        entry = cast(dict[str, object], each) if isinstance(each, dict) else {}
        title = entry.get("title")
        words = cast(dict[str, object], title).get("en") if isinstance(title, dict) else None
        found.append(plain(str(entry.get("id") or words or "?")))
    return found or ["none"]


__all__ = ["UPLOAD_ATTEMPTS", "Drafted", "discard", "experiment_name", "init", "push", "send", "status"]
