"""Drafts: make or update one, reserve its DOI, upload its files; read it as the service holds it; discard it. Never
publish: that is a later piece, with its own confirmation.

Every step is check-then-act against the service's state, read again on every run, so a push stopped at any request
is resumed by running the same command (design §2.6), at the level of whole files: a file in the draft whose size and
md5 are the planned file's was uploaded and verified, and is not sent again; any other is deleted and sent whole, into
an entry started for it (Zenodo keeps no partial upload, ignores ``Content-Range``, and deletes a completed file that
is sent again). After each upload the service's md5 and size must equal MIME's; MIME's sha256 is kept in push's record,
since Zenodo states md5 only. A file in the draft that this plan does not hold is deleted where push uploaded it
earlier, and refused where it did not (``push.draft-changed``): one added on the site is the author's to remove.

Before it writes to a draft or deletes one, push checks the draft is the one it made: the draft carries the identifier
push wrote into it, a UUID of the service, the experiment's name and push's mark (``push.draft-foreign`` otherwise),
and its metadata is what push last wrote, or what push would write now (``push.metadata-edited`` otherwise, naming each
field edited on the site, until it is copied into sharing.yaml or ``overwrite_site_edits`` writes over it). Every push,
``--status`` and ``--discard`` ends with the draft's link on a line of its own.
"""

from __future__ import annotations

import json
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
from mime.push._client import Client, Dropped, Errors, Gone, Json, Owner, Transport, owned_refusals
from mime.push._owners import FIELD_OWNERS, field_owner
from mime.push._plan import Plan, RecordFile, make_plan
from mime.push._service import TEST_PREFIX, ZENODO, ZENODO_SANDBOX, Service
from mime.push._sharing import read_sharing, scaffold, sharing_path
from mime.push._state import (
    MARK_SCHEME,
    DraftState,
    Uploaded,
    clear_state,
    draft_identifier,
    metadata_marks,
    read_state,
    staging,
    state_directory,
    write_state,
)

#: How often one file is started again from the beginning after its upload failed, before push stops.
UPLOAD_ATTEMPTS = 3
#: The flag that writes push's metadata over what was edited on the site.
OVERWRITE_FLAG = "--overwrite-site-edits"
#: Where each metadata field the author writes is kept in sharing.yaml, by its name in the record.
SHARING_KEYS = {"title": "title", "creators": "creators", "contributors": "contributors", "subjects": "keywords",
                "related_identifiers": "related"}
#: The longest value a refusal quotes of a field.
_SHOWN = 160

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


def _usable(service: Service) -> None:
    """Refuse a service that cannot be sent its token (``push.service``), before anything is read or sent."""
    problem = service.problem()
    if problem is not None:
        raise BuildRefused([_refuse("push.service", RefusalClass.USER, problem,
                                    "push to a service by its name, `mime push <folder> --to zenodo-sandbox`")])


def _token(service: Service) -> Token:
    """The token push sends ``service``, and only where MIME may go online (``MIME_OFFLINE`` closes push too). The two
    real services' stored tokens are never the same one: a token stored under the other's name is refused."""
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
    others = [each.credential for each in (ZENODO, ZENODO_SANDBOX) if each.credential != service.credential]
    if service.credential in (ZENODO.credential, ZENODO_SANDBOX.credential):
        for other in others:
            held = credentials.token(other)
            if held is not None and held.reveal() == found.reveal():
                raise BuildRefused([_refuse(
                    "push.token", RefusalClass.ENVIRONMENT,
                    f"the tokens held for {service.credential} and {other} are the same token, and a token is the "
                    f"token of one service's account: one of them is held under the wrong name; nothing was sent",
                    f"make each on its own service's site and store it under that service's name, `mime credentials "
                    f"set {service.credential}` and `mime credentials set {other}`")])
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


def _errors_refusal(service: Service, record: str, what: str, errors: Errors | list[object],
                    owner: Owner = field_owner) -> BuildRefused:
    """A refusal for each field the service names in ``errors``, by who owns it (``owner``): the author's in
    sharing.yaml or the experiment, to correct there; MIME's, its defect, said as one."""
    listed = errors.errors if isinstance(errors, Errors) else errors
    return BuildRefused(owned_refusals(service, f"{service.name} kept the draft {record} and refused part of {what}",
                                       listed, owner, "its answer listed errors"))


def _identifiers(draft: Json) -> list[str]:
    metadata = draft.get("metadata")
    held = cast(dict[str, object], metadata).get("identifiers") if isinstance(metadata, dict) else None
    return [str(cast(dict[str, object], each).get("identifier")) for each in cast(list[object], held)
            if isinstance(each, dict)] if isinstance(held, list) else []


def _ours(draft: Json, service: Service, state: DraftState) -> bool:
    """Whether ``draft`` carries the identifier push wrote into the draft it made for the record's experiment."""
    return draft_identifier(service, state.experiment, state.mark) in _identifiers(draft)


def _foreign(service: Service, state: DraftState) -> BuildRefused:
    expected = draft_identifier(service, state.experiment, state.mark)
    page = state.page or service.draft_page(str(state.record))
    return BuildRefused([_refuse(
        "push.draft-foreign", RefusalClass.USER,
        f"the draft {state.record} on {service.name} does not carry the identifier push wrote into the draft it made "
        f"for {plain(state.experiment)!r}, {expected}: push's record names another draft (another experiment's, or a "
        f"record edited or copied), or the identifier was removed on the site, so push changes and deletes nothing in "
        f"it", f"look at the draft, {page}: if it is this experiment's, add the alternate identifier {expected} back "
           f"on the site; if it is not, delete push's record, {state_directory(service, state.experiment)}, and run "
           f"`mime push <folder> --to {service.name}`, which makes a new draft")])


def _metadata(draft: Json) -> dict[str, object]:
    held = draft.get("metadata")
    return cast(dict[str, object], held) if isinstance(held, dict) else {}


def edited_fields(draft: Json, state: DraftState) -> list[str]:
    """The fields of ``draft``'s metadata that differ from what push last wrote: edited on the site since."""
    now, then = metadata_marks(draft), state.written
    return sorted(key for key in set(now) | set(then) if now.get(key) != then.get(key))


def _shown(value: object) -> str:
    if value in (None, "", [], {}):
        return "nothing"
    text = plain(value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False))
    return text if len(text) <= _SHOWN else text[:_SHOWN] + "..."


def _site_edits(service: Service, record: str, fields: list[str], site: dict[str, object],
                ours: dict[str, object]) -> BuildRefused:
    """A refusal for each field edited on the site, with its value there and the value push would write."""
    found: list[Refusal] = []
    flag = OVERWRITE_FLAG
    for name in fields:
        owner = FIELD_OWNERS.get(name, "mime")
        source = "sharing.yaml" if owner == "sharing" else "the experiment" if owner == "experiment" else "MIME"
        message = (f"the draft {record} on {service.name} was edited on the site since push last wrote it, and push "
                   f"would write over it: metadata.{name} on the site: {_shown(site.get(name))}; from {source}: "
                   f"{_shown(ours.get(name))}")
        if name in SHARING_KEYS:
            action = (f"to keep the site's, copy it into sharing.yaml's `{SHARING_KEYS[name]}` and push again; to "
                      f"drop it, write over it, `mime push <folder> --to {service.name} {flag}`")
        elif name == "description":
            action = (f"the description is sharing.yaml's `description` paragraph and the card, which MIME writes: to "
                      f"keep a paragraph of the site's, copy it into that key, then write over the rest, `mime push "
                      f"<folder> --to {service.name} {flag}`")
        elif name == "rights":
            action = (f"the licences are the files', declared where the experiment is built: to keep the site's, "
                      f"declare them there, exp.licence(...), and save again, built.save(folder); to drop the edit, "
                      f"write over it, `mime push <folder> --to {service.name} {flag}`")
        else:
            action = (f"sharing.yaml has no key for {name}, so writing over it removes the site's from the draft, "
                      f"`mime push <folder> --to {service.name} {flag}`; to keep it, leave the draft as it is and do "
                      f"not push it again")
        found.append(_refuse("push.metadata-edited", RefusalClass.USER, message, action))
    return BuildRefused(found)


def _published(service: Service, record: str) -> BuildRefused:
    return BuildRefused([_refuse(
        "push.published", RefusalClass.USER,
        f"the record {record} on {service.name} is published: a published record cannot be deleted or changed by "
        f"push, and a new version of it comes with a later MIME", f"look at it, `mime push <folder> --to "
                                                                  f"{service.name} --status`")])


class _Drafting:
    """One push's requests, its record kept current after every step that changes the service."""

    def __init__(self, client: Client, plan: Plan, state: DraftState | None, say: Say, overwrite: bool) -> None:
        self.client = client
        client.owner = plan.owner
        self.plan = plan
        self.service = plan.service
        self.state = state
        self.say = say
        self.overwrite = overwrite

    def payload(self, state: DraftState) -> Json:
        """The plan's metadata, with push's identifier of the draft among its alternate identifiers."""
        metadata = dict(cast(Json, self.plan.payload["metadata"]))
        metadata["identifiers"] = [{"identifier": draft_identifier(self.service, state.experiment, state.mark),
                                    "scheme": MARK_SCHEME}]
        return {**self.plan.payload, "metadata": metadata}

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
            elif not _ours(read[0], service, state):
                raise _foreign(service, state)
            else:
                found = read[0]
        if state is None or state.record is None or found is None:
            mark = state.mark if state is not None and state.creating else None
            if mark is not None:
                lost = draft_identifier(service, self.plan.experiment, mark)
                self.say(f"warning: an earlier push stopped while it asked {service.name} to make the draft, before "
                         f"its answer came, so {service.name} may hold a draft MIME has no record of: look at "
                         f"{service.site}/me/uploads, and delete one there that this push does not name (it carries "
                         f"the alternate identifier {lost})")
            fresh = DraftState(service.name, service.api, self.plan.experiment, creating=True)
            if mark is not None:
                fresh.mark = mark  # the draft made now carries the identifier the lost one may carry
            self.state = state = fresh
            self.save()
            draft, errors = self.client.create_draft(self.payload(state))
            record = str(draft["id"])
            self.state = state = DraftState(service.name, service.api, self.plan.experiment, record,
                                            _parent_id(draft), None, _page(draft, service, record), mark=state.mark,
                                            written=metadata_marks(draft))
            self.save()
            self.say(f"made the draft {record}: {state.page}")
            if errors:
                raise _errors_refusal(service, record, "its metadata", errors, self.plan.owner)
        else:
            record = state.record
            assert record is not None
            body = self.payload(state)
            edited = [name for name in edited_fields(found, state)
                      if _metadata(found).get(name) != cast(Json, body["metadata"]).get(name)]
            if edited and not self.overwrite:
                raise _site_edits(service, record, edited, _metadata(found), cast(Json, body["metadata"]))
            body["pids"] = found.get("pids", {})
            draft, errors = self.client.update_draft(record, body)
            state.written = metadata_marks(draft)
            self.save()
            if edited:
                self.say(f"wrote over what was edited on the site ({OVERWRITE_FLAG}): {', '.join(edited)}")
            self.say(f"wrote the metadata of the draft {record} again, from sharing.yaml and the folder")
            if errors:
                raise _errors_refusal(service, record, "its metadata", errors, self.plan.owner)
        doi = _doi_of(draft)
        if doi is None:
            try:
                draft = self._reserved(record)
            except Errors as answered:
                raise _errors_refusal(service, record, "the DOI's reservation", answered, self.plan.owner) from None
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

    def _reserved(self, record: str) -> Json:
        """The draft with its DOI reserved. An answer lost is never taken for a refusal: the draft is read again, and
        the reservation asked once more only where the draft holds no DOI."""
        for _ in range(2):
            try:
                return self.client.reserve_doi(record)
            except Dropped:
                pass
            read = self.client.read_draft(record)
            if read is not None and _doi_of(read[0]) is not None:
                return read[0]
        raise self.client.lost("POST", f"/records/{record}/draft/pids/doi", "twice, and the draft holds no DOI")

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

    def _start(self, record: str, key: str) -> None:
        """Start the entry ``key``; an answer lost is not taken for a failure where the files, read again, hold the
        entry (it was started, and is never started twice)."""
        try:
            self.client.start_file(record, key)
        except Dropped:
            if key not in self.client.files(record):
                raise

    def _send(self, record: str, file: RecordFile) -> None:
        """One file, whole, into an entry started for it; after a failure the entry is read again, and the file is
        kept where the service holds it complete and verified (an answer lost after the service acted), or else
        cleared and sent again from its start, at most :data:`UPLOAD_ATTEMPTS` times."""
        problem = ""
        mismatch = False
        for _ in range(UPLOAD_ATTEMPTS):
            entry: Json | None = None
            try:
                self._start(record, file.key)
                entry = self.client.send_content(record, file.key, file.path)
                entry = self.client.commit_file(record, file.key)
            except Dropped:
                problem, mismatch, entry = "its connection dropped, or its answer was lost", False, None
            except Gone:
                problem, mismatch, entry = "its entry was gone before it was complete", False, None
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
        """The draft holds exactly the planned files, each verified, and push's identifier; what its metadata is now is
        kept, for the next push to find what was edited on the site since."""
        state = self.state
        assert state is not None and state.record is not None
        read = self.client.read_draft(state.record)
        if read is None or not _ours(read[0], self.service, state):
            raise BuildRefused([_refuse(
                "push.service", RefusalClass.ENVIRONMENT,
                f"read again after the uploads, the draft {state.record} is "
                f"{'gone' if read is None else 'without the identifier push wrote into it'}",
                "look at the draft on the service's site, then run the same command again, `mime push <folder> --to "
                "<service>`")])
        state.written = metadata_marks(read[0])
        self.save()
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
         sleep: Callable[[float], None] = time.sleep, overwrite_site_edits: bool = False) -> Drafted | None:
    """Show the plan, then make or update the draft of the saved experiment in ``folder`` on ``service``, reserve its
    DOI and upload its files; never publish. Nothing is sent before the plan is shown and confirmed (``yes``, or
    ``confirm`` answering true); without either, refused (``push.unconfirmed``). ``None`` when the plan is declined.
    Metadata edited on the site since push last wrote it is refused (``push.metadata-edited``), and written over
    where ``overwrite_site_edits``."""
    _usable(service)
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
            if state is not None and state.page:
                say(state.page)
            return None
    return send(plan, state, Client(service, token, transport, pause=pause, sleep=sleep), say,
                overwrite_site_edits=overwrite_site_edits)


def send(plan: Plan, state: DraftState | None, client: Client, say: Say = _quiet, *,
         overwrite_site_edits: bool = False) -> Drafted:
    """What ``push`` does once its plan is shown and confirmed: make or update the draft, reserve its DOI, upload each
    file that differs, and read the draft again to check it holds exactly the plan's files, verified. Whatever
    happens, once there is a draft its link is the last line said."""
    drafting = _Drafting(client, plan, state, say, overwrite_site_edits)
    try:
        state, _ = drafting.draft()
        sent, kept = drafting.upload()
        drafting.check()
    except BaseException:
        if drafting.state is not None and drafting.state.page:
            say(drafting.state.page)
        raise
    assert state.record is not None and state.page is not None and state.doi is not None
    say(f"The draft {state.record} on {plan.service.name} holds the {len(plan.files)} record files, each md5 "
        f"verified; its DOI {state.doi} is reserved, and nothing is published. Look at it: {state.page}")
    say(state.page)
    return Drafted(state.record, state.page, state.doi, state.concept, tuple(sent), tuple(kept))


def status(folder: str | os.PathLike[str], service: Service, *, say: Say = _quiet, transport: Transport | None = None,
           pause: float | None = None) -> DraftState | None:
    """Say what push knows of the experiment's draft on ``service`` and the draft as the service holds it: its page,
    reserved DOI, concept, title, creators and licences, and each file with its size and md5, against what push
    uploaded and verified, and whether it is the draft push made and what was edited on the site since push last wrote
    it. Changes nothing; ends with the draft's link."""
    _usable(service)
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
            say(f"{service.site}/records/{state.record}")
        else:
            say(f"the draft {state.record} MIME made is no longer on {service.name} (discarded on the site?); the next "
                f"push makes a new one")
        return state
    draft, errors = read
    held = _metadata(draft)
    page = _page(draft, service, state.record)
    say(f"The draft {state.record} on {service.name}: {page}")
    if not _ours(draft, service, state):
        say(f"  it does not carry the identifier push wrote into the draft it made, "
            f"{draft_identifier(service, experiment, state.mark)}: push's record may name another draft, and push "
            f"changes nothing in it")
    elif edited := edited_fields(draft, state):
        say(f"  edited on the site since push last wrote it: {', '.join(plain(each) for each in edited)} (a push "
            f"refuses to write over them without {OVERWRITE_FLAG})")
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
    say(page)
    return state


def discard(folder: str | os.PathLike[str], service: Service, *, yes: bool = False, confirm: Ask | None = None,
            say: Say = _quiet, transport: Transport | None = None, pause: float | None = None) -> bool:
    """Delete the experiment's draft on ``service`` (only a draft can be deleted) and push's record of it: whether a
    draft was deleted. Only the draft push made is deleted (``push.draft-foreign`` otherwise). Ends with a link: the
    draft's when it is kept, the account's drafts when it is deleted."""
    _usable(service)
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
    if not _ours(read[0], service, state):
        raise _foreign(service, state)
    page = state.page or service.draft_page(state.record)
    question = f"Delete the draft {state.record} on {service.name} and its files ({page})?"
    if not yes:
        if confirm is None:
            raise _unconfirmed(f"the deletion of the draft {state.record}")
        if not confirm(question):
            say("Not deleted.")
            say(page)
            return False
    client.delete_draft(state.record)
    clear_state(service, experiment)
    say(f"deleted the draft {state.record} on {service.name}, and push's record of it")
    say(f"{service.site}/me/uploads")
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


__all__ = ["OVERWRITE_FLAG", "SHARING_KEYS", "UPLOAD_ATTEMPTS", "Drafted", "discard", "edited_fields",
           "experiment_name", "init", "push", "send", "status"]
