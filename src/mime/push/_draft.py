"""Drafts: make or update one (a first version, or a new version of a published record), reserve its DOI, upload its
files; read it as the service holds it; discard it. Publishing, the one step that cannot be undone, is
:mod:`mime.push._publish`'s.

Every step is check-then-act against the service's state, read again on every run, so a push stopped at any request
is resumed by running the same command (design §2.6), at the level of whole files: a file in the draft whose size and
md5 are the planned file's was uploaded and verified, and is not sent again; any other is deleted and sent whole, into
an entry started for it (Zenodo keeps no partial upload, ignores ``Content-Range``, and deletes a completed file that
is sent again). After each upload the service's md5 and size must equal MIME's; MIME's sha256 is kept in push's record,
since Zenodo states md5 only. A file in the draft that this plan does not hold is deleted where push uploaded it
earlier, and refused where it did not (``push.draft-changed``): one added on the site is the author's to remove.

**A private draft is made without a question** (design #199 (a)): the plan is shown before anything is written, and
``--discard`` undoes it. **Nothing changed, one line**: a push whose folder and ``sharing.yaml`` are what push last sent
reads the draft and, finding it as push left it, says so in one line, writing nothing; anything to write is shown in
the plan first. The plan is short by default (what goes up, what travels as a reference, warnings on lines of their
own), and in full with ``full``.

**The bundle and the description name the record.** The draft is made and its DOI reserved before the bundle is packed
with the staged lock that names that record (:mod:`mime.push._stage`), and the description (the card) is written again
naming it, so a first push packs the folder twice, once for the plan and once with the reserved DOI, and writes the
metadata twice.

**Versions.** Once a version is published, a push whose content changed makes a new version of it (``POST
.../versions``), links the files of the version before into it (``files-import``) and uploads only what differs, so an
unchanged asset is linked, never sent again; a push whose content did not change says ``unchanged since <DOI>`` in one
line.

Before it writes to a draft or deletes one, push checks the draft is the one it made: the draft carries the identifier
push wrote into it, a UUID of the service, the experiment's name and push's mark (``push.draft-foreign`` otherwise).
Each field of its metadata edited on the site since push last wrote it is shown against what push would write, and
kept (written into ``sharing.yaml``) or written over, field by field (:mod:`mime.push._edits`). A new version's draft
holds the service's copy of the published record: what was edited on the record's page is found against the published
marks, and settled the same way, until it is settled, however often the same command runs; only the fields the service
sets anew on a version (:data:`VERSION_RESETS`) are left out. An edit made while the files upload is not push's own: it
is found by the next push. Every push, ``--status`` and ``--discard`` ends with a link on a line of its own.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from mime._terminal import plain
from mime.assets import size_text
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.credentials import Token
from mime.push._client import Client, Dropped, Errors, Gone, Json, Owner, Transport, owned_refusals
from mime.push._edits import (
    SHARING_KEYS,
    apart,
    as_yaml,
    differs,
    field_value,
    kept,
    words_of,
    write_back,
    written_value,
)
from mime.push._fields import access_words
from mime.push._owners import ACCESS_FIELD, FIELD_OWNERS, field_owner
from mime.push._plan import Plan, RecordFile, make_plan, restaged, with_sharing
from mime.push._service import TEST_PREFIX, ZENODO, ZENODO_SANDBOX, Service
from mime.push._sharing import read_sharing, scaffold, sharing_path
from mime.push._state import (
    ACCESS,
    CUSTOM,
    MARK_SCHEME,
    DraftState,
    Published,
    Uploaded,
    clear_state,
    draft_identifier,
    metadata_marks,
    payload_identity,
    read_state,
    staging,
    state_directory,
    write_state,
)
from mime.saved._schema import PublicationRecord

#: How often one file is started again from the beginning after its upload failed, before push stops.
UPLOAD_ATTEMPTS = 3
#: The flag that writes push's metadata over what was edited on the site.
OVERWRITE_FLAG = "--overwrite-site-edits"
#: The flag that keeps, without asking, each edit made on the site that sharing.yaml can hold.
KEEP_FLAG = "--keep-site-edits"
#: The flag that shows the full plan.
FULL_FLAG = "--full-plan"
#: The fields the service may set anew on a new version's copy of the published record, so no edit is read in them. Each
#: other field of the copy is held to the published marks. Which fields these are is the sandbox session's to measure.
VERSION_RESETS = frozenset({"publication_date", "version"})

type Say = Callable[[str], None]
type Ask = Callable[[str], bool]


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> Refusal:
    return Refusal(code, category, message, action)


def _quiet(line: str) -> None:
    pass


@dataclass(frozen=True)
class Drafted:
    """What a push left on the service: the draft's id, its page, its reserved DOI, its concept's id, the files it
    uploaded this run, those it found already uploaded and verified (or linked from the version before), whether it
    wrote anything, and the version's number. A push of a published experiment whose content did not change leaves the
    published record's id, page and DOI."""

    record: str
    page: str
    doi: str
    concept: str | None
    uploaded: tuple[str, ...]
    kept: tuple[str, ...]
    changed: bool = True
    version: int = 1


@dataclass(frozen=True)
class Choices:
    """How the edits made on the site are settled: asked, field by field, in a terminal (``ask``); or by a flag, every
    edit written over (``overwrite``) or each that sharing.yaml can hold kept (``keep``). With none, an edit stops the
    push (``push.metadata-edited``)."""

    ask: Ask | None = None
    overwrite: bool = False
    keep: bool = False


def experiment_name(folder: Path) -> str:
    from mime.saved import read_folder

    return read_folder(folder, absent_referenced=True).document["experiment"]


def usable(service: Service) -> None:
    """Refuse a service that cannot be sent its token (``push.service``), before anything is read or sent."""
    problem = service.problem()
    if problem is not None:
        raise BuildRefused([_refuse("push.service", RefusalClass.USER, problem,
                                    "push to a service by its name, `mime push <folder> --to zenodo-sandbox`")])


def token_of(service: Service) -> Token:
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
            f"was sent", f"make a personal access token with the scopes deposit:write and deposit:actions on the "
                         f"service's site and store it, `mime credentials set {service.credential}`")])
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


def unconfirmed(what: str, action: str = "read it, then run the same command again with --yes") -> BuildRefused:
    return BuildRefused([_refuse(
        "push.unconfirmed", RefusalClass.USER,
        f"{what} was shown and not confirmed, and there is no terminal to ask in: nothing was sent", action)])


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


def entry_verified(entry: Json | None, file: RecordFile) -> bool:
    """Whether the service holds ``file`` complete: its stated size and md5 are MIME's."""
    return entry is not None and entry.get("status", "completed") == "completed" and entry.get("size") == file.size \
        and entry.get("checksum") == f"md5:{file.md5}"


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


def ours(draft: Json, service: Service, state: DraftState) -> bool:
    """Whether ``draft`` carries the identifier push wrote into the draft it made for the record's experiment."""
    return draft_identifier(service, state.experiment, state.mark) in _identifiers(draft)


def foreign(service: Service, state: DraftState) -> BuildRefused:
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


def metadata_of(draft: Json) -> dict[str, object]:
    held = draft.get("metadata")
    return cast(dict[str, object], held) if isinstance(held, dict) else {}


def edited_fields(draft: Json, state: DraftState | Published) -> list[str]:
    """The fields of ``draft``'s metadata that differ from what push last wrote: edited on the site since."""
    now, then = metadata_marks(draft), state.written
    return sorted(key for key in set(now) | set(then) if now.get(key) != then.get(key))


def _source(name: str) -> str:
    owner = FIELD_OWNERS.get(name, "sharing" if name.startswith(CUSTOM) or name == ACCESS_FIELD else "mime")
    return "sharing.yaml" if owner == "sharing" else "the experiment" if owner == "experiment" else "MIME"


def label(name: str) -> str:
    return name if name.startswith(CUSTOM) or name == ACCESS else f"metadata.{name}"


def site_edits(service: Service, record: str, fields: list[str], site: Json, ours_: Json) -> BuildRefused:
    """A refusal for each field edited on the site, with its value there and the value push would write."""
    found: list[Refusal] = []
    for name in fields:
        on_site, from_ours = apart(words_of(field_value(site, name)), words_of(field_value(ours_, name)))
        message = (f"the draft {record} on {service.name} was edited on the site since push last wrote it, and push "
                   f"would write over it: {label(name)} on the site: {on_site}; from {_source(name)}: {from_ours}")
        if name in SHARING_KEYS:
            action = (f"keep it, `mime push <folder> --to {service.name} {KEEP_FLAG}` (written into sharing.yaml's "
                      f"`{SHARING_KEYS[name]}`), or run push in a terminal, which asks; to drop it, write over it, "
                      f"`mime push <folder> --to {service.name} {OVERWRITE_FLAG}`")
        elif name == "rights":
            action = (f"the licences are the files', declared where the experiment is built: to keep the site's, "
                      f"declare them there, exp.licence(...), and save again, built.save(folder); to drop the edit, "
                      f"write over it, `mime push <folder> --to {service.name} {OVERWRITE_FLAG}`")
        else:
            action = (f"sharing.yaml has no key for {name}, so writing over it removes the site's from the draft, "
                      f"`mime push <folder> --to {service.name} {OVERWRITE_FLAG}`; to keep it, leave the draft as it "
                      f"is and do not push it again")
        found.append(_refuse("push.metadata-edited", RefusalClass.USER, message, action))
    return BuildRefused(found)


def errors_refusal(service: Service, record: str, what: str, errors: Errors, owner: Owner = field_owner) -> BuildRefused:
    return _errors_refusal(service, record, what, errors, owner)


def published_unrecorded(service: Service, record: str) -> BuildRefused:
    return BuildRefused([_refuse(
        "push.published", RefusalClass.USER,
        f"the draft {record} on {service.name} is published, and push's record of it says it is a draft: an earlier "
        f"publish ended before push wrote it down, or it was published on the site", f"check it and record it, "
        f"`mime push <folder> --to {service.name} --publish`, which publishes nothing again")])


def marked(payload: Json, service: Service, state: DraftState) -> Json:
    """``payload`` with push's identifier of the draft among its alternate identifiers."""
    metadata = dict(cast(Json, payload["metadata"]))
    metadata["identifiers"] = [{"identifier": draft_identifier(service, state.experiment, state.mark),
                                "scheme": MARK_SCHEME}]
    return {**payload, "metadata": metadata}


def publication(service: Service, state: DraftState) -> PublicationRecord | None:
    """The record a draft will be published as, as the staged lock names it, once its DOI is reserved."""
    if state.record is None or state.doi is None or state.concept is None:
        return None
    held: dict[str, object] = {"service": service.name, "record": state.record, "doi": state.doi,
                               "concept": state.concept, "version": state.version}
    if state.published is not None and state.published.concept_doi:
        held["concept_doi"] = state.published.concept_doi
    return cast(PublicationRecord, held)


def published_publication(service: Service, published: Published) -> PublicationRecord:
    """The publication the published version's bundle names, as its lock holds it."""
    if published.publication:
        return cast(PublicationRecord, dict(published.publication))
    return cast(PublicationRecord, {"service": service.name, "record": published.record, "doi": published.doi,
                                    "concept": published.concept, "version": published.version})


def planned(root: Path, service: Service, state: DraftState | None) -> Plan:
    """The plan of the folder as this push packs it: with the staged lock of the draft push made (its DOI reserved),
    or of the version last published (to tell whether the content changed since), or with the folder's own lock."""
    sharing = read_sharing(root)
    experiment = experiment_name(root)
    where = staging(service, experiment)
    if state is not None and state.record is not None:
        own: frozenset[str] = frozenset({state.published.doi}) if state.published is not None else frozenset()
        return make_plan(root, service, sharing, where, publication=publication(service, state),
                         version=state.version, own=own)
    if state is not None and state.published is not None:
        published = state.published
        return make_plan(root, service, sharing, where, publication=published_publication(service, published),
                         version=published.version, own=frozenset({published.doi}))
    return make_plan(root, service, sharing, where)


def unchanged(plan: Plan, published: Published) -> bool:
    """Whether the folder, packed as the published version was, makes the published version's files: the same keys,
    each of the same size and md5."""
    return {file.key: (file.size, file.md5) for file in plan.files} == \
        {key: (held.size, held.md5) for key, held in published.files.items()}


class Drafting:
    """One push's requests, its record kept current after every step that changes the service. ``show`` prints the
    plan, once, before the first request that writes; a push with nothing to write shows none."""

    def __init__(self, client: Client, plan: Plan, state: DraftState | None, say: Say, choices: Choices,
                 show: Callable[[], None], full: bool) -> None:
        self.client = client
        client.owner = plan.owner
        self.plan: Plan = plan
        self.service = plan.service
        self.state: DraftState | None = state
        self.say = say
        self.choices = choices
        self._show = show
        self.full = full
        self.shown = False
        self.wrote = False
        self.made = False
        self.linked: list[str] = []

    def show(self) -> None:
        if not self.shown:
            self.shown = True
            self._show()

    def detail(self, line: str) -> None:
        """A line of the full plan's detail: said with ``full`` once the plan is shown (a push with nothing to write
        says one line, however asked)."""
        if self.full and self.shown:
            self.say(line)

    def payload(self, state: DraftState) -> Json:
        """The plan's metadata, with push's identifier of the draft among its alternate identifiers."""
        return marked(self.plan.payload, self.service, state)

    def save(self) -> None:
        assert self.state is not None
        write_state(self.service, self.state)

    # --- the edits made on the site ------------------------------------------------------------------------------

    def settle(self, found: Json, record: str) -> list[str]:
        """Each field of ``found`` edited on the site since push last wrote it that push would write over, settled:
        kept (written into sharing.yaml, the plan made again from it), written over, or the push stopped. The fields
        written over. A new version's draft whose metadata push has not written yet is the service's copy of the
        published record, held to the published marks (:data:`VERSION_RESETS` aside), and its description to the
        published card."""
        state = self.state
        assert state is not None
        body = self.payload(state)
        copied = state.published is not None and not state.sent
        card = state.published.card if copied and state.published is not None else None
        edited = [name for name in edited_fields(found, state) if differs(name, found, body)
                  and not (copied and name in VERSION_RESETS)]
        if not edited:
            return []
        self.show()
        keep: dict[str, object] = {}
        kept_names: list[str] = []
        over: list[str] = []
        stop: list[str] = []
        for name in edited:
            updates = kept(name, found, self.plan, card)
            if self.choices.overwrite:
                over.append(name)
                continue
            if self.choices.keep:
                (kept_names if updates is not None else stop).append(name)
                keep.update(updates or {})
                continue
            if self.choices.ask is None:
                stop.append(name)
                continue
            self.say(f"! {label(name)} was edited on the site since push last wrote it:")
            if updates is not None:
                for key, value in updates.items():
                    on_site, in_file = apart(as_yaml(value), as_yaml(written_value(self.plan, key)))
                    self.say(f"    on the site:  {key}: {on_site}")
                    self.say(f"    sharing.yaml: {key}: {in_file}")
            else:
                on_site, ours_now = apart(words_of(field_value(found, name)), words_of(field_value(body, name)))
                self.say(f"    on the site:  {on_site}")
                self.say(f"    {_source(name)} writes: {ours_now}")
            if updates is not None:
                if self.choices.ask("Write sharing.yaml's over the site's? (no keeps the site's, written into "
                                    f"sharing.yaml's `{SHARING_KEYS[name]}`)"):
                    over.append(name)
                else:
                    kept_names.append(name)
                    keep.update(updates)
            elif self.choices.ask(f"sharing.yaml cannot hold it ({_source(name)} writes it): write over it? (no stops "
                                  f"the push)"):
                over.append(name)
            else:
                stop.append(name)
        if stop:
            raise site_edits(self.service, record, stop, found, body)
        if keep:
            self.plan = write_back(self.plan, keep, kept_names, found, card)
            self.say(f"kept what was edited on the site, written into {sharing_path(self.plan.folder)}: "
                     f"{', '.join(label(name) for name in kept_names)}")
        return over

    # --- the draft -----------------------------------------------------------------------------------------------

    def draft(self) -> DraftState:
        """Make the draft (or a new version of the published record), or update the one push made earlier; reserve its
        DOI; and pack the bundle with the staged lock naming it."""
        service, state = self.service, self.state
        found: Json | None = None
        if state is not None and state.record is not None:
            read = self.client.read_draft(state.record)
            if read is None:
                if self.client.read_record(state.record) is not None:
                    raise published_unrecorded(service, state.record)
                self.show()
                self.say(f"! the draft {state.record} MIME made earlier is no longer on {service.name} (discarded on "
                         f"the site?): a new one is made")
                state = replace(state, record=None, doi=None, page=None, files={}, written={}, sent="", confirmed=[])
                self.state = state
            elif read[0].get("is_published") is True:
                raise published_unrecorded(service, state.record)
            elif not ours(read[0], service, state) and (state.published is None or state.sent):
                raise foreign(service, state)  # a version's draft push has not written yet holds the service's copy
            else:
                found = read[0]
        if state is None or state.record is None or found is None:
            draft = self._new_version() if state is not None and state.published is not None else self._create()
        else:
            draft = self.update(found)
        state = self.state
        assert state is not None and state.record is not None
        record = state.record
        doi = doi_of(draft)
        if doi is None:
            self.show()
            try:
                draft = self._reserved(record)
            except Errors as answered:
                raise _errors_refusal(service, record, "the DOI's reservation", answered, self.plan.owner) from None
            doi = doi_of(draft)
            if doi is None:
                raise BuildRefused([_refuse(
                    "push.service", RefusalClass.ENVIRONMENT,
                    f"{service.name} answered the DOI's reservation for the draft {record} with no DOI",
                    "run the same command again, `mime push <folder> --to <service>`")])
            self.wrote = True
            self.detail(f"reserved the DOI {doi} for the draft {record} (registered only when it is published)")
        if service.production and doi.startswith(TEST_PREFIX):
            raise BuildRefused([_refuse(
                "push.service", RefusalClass.ENVIRONMENT,
                f"{service.name}, a production service, reserved {doi}, under DataCite's test prefix: push stops "
                f"rather than give a production record a DOI that does not resolve",
                "look at the draft, `mime push <folder> --to zenodo --status`, and discard it, `mime push <folder> "
                "--to zenodo --discard`")])
        if state.doi is not None and doi != state.doi:
            self.show()
            self.say(f"! the draft holds the DOI {plain(doi)}, and push reserved {state.doi}: changed on the site, so "
                     f"the bundle is packed again naming the draft's")
        state.doi = doi
        state.concept = parent_id(draft) or state.concept
        self.save()
        staged = publication(service, state)
        if staged is not None and self.plan.staging is not None and self.plan.publication != staged:
            self.plan = restaged(self.plan, staged)
            if payload_identity(self.payload(state)) != state.sent:
                self._write(record, draft, [])  # the card names the record now
        return state

    def _create(self) -> Json:
        """A first version's draft, made with the plan's metadata."""
        service, state = self.service, self.state
        self.show()
        mark = state.mark if state is not None and state.creating else None
        if mark is not None:
            lost = draft_identifier(service, self.plan.experiment, mark)
            self.say(f"! an earlier push stopped while it asked {service.name} to make the draft, before its answer "
                     f"came, so {service.name} may hold a draft MIME has no record of: look at "
                     f"{service.site}/me/uploads, and delete one there that this push does not name (it carries the "
                     f"alternate identifier {lost})")
        fresh = DraftState(service.name, service.api, self.plan.experiment, creating=True)
        if mark is not None:
            fresh.mark = mark  # the draft made now carries the identifier the lost one may carry
        self.state = state = fresh
        self.save()
        body = self.payload(state)
        draft, errors = self.client.create_draft(body)
        record = str(draft["id"])
        self.state = state = DraftState(service.name, service.api, self.plan.experiment, record, parent_id(draft),
                                        None, page_of(draft, service, record), mark=state.mark,
                                        written=metadata_marks(draft), sent=payload_identity(body))
        self.save()
        self.made = self.wrote = True
        if errors:
            raise _errors_refusal(service, record, "its metadata", errors, self.plan.owner)
        return draft

    def _new_version(self) -> Json:
        """A new version of the published record: its draft (the concept's open one, where the service holds one),
        numbered as the service counts it where it says, the edits made on the site to the published record's metadata
        settled, the files of the version before linked into it, and its metadata written. Push's record holds the
        published marks until the edits are settled, so a push stopped at one stops at it again."""
        service, state = self.service, self.state
        assert state is not None and state.published is not None
        published = state.published
        self.show()
        fresh = replace(state, record=None, doi=None, page=None, files={}, written={}, sent="", confirmed=[],
                        creating=True, version=published.version + 1)
        self.state = fresh
        self.save()
        try:
            draft = self.client.new_version(published.record)
        except Dropped:
            raise self.client.lost("POST", f"/records/{published.record}/versions", "the connection dropped") from None
        except Gone:
            raise BuildRefused([_refuse(
                "push.service", RefusalClass.ENVIRONMENT,
                f"the published record {published.record} is no longer on {service.name} (deleted there, or a tombstone "
                f"now), so a new version of it cannot be made", f"look at its page, {published.page}; to start again as "
                f"a new concept, delete push's record, {state_directory(service, state.experiment)}, and run `mime "
                f"push <folder> --to {service.name}`")]) from None
        except Errors as answered:
            raise _errors_refusal(service, published.record, "the new version", answered, self.plan.owner) from None
        record = str(draft["id"])
        number = version_index(draft, published)
        if number != published.version + 1:
            self.say(f"! {service.name} counts this as version {number}, and push's last was {published.version}: a "
                     f"version was published on the site since")
            self.plan = with_sharing(replace(self.plan, version=number), self.plan.sharing)
        self.state = replace(fresh, record=record, concept=parent_id(draft) or published.concept,
                             page=page_of(draft, service, record), creating=False, version=number,
                             written=dict(published.written))
        self.save()
        self.made = self.wrote = True
        return self.update(draft)

    def _link_earlier_files(self, record: str) -> None:
        """The files of the version before in a new version's draft: linked by an import where the draft holds no file;
        where it holds some (the concept's open draft, or an import that was done), those that are the published
        version's. A file added on the site is neither: :meth:`upload` refuses it (``push.draft-changed``)."""
        state = self.state
        assert state is not None and state.published is not None
        published = state.published
        held = self.client.files(record)
        linked = {key: entry for key, entry in held.items() if key in published.files and
                  (entry.get("size"), entry.get("checksum")) == (published.files[key].size,
                                                                 f"md5:{published.files[key].md5}")} \
            if held else self.client.import_files(record)
        self.linked = sorted(linked)
        for key, entry in linked.items():
            size, checksum = entry.get("size"), str(entry.get("checksum", ""))
            if isinstance(size, int) and checksum.startswith("md5:"):
                state.files[key] = Uploaded(size, checksum.removeprefix("md5:"), "")
        self.save()
        self.detail(f"linked the {len(linked)} files of version {published.version} into the draft {record}: "
                    f"{', '.join(plain(key) for key in sorted(linked))}")

    def update(self, found: Json, *, dated: bool = False) -> Json:
        """The draft push made earlier, its edits made on the site settled, and its metadata written where it differs
        from what push last wrote (or, where ``dated``, where its publication date is not the plan's: the day it is
        published)."""
        state = self.state
        assert state is not None and state.record is not None
        copied = state.published is not None and not state.sent
        over = self.settle(found, state.record)
        if copied:
            self._link_earlier_files(state.record)
        return self.written_where_needed(found, over, dated=dated)

    def written_where_needed(self, found: Json, over: list[str], *, dated: bool = False) -> Json:
        """The draft's metadata written, where something was written over, or it is not what push last wrote, or (where
        ``dated``) its publication date is not the day's."""
        state = self.state
        assert state is not None and state.record is not None
        date = metadata_of(found).get("publication_date") == metadata_of(self.plan.payload).get("publication_date")
        if not over and payload_identity(self.payload(state)) == state.sent and (date or not dated):
            return found
        if not over and state.sent == "" and state.published is None and edited_fields(found, state) == [] and \
                not any(differs(name, found, self.payload(state)) for name in SHARING_KEYS):
            state.sent = payload_identity(self.payload(state))  # a record of P2a's, which kept no identity
            self.save()
            return found
        return self._write(state.record, found, over)

    def _write(self, record: str, found: Json, over: list[str]) -> Json:
        state = self.state
        assert state is not None
        self.show()
        body = self.payload(state)
        identity = payload_identity(body)
        body["pids"] = found.get("pids", {})
        chosen = found.get("files")
        if isinstance(chosen, dict):  # a preview chosen on the draft's page: a PUT without it would reset it
            body["files"] = {**cast(Json, body["files"]), **{
                key: value for key, value in cast(Json, chosen).items() if key == "default_preview" and value}}
        draft, errors = self.client.update_draft(record, body)
        state.written = metadata_marks(draft)
        state.sent = identity
        self.save()
        self.wrote = True
        if over:
            self.say(f"! wrote over what was edited on the site: {', '.join(label(name) for name in over)}")
        self.detail(f"wrote the metadata of the draft {record}, from sharing.yaml and the folder")
        if errors:
            raise _errors_refusal(self.service, record, "its metadata", errors, self.plan.owner)
        return draft

    def _reserved(self, record: str) -> Json:
        """The draft with its DOI reserved. An answer lost is never taken for a refusal: the draft is read again, and
        the reservation asked once more only where the draft holds no DOI."""
        for _ in range(2):
            try:
                return self.client.reserve_doi(record)
            except Dropped:
                pass
            read = self.client.read_draft(record)
            if read is not None and doi_of(read[0]) is not None:
                return read[0]
        raise self.client.lost("POST", f"/records/{record}/draft/pids/doi", "twice, and the draft holds no DOI")

    # --- the files -----------------------------------------------------------------------------------------------

    def upload(self) -> tuple[list[str], list[str]]:
        """Each planned file into the draft, unless it is there already, verified: the files sent and those kept."""
        state = self.state
        assert state is not None and state.record is not None
        record = state.record
        held = self.client.files(record)
        planned_keys = {file.key for file in self.plan.files}
        strays = sorted(key for key in held if key not in planned_keys)
        unknown = [key for key in strays if key not in state.files and key not in self.linked]
        if unknown:
            raise BuildRefused([_refuse(
                "push.draft-changed", RefusalClass.USER,
                f"the draft {record} on {self.service.name} holds {', '.join(plain(key) for key in unknown)}, which "
                f"this plan does not and push did not upload (added on the site?)",
                f"remove it on the draft's page, {state.page}, or start again from no draft, `mime push <folder> --to "
                f"{self.service.name} --discard`")])
        sent: list[str] = []
        kept_keys: list[str] = []
        for key in strays:
            self.show()
            self.client.delete_file(record, key)
            state.files.pop(key, None)
            self.save()
            self.wrote = True
            self.detail(f"{key}: deleted from the draft (push put it there earlier, and this plan does not hold it)")
        for file in self.plan.files:
            entry = held.get(file.key)
            mine = Uploaded(file.size, file.md5, file.sha256)
            if entry_verified(entry, file):
                if state.files.get(file.key) != mine:
                    state.files[file.key] = mine
                    self.save()
                kept_keys.append(file.key)
                where = "linked from the version before" if file.key in self.linked else "already in the draft"
                self.detail(f"{file.key}: {where}, its md5 {file.md5} and size verified, not sent again")
                continue
            self.show()
            if state.files.pop(file.key, None) is not None:
                self.save()
            if entry is not None:
                self.client.delete_file(record, file.key)
            self._send(record, file)
            state.files[file.key] = mine
            self.save()
            self.wrote = True
            sent.append(file.key)
            self.detail(f"{file.key}: uploaded, {size_text(file.size)}, its md5 {file.md5} equal to the one "
                        f"{self.service.name} states")
        return sent, kept_keys

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
            if entry is not None and entry_verified(entry, file):
                return
            if entry is not None:
                problem = (f"{self.service.name} states size {entry.get('size')} and "
                           f"{plain(str(entry.get('checksum')))}, and MIME sent {file.size} bytes of md5 {file.md5}")
                mismatch = True
            now = self.client.files(record).get(file.key)
            if now is not None and entry_verified(now, file):
                return
            if now is not None:
                self.client.delete_file(record, file.key)
            self.say(f"! {file.key}: {problem}; its entry cleared, and the file sent again from its start (Zenodo "
                     f"keeps no partial upload)")
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
        """The draft holds exactly the planned files, each verified, and push's identifier. What its metadata is now is
        kept, for the next push to find what is edited on the site after this; a field that is not what push wrote was
        edited while the files uploaded, and keeps its earlier mark, so the next push finds that too."""
        state = self.state
        assert state is not None and state.record is not None
        if not self.wrote:
            return
        read = self.client.read_draft(state.record)
        if read is None or not ours(read[0], self.service, state):
            raise BuildRefused([_refuse(
                "push.service", RefusalClass.ENVIRONMENT,
                f"read again after the uploads, the draft {state.record} is "
                f"{'gone' if read is None else 'without the identifier push wrote into it'}",
                "look at the draft on the service's site, then run the same command again, `mime push <folder> --to "
                "<service>`")])
        now, body = metadata_marks(read[0]), self.payload(state)
        for name in set(now) | set(state.written):
            if now.get(name) != state.written.get(name) and differs(name, read[0], body):
                now.pop(name, None)
                if name in state.written:
                    now[name] = state.written[name]
        state.written = now
        self.save()
        held = self.client.files(state.record)
        wrong = [file.key for file in self.plan.files if not entry_verified(held.get(file.key), file)]
        extra = sorted(set(held) - {file.key for file in self.plan.files})
        if wrong or extra:
            raise BuildRefused([_refuse(
                "push.service", RefusalClass.ENVIRONMENT,
                f"read again after the uploads, the draft {state.record} does not hold the plan's files: "
                f"{', '.join(plain(key) for key in wrong + extra)} differ", "run the same command again, `mime push "
                                                                             "<folder> --to <service>`")])


def version_index(draft: Json, published: Published) -> int:
    """A new version's number: the service's, where its draft states one past the last push published; one past it
    otherwise."""
    versions = draft.get("versions")
    index = cast(dict[str, object], versions).get("index") if isinstance(versions, dict) else None
    return index if isinstance(index, int) and index > published.version else published.version + 1


def parent_id(draft: Json) -> str | None:
    parent = draft.get("parent")
    found = cast(dict[str, object], parent).get("id") if isinstance(parent, dict) else None
    return str(found) if found is not None else None


def doi_of(draft: Json) -> str | None:
    pids = draft.get("pids")
    doi = cast(dict[str, object], pids).get("doi") if isinstance(pids, dict) else None
    identifier = cast(dict[str, object], doi).get("identifier") if isinstance(doi, dict) else None
    return identifier if isinstance(identifier, str) and identifier.startswith("10.") else None


def concept_doi_of(record: Json) -> str | None:
    parent = record.get("parent")
    return doi_of(cast(Json, parent)) if isinstance(parent, dict) else None


def page_of(draft: Json, service: Service, record: str) -> str:
    links = draft.get("links")
    page = cast(dict[str, object], links).get("self_html") if isinstance(links, dict) else None
    return page if isinstance(page, str) and page.startswith(service.site + "/") else service.draft_page(record)


def record_page(service: Service, record: str) -> str:
    return f"{service.site}/records/{record}"


def _show_plan(plan_of: Callable[[], Plan], state: DraftState | None, say: Say, full: bool) -> Callable[[], None]:
    def show() -> None:
        current = plan_of()
        for line in (current.lines(state) if full else current.summary(state)):
            say(line)
    return show


def push(folder: str | os.PathLike[str], service: Service, *, yes: bool = False, confirm: Ask | None = None,
         say: Say = _quiet, transport: Transport | None = None, pause: float | None = None,
         sleep: Callable[[float], None] = time.sleep, overwrite_site_edits: bool = False,
         keep_site_edits: bool = False, full: bool = False) -> Drafted:
    """Make or update the draft of the saved experiment in ``folder`` on ``service`` (a new version of it, once one is
    published and its content changed), reserve its DOI and upload its files; never publish. The plan is shown before
    anything is written (in full where ``full``), and a private draft asks no question. A push with nothing to write
    says so in one line. Each field edited on the site since push last wrote it is asked about in a terminal
    (``confirm``), kept (``keep_site_edits``) or written over (``overwrite_site_edits``), and otherwise stops the push
    (``push.metadata-edited``); ``yes`` answers none of those questions."""
    del yes  # a private draft asks nothing, and --yes never settles an edit made on the site
    usable(service)
    root = Path(os.path.abspath(folder))
    experiment = experiment_name(root)
    state = read_state(service, experiment)
    plan = planned(root, service, state)
    if state is not None and state.record is None and state.published is not None:
        published = state.published
        if unchanged(plan, published):
            if published.sent and payload_identity(marked(plan.payload, service, state)) != published.sent:
                say("! sharing.yaml differs from the published record's metadata, which a later MIME updates "
                    "(`mime push --metadata`): until then, edit it on the record's page")
            if service.production and plan.staged is not None and plan.lock != plan.staged:
                say(f"! the folder's lock does not name the record yet: `mime push <folder> --to {service.name} "
                    f"--publish` checks the record and writes it")
            say(f"{plain(experiment)!r} is unchanged since {published.doi} (version {published.version} on "
                f"{service.name}), so no version is made: {published.page}")
            return Drafted(published.record, published.page, published.doi, published.concept, (), (), False,
                           published.version)
        bundle = replace(plan.files[0], what="the bundle, packed again once the new version's DOI is reserved, its "
                                             "lock naming that record")
        plan = with_sharing(replace(plan, version=published.version + 1, files=(bundle, *plan.files[1:]),
                                    publication=None), plan.sharing)
    choices = Choices(confirm, overwrite_site_edits, keep_site_edits)
    holder = [plan]
    show = _show_plan(lambda: holder[0], state, say, full)
    local = state is not None and state.record is not None and state.doi is not None and \
        {file.key: Uploaded(file.size, file.md5, file.sha256) for file in plan.files} == state.files
    if not local:
        show()
        show = _nothing
    client = Client(service, token_of(service), transport, pause=pause, sleep=sleep)
    return send(plan, state, client, say, choices=choices, show=show, full=full, holder=holder)


def _nothing() -> None:
    pass


def send(plan: Plan, state: DraftState | None, client: Client, say: Say = _quiet, *,
         overwrite_site_edits: bool = False, choices: Choices | None = None, show: Callable[[], None] = _nothing,
         full: bool = False, holder: list[Plan] | None = None) -> Drafted:
    """What ``push`` does once its plan is computed: make or update the draft, reserve its DOI, upload each file that
    differs, and read the draft again to check it holds exactly the plan's files, verified. ``show`` prints the plan
    before the first request that writes. Whatever happens, once there is a draft its link is the last line said."""
    drafting = Drafting(client, plan, state, say, choices or Choices(overwrite=overwrite_site_edits), show, full)
    try:
        state = drafting.draft()
        if holder is not None:
            holder[0] = drafting.plan
        sent, kept_keys = drafting.upload()
        drafting.check()
    except BaseException:
        if drafting.state is not None and drafting.state.page:
            say(drafting.state.page)
        raise
    assert state.record is not None and state.page is not None and state.doi is not None
    service = plan.service.name
    if not drafting.wrote:
        say(f"{plain(plan.experiment)!r} is unchanged: the draft {state.record} on {service} holds it as it is, and "
            f"nothing was sent: {state.page}")
        return Drafted(state.record, state.page, state.doi, state.concept, (), tuple(kept_keys), False, state.version)
    if drafting.made:
        made = f"version {state.version} of {state.published.doi}'s concept" if state.published is not None else \
            "a first version"
        say(f"made the draft {state.record} ({made}) and reserved its DOI {state.doi}")
    linked = [key for key in kept_keys if key in drafting.linked]
    say((f"uploaded {len(sent)} file{'s' if len(sent) != 1 else ''} "
         f"({size_text(sum(file.size for file in drafting.plan.files if file.key in sent))})" if sent else
         "no file sent") + (f", linked {len(linked)} from the version before" if linked else "") +
        f"; the draft holds the {len(drafting.plan.files)} record files, each md5 verified")
    say(f"Nothing is published: look at the draft, then `mime push <folder> --to {service} --publish`.")
    say(state.page)
    return Drafted(state.record, state.page, state.doi, state.concept, tuple(sent), tuple(kept_keys), True,
                   state.version)


def status(folder: str | os.PathLike[str], service: Service, *, say: Say = _quiet, transport: Transport | None = None,
           pause: float | None = None) -> DraftState | None:
    """Say what push knows of the experiment's draft on ``service`` and the draft as the service holds it: its page,
    reserved DOI, concept, title, creators and licences, and each file with its size and md5, against what push
    uploaded and verified, and whether it is the draft push made and what was edited on the site since push last wrote
    it; and the version last published. Changes nothing; ends with a link."""
    usable(service)
    root = Path(os.path.abspath(folder))
    experiment = experiment_name(root)
    state = read_state(service, experiment)
    published = state.published if state is not None else None
    if published is not None:
        say(f"published: version {published.version}, {published.doi} (concept "
            f"{published.concept_doi or published.concept}): {published.page}")
    if state is None or state.record is None:
        if published is None:
            say(f"no draft of {plain(experiment)!r} on {service.name} that MIME knows of (push keeps its record in "
                f"{state_directory(service, experiment)})")
        else:
            say(f"no draft of a new version; a push of a changed folder makes one")
            say(published.page)
        return state
    client = Client(service, token_of(service), transport, pause=pause)
    read = client.read_draft(state.record)
    if read is None:
        if client.read_record(state.record) is not None:
            say(f"the draft {state.record} on {service.name} is published, and push has not recorded it: `mime push "
                f"<folder> --to {service.name} --publish` checks it and records it")
            say(record_page(service, state.record))
        else:
            say(f"the draft {state.record} MIME made is no longer on {service.name} (discarded on the site?); the next "
                f"push makes a new one")
        return state
    draft, errors = read
    held = metadata_of(draft)
    page = page_of(draft, service, state.record)
    say(f"The draft {state.record} on {service.name} (version {state.version}): {page}")
    if not ours(draft, service, state):
        say(f"  it does not carry the identifier push wrote into the draft it made, "
            f"{draft_identifier(service, experiment, state.mark)}: push's record may name another draft, and push "
            f"changes nothing in it")
    elif edited := edited_fields(draft, state):
        say(f"  edited on the site since push last wrote it: {', '.join(plain(each) for each in edited)} (a push "
            f"asks about each in a terminal; {KEEP_FLAG} keeps them, {OVERWRITE_FLAG} writes over them)")
    say(f"  status: {plain(str(draft.get('status', 'draft')))}, published: {draft.get('is_published') is True}; "
        f"reserved DOI: {doi_of(draft) or 'none'}; concept: {parent_id(draft) or 'none'}")
    say(f"  title: {plain(str(held.get('title', '')))}")
    say(f"  creators: {'; '.join(_names(held.get('creators')))}")
    say(f"  licences: {'; '.join(_rights(held.get('rights')))}")
    if draft.get(ACCESS):
        say(f"  access once published: {access_words(cast(Json, draft[ACCESS]))}")
    communities = cast(dict[str, object], draft.get("parent") or {}).get("communities")
    if isinstance(communities, dict) and cast(dict[str, object], communities).get("ids"):
        say(f"  communities: {plain(str(cast(dict[str, object], communities)['ids']))}")
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
    """Delete the experiment's draft on ``service`` (only a draft can be deleted; a new version's draft leaves the
    published versions as they are) and push's record of it: whether a draft was deleted. Only the draft push made is
    deleted (``push.draft-foreign`` otherwise). Asks first. Ends with a link: the draft's when it is kept, the
    account's drafts when it is deleted."""
    usable(service)
    root = Path(os.path.abspath(folder))
    experiment = experiment_name(root)
    state = read_state(service, experiment)
    if state is None or state.record is None:
        say(f"no draft of {plain(experiment)!r} on {service.name} that MIME knows of: nothing to discard")
        return False
    client = Client(service, token_of(service), transport, pause=pause)
    read = client.read_draft(state.record)
    if read is None:
        if client.read_record(state.record) is not None:
            raise published_unrecorded(service, state.record)
        _forget(service, state)
        say(f"the draft {state.record} is no longer on {service.name}; push's record of it is cleared")
        return False
    if read[0].get("is_published") is True:
        raise published_unrecorded(service, state.record)
    if not ours(read[0], service, state):
        raise foreign(service, state)
    page = state.page or service.draft_page(state.record)
    title = plain(str(metadata_of(read[0]).get("title", "")))
    question = (f"Delete the draft {state.record} on {service.name}, {title!r} of {plain(experiment)!r} (version "
                f"{state.version}), and its files ({page})?")
    if not yes:
        if confirm is None:
            raise unconfirmed(f"the deletion of the draft {state.record}")
        if not confirm(question):
            say("Not deleted.")
            say(page)
            return False
    client.delete_draft(state.record)
    _forget(service, state)
    say(f"deleted the draft {state.record} on {service.name}, and push's record of it")
    say(f"{service.site}/me/uploads")
    return True


def _forget(service: Service, state: DraftState) -> None:
    """Push's record of a draft that is gone: cleared, but for the versions published before it."""
    if state.published is None:
        clear_state(service, state.experiment)
        return
    write_state(service, replace(state, record=None, doi=None, page=None, files={}, written={}, sent="", confirmed=[],
                                 creating=False, concept=state.published.concept,
                                 version=state.published.version + 1))


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


__all__ = ["FULL_FLAG", "KEEP_FLAG", "OVERWRITE_FLAG", "UPLOAD_ATTEMPTS", "VERSION_RESETS", "Choices", "Drafted",
           "Drafting", "concept_doi_of", "doi_of", "errors_refusal", "label", "marked", "page_of", "parent_id",
           "discard", "edited_fields", "entry_verified", "experiment_name", "foreign", "init", "metadata_of", "ours",
           "planned", "publication", "published_publication", "published_unrecorded", "push", "record_page", "send",
           "site_edits", "status", "token_of", "unchanged", "unconfirmed", "usable"]
