"""Drafts against the offline fake: the plan shown before any request, a draft made, its DOI reserved and its files
uploaded and verified; pushed again, nothing sent again; ``--status`` and ``--discard``; and resumed, at the level of
whole files, from a push killed at any request.

Controls, each caught on every run:

- **the plan skipped**: push's first request comes after the plan's last line and the confirmation, and a push whose
  plan is planted away is caught sending before anything was shown;
- **a draft for the sandbox reused for production**: a production push after a sandbox push makes its own draft and
  never names the sandbox's; a record whose key planted away the service is caught taking the sandbox's draft;
- **a file that may not be redistributed reaching the bucket**: the fake's stored bytes (and every member of a stored
  zip) are held to the sha256 of each such file, for the folder whose mesh may not be redistributed and over drawn
  lock assets of every verdict; a planted choice that uploads every asset is caught;
- **a lost md5 check in the resumed push**: with the comparison planted away, a corrupt upload ends the property's
  run with a draft that does not hold the planned bytes;
- **a record believed whatever draft it names**: with push's identifier check planted away, a record naming another
  experiment's draft is caught rewriting that draft's title, and ``--discard`` deleting it;
- **an edit made on the site written over**: with the check of what push last wrote planted away, a title edited on
  the site is caught overwritten without a word;
- **a test writing outside its own state directory**: the conftest's guard finds a record planted in the module's.

Every refusal is captured with its traceback's locals, and none holds the token, in any form ``Token.occurs_in``
finds; nor does any line push said, any file it wrote, or any request it sent but the header.
"""

from __future__ import annotations

import json
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from homes import Outside, isolate
from push_cases import client, copy_of, plan_of, run_until_drafted, saved_toy, with_token, write_sharing
from terms_cases import CONDITIONAL_IDS, NO_REDISTRIBUTION, OPEN_IDS, UNKNOWN_IDS, layered, same_reference_run, \
    write_mesh
from zenodo_records import FakeRecords, FakeTransport, Killed

from mime import push
from mime.assets import AssetRef
from mime.build import BuildRefused
from mime.credentials import Token
from mime.push import _draft, _plan, _state
from mime.saved._terms import FolderTerms
from mime.terms import Evidence, Permission, Scope, Terms

ORIGIN = "https://example.org/ar4_ros_driver/6a3ebb1/meshes/ar4_visual.usda"


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = saved_toy(tmp_path_factory.mktemp("toy"))
    write_sharing(folder)
    return folder


@pytest.fixture(scope="module")
def ar4(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The 7n case: a mesh whose licence forbids redistribution (synthetic), an MIT layer, the folder CC-BY-4.0."""
    root = tmp_path_factory.mktemp("ar4")
    visual = write_mesh(root / "made", "ar4_visual", 6000)
    kinematics = write_mesh(root / "made", "ar4_kinematics", 1200)
    with same_reference_run():
        layered("ar4_like", [(visual, Terms(NO_REDISTRIBUTION, "Robot Maker Ltd", "presentation", (ORIGIN,))),
                             (kinematics, Terms("MIT", "Copyright (c) 2021 A. Driver", "results"))],
                licence="CC-BY-4.0").build().save(root / "ar4_like")
    write_sharing(root / "ar4_like")
    return root / "ar4_like"


@pytest.fixture
def token(monkeypatch: pytest.MonkeyPatch) -> str:
    return with_token(monkeypatch)


class Seen:
    """What push said and asked, and each request the fake got, in one order."""

    def __init__(self, fake: FakeRecords) -> None:
        self.events: list[str] = []
        self.fake = fake
        real = fake._handle

        def handled(*given: object) -> object:
            self.events.append("request")
            return real(*given)  # type: ignore[arg-type]

        fake._handle = handled  # type: ignore[method-assign]

    def say(self, line: str) -> None:
        self.events.append(f"said {line}")

    def ask(self, question: str) -> bool:
        self.events.append(f"asked {question}")
        return True


def pushed(folder: Path, fake: FakeRecords, *, service: push.Service | None = None, seen: Seen | None = None,
           **given: object) -> push.Drafted | None:
    held = seen or Seen(fake)
    return push.push(folder, service or fake.service(), confirm=held.ask, say=held.say,
                     transport=FakeTransport(fake), pause=0, sleep=lambda seconds: None, **given)  # type: ignore[arg-type]


def codes(action: Callable[[], object]) -> list[str]:
    with pytest.raises(BuildRefused) as refused:
        action()
    return refused.value.codes()


# --- the plan, then the draft ----------------------------------------------------------------------------------------


def plan_comes_first(events: list[str]) -> list[str]:
    """What is wrong with the order of a push's events: a request before the plan's last line, or before the
    confirmation."""
    first = next((index for index, event in enumerate(events) if event == "request"), len(events))
    shown = [index for index, event in enumerate(events) if event.startswith("said Next: ")]
    asked = [index for index, event in enumerate(events) if event.startswith("asked ")]
    problems = []
    if not shown or shown[0] > first:
        problems.append("a request before the plan was shown")
    if not asked or asked[0] > first:
        problems.append("a request before the plan was confirmed")
    return problems


def test_the_plan_is_shown_and_confirmed_before_the_first_request(toy: Path, token: str, tmp_path: Path):
    fake = FakeRecords(token)
    seen = Seen(fake)
    drafted = pushed(copy_of(toy, tmp_path), fake, seen=seen)
    assert drafted is not None and plan_comes_first(seen.events) == []
    said = "\n".join(event for event in seen.events if event.startswith("said "))
    for line in ("Record files (2", "full_toy.mimebundle", "NOTICE", "Inside the bundle (4 files)",
                 "experiment.mime.yaml", "yours; permitted, open", "Referenced, not uploaded (0): none",
                 "The experiment's licence: CC-BY-4.0", "Creators: Carberry, Josiah",
                 "Related: Requires and IsCompiledBy mime-engine", "nothing is published", "Title: full_toy",
                 "Description: the card, as HTML (", "Publisher: Zenodo", "Publication date: 20",
                 "Resource type: dataset", "Access once published: the record public, its files public",
                 "Alternate identifier: a urn:uuid push makes for the draft"):
        assert line in said, line


def test_the_plan_names_every_field_of_the_metadata_push_sends(toy: Path, token: str, tmp_path: Path):
    """Each field of the metadata push sends is named in the plan (the alternate identifier, made with the draft, by
    what it is)."""
    fake = FakeRecords(token)
    seen = Seen(fake)
    drafted = pushed(copy_of(toy, tmp_path), fake, seen=seen)
    assert drafted is not None
    said = "\n".join(event.removeprefix("said ") for event in seen.events if event.startswith("said "))
    named = {"title": "Title:", "description": "Description:", "publisher": "Publisher:",
             "publication_date": "Publication date:", "resource_type": "Resource type:", "creators": "Creators:",
             "rights": "Licences on the record:", "subjects": "Keywords:", "related_identifiers": "Related:",
             "identifiers": "Alternate identifier:", "contributors": "Contributors:"}
    sent = fake.records[drafted.record].body
    assert set(sent["metadata"]) <= set(named), set(sent["metadata"]) - set(named)  # type: ignore[arg-type]
    assert all(named[field] in said for field in sent["metadata"])  # type: ignore[union-attr]
    assert "Access once published:" in said and set(sent) <= {"metadata", "access", "files", "pids"}


def test_control_a_push_whose_plan_is_planted_away_is_caught(toy: Path, token: str, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_plan.Plan, "lines", lambda self, state: [])
    fake = FakeRecords(token)
    seen = Seen(fake)
    pushed(copy_of(toy, tmp_path), fake, seen=seen)
    assert "a request before the plan was shown" in plan_comes_first(seen.events)


def test_without_a_terminal_or_yes_nothing_is_sent(toy: Path, token: str, tmp_path: Path):
    fake = FakeRecords(token)
    folder = copy_of(toy, tmp_path)
    assert codes(lambda: push.push(folder, fake.service(), transport=FakeTransport(fake), pause=0)) == \
        ["push.unconfirmed"]
    declined = push.push(folder, fake.service(), confirm=lambda question: False, transport=FakeTransport(fake),
                         pause=0)
    assert declined is None and fake.requests == []


def test_a_push_makes_the_draft_reserves_its_doi_and_uploads_each_file_verified(toy: Path, token: str,
                                                                               tmp_path: Path):
    fake = FakeRecords(token)
    folder = copy_of(toy, tmp_path)
    drafted = pushed(folder, fake)
    assert drafted is not None and drafted.doi == "10.5072/zenodo.629525" and drafted.concept == "629524"
    assert drafted.page == "https://zenodo-records.invalid/uploads/629525"
    record = fake.records[drafted.record]
    assert set(record.files) == {"full_toy.mimebundle", "NOTICE"}
    staged = _state.state_directory(fake.service(), "full_toy") / _state.STAGED
    for key, entry in record.files.items():
        assert entry.data == (staged / key).read_bytes() and entry.status == "completed"
    metadata = record.body["metadata"]
    assert metadata["rights"] == [{"id": "cc-by-4.0"}] and metadata["resource_type"] == {"id": "dataset"}  # type: ignore[index]
    assert metadata["description"].startswith("<h1>full_toy</h1>")  # type: ignore[index]
    held = push.read_state(fake.service(), "full_toy")
    assert held is not None and held.files["NOTICE"].md5 == record.files["NOTICE"].checksum.removeprefix("md5:")  # type: ignore[union-attr]


def test_pushed_again_nothing_is_sent_again_and_the_metadata_is_written_again(toy: Path, token: str,
                                                                              tmp_path: Path):
    fake = FakeRecords(token)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    before = len(fake.requests)
    again = pushed(folder, fake)
    assert again is not None and again.uploaded == () and set(again.kept) == {"full_toy.mimebundle", "NOTICE"}
    later = [(method, path) for method, path, _, _ in fake.requests[before:]]
    assert ("PUT", "/records/629525/draft") in later and not any(path.endswith("/content") for _, path in later)
    assert len(fake.drafts()) == 1


def test_a_changed_folder_sends_only_the_files_that_differ(toy: Path, token: str, tmp_path: Path):
    fake = FakeRecords(token)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    sharing = folder / "sharing.yaml"
    sharing.write_text(sharing.read_text(encoding="utf-8").replace('title: "full_toy"', 'title: "renamed"'),
                       encoding="utf-8")
    again = pushed(folder, fake)
    assert again is not None and again.uploaded == () and \
        fake.records[again.record].body["metadata"]["title"] == "renamed"  # type: ignore[index]


def test_status_reads_the_draft_as_the_service_holds_it(toy: Path, token: str, tmp_path: Path):
    fake = FakeRecords(token)
    folder = copy_of(toy, tmp_path)
    said: list[str] = []
    push.status(folder, fake.service(), say=said.append, transport=FakeTransport(fake), pause=0)
    assert said and said[0].startswith("no draft of 'full_toy' on zenodo-sandbox that MIME knows of")
    pushed(folder, fake)
    fake.records["629525"].files["extra.txt"] = type(fake.records["629525"].files["NOTICE"])("extra.txt")
    said.clear()
    push.status(folder, fake.service(), say=said.append, transport=FakeTransport(fake), pause=0)
    text = "\n".join(said)
    assert "reserved DOI: 10.5072/zenodo.629525; concept: 629524" in text and "creators: Carberry" in text
    assert "NOTICE" in text and "as push uploaded and verified it" in text and "extra.txt" in text and \
        "not complete" in text


def test_discard_deletes_the_draft_and_the_record_and_only_a_draft(toy: Path, token: str, tmp_path: Path):
    fake = FakeRecords(token)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    assert codes(lambda: push.discard(folder, fake.service(), transport=FakeTransport(fake), pause=0)) == \
        ["push.unconfirmed"]
    assert push.discard(folder, fake.service(), yes=True, transport=FakeTransport(fake), pause=0) is True
    assert fake.drafts() == [] and push.read_state(fake.service(), "full_toy") is None
    assert push.discard(folder, fake.service(), yes=True, transport=FakeTransport(fake), pause=0) is False
    pushed(folder, fake)
    fake.publish("629527")
    assert codes(lambda: push.discard(folder, fake.service(), yes=True, transport=FakeTransport(fake),
                                      pause=0)) == ["push.published"]
    assert codes(lambda: pushed(folder, fake)) == ["push.published"]


def test_a_draft_discarded_on_the_site_is_made_again(toy: Path, token: str, tmp_path: Path):
    fake = FakeRecords(token)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    del fake.records["629525"]
    again = pushed(folder, fake)
    assert again is not None and again.record == "629527" and len(fake.drafts()) == 1


def test_a_file_added_on_the_site_is_refused_and_one_push_uploaded_earlier_is_deleted(toy: Path, token: str,
                                                                                      tmp_path: Path):
    fake = FakeRecords(token)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    record = fake.records["629525"]
    record.files["notes.txt"] = type(record.files["NOTICE"])("notes.txt", b"mine", "completed", "md5:x")
    assert codes(lambda: pushed(folder, fake)) == ["push.draft-changed"]
    held = push.read_state(fake.service(), "full_toy")
    assert held is not None
    held.files["notes.txt"] = _state.Uploaded(4, "x", "y")
    _state.write_state(fake.service(), held)
    pushed(folder, fake)
    assert "notes.txt" not in record.files


# --- two services --------------------------------------------------------------------------------------------------


def test_a_production_push_after_a_sandbox_push_makes_its_own_draft(toy: Path, monkeypatch: pytest.MonkeyPatch,
                                                                    tmp_path: Path):
    sandbox_token, production_token = with_token(monkeypatch, "zenodo-sandbox"), with_token(monkeypatch, "zenodo")
    sandbox, production = FakeRecords(sandbox_token), FakeRecords(production_token, prefix="10.5281", first=900001)
    folder = copy_of(toy, tmp_path)
    first = pushed(folder, sandbox, service=sandbox.service())
    second = pushed(folder, production, service=production.service("zenodo", production=True))
    assert first is not None and second is not None and second.record == "900001"
    assert not any(first.record in path for _, path, _, _ in production.requests)
    assert second.doi.startswith("10.5281/")


def test_control_a_record_keyed_without_its_service_is_caught_taking_the_sandboxs_draft(
        toy: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sandbox_token, production_token = with_token(monkeypatch, "zenodo-sandbox"), with_token(monkeypatch, "zenodo")
    monkeypatch.setattr(_state, "state_directory", lambda service, experiment: _state.state_root() / experiment)
    monkeypatch.setattr(_state, "read_state", lambda service, experiment: _read_any(experiment))
    monkeypatch.setattr(_draft, "read_state", lambda service, experiment: _read_any(experiment))
    sandbox, production = FakeRecords(sandbox_token), FakeRecords(production_token, prefix="10.5281", first=629525)
    folder = copy_of(toy, tmp_path)
    pushed(folder, sandbox, service=sandbox.service())
    try:
        pushed(folder, production, service=production.service("zenodo", production=True))
    except BuildRefused:
        pass
    assert ("GET", "/records/629525/draft") in [(m, p) for m, p, _, _ in production.requests], \
        "the planted record keyed by name alone takes the sandbox's draft id to production"


def _read_any(experiment: str) -> _state.DraftState | None:
    path = _state.state_root() / experiment / _state.STATE_FILE
    if not path.is_file():
        return None
    held = json.loads(path.read_text(encoding="utf-8"))
    return _state.DraftState(held["service"], held["api"], held["experiment"], held["record"], held["concept"],
                             held["doi"], held["page"], held["creating"])


def test_a_record_of_another_service_is_refused(toy: Path, token: str, tmp_path: Path):
    fake = FakeRecords(token)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    path = _state.state_directory(fake.service(), "full_toy") / _state.STATE_FILE
    held = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**held, "service": "zenodo", "api": "https://zenodo.org/api"}), encoding="utf-8")
    assert codes(lambda: pushed(folder, fake)) == ["push.state"]
    path.write_text("not json", encoding="utf-8")
    assert codes(lambda: pushed(folder, fake)) == ["push.state"]


def test_a_production_service_that_reserves_a_test_prefix_doi_is_refused(toy: Path,
                                                                         monkeypatch: pytest.MonkeyPatch,
                                                                         tmp_path: Path):
    fake = FakeRecords(with_token(monkeypatch, "zenodo"))
    assert codes(lambda: pushed(copy_of(toy, tmp_path), fake, service=fake.service("zenodo", production=True))) \
        == ["push.service"]


# --- tiers: what reaches the bucket --------------------------------------------------------------------------------


def test_a_file_that_may_not_be_redistributed_never_reaches_the_bucket(ar4: Path, token: str, tmp_path: Path):
    fake = FakeRecords(token)
    seen = Seen(fake)
    pushed(copy_of(ar4, tmp_path), fake, seen=seen)
    from mime import assets

    forbidden = {ref.sha256 for ref in assets.folder_assets(ar4)}
    assert len(forbidden) == 1 and not forbidden & fake.stored_sha256s()
    assert len(fake.stored()) > 2, "the stored bundle's members are among the bytes held to it"
    said = "\n".join(seen.events)
    assert "Referenced, not uploaded (1):" in said and "scene/layers/ar4_visual.usda" in said
    rights = fake.records["629525"].body["metadata"]["rights"]  # type: ignore[index]
    assert rights == [{"id": "cc-by-4.0"}, {"id": "mit"}], "the record lists every licence of a file it holds"
    contributors = fake.records["629525"].body["metadata"]["contributors"]  # type: ignore[index]
    names = [each["person_or_org"]["name"] for each in contributors]  # type: ignore[index]
    assert names == ["Copyright (c) 2021 A. Driver"], \
        "the rights holder of the file the record holds is a contributor; the referenced file's is not"


LICENCES = st.sampled_from([*OPEN_IDS, *CONDITIONAL_IDS, *UNKNOWN_IDS[:40], "NOASSERTION"])


def oracle_carried(licence: str) -> bool:
    """This test's own reading of #198's tiers: open and conditional travel; anything else is referenced."""
    return licence in OPEN_IDS or licence in CONDITIONAL_IDS


@given(st.lists(st.tuples(LICENCES, st.booleans(), st.booleans()), min_size=1, max_size=8))
def test_no_asset_whose_licence_forbids_it_is_ever_chosen_for_upload(drawn: list[tuple[str, bool, bool]]):
    refs, judged_files = [], []
    for index, (licence, zenodo, _) in enumerate(drawn):
        location = f"doi:10.5281/zenodo.{index + 1}/a{index}.npz" if zenodo else f"https://example.org/a{index}.npz"
        refs.append(AssetRef(f"a{index}", f"{index:064x}", 10, licence, (location,), unlisted=True))
    choices = _plan.asset_choices(refs, FolderTerms(tuple(judged_files), "CC-BY-4.0", {}, (), 4), {})  # type: ignore[arg-type]
    for choice, (licence, zenodo, _) in zip(choices, drawn, strict=True):
        assert choice.upload == (oracle_carried(licence) and not zenodo), (licence, zenodo, choice)


def test_control_a_choice_that_uploads_every_asset_is_caught_every_run(ar4: Path, token: str, tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch):
    real = _plan.asset_choices
    monkeypatch.setattr(_plan, "asset_choices", lambda refs, judged, lock: [
        _plan.AssetChoice(each.ref, each.verdict, True, "planted") for each in real(refs, judged, lock)])
    import shutil

    from mime import assets
    from terms_cases import mesh_bytes

    folder = copy_of(ar4, tmp_path)
    ref = assets.folder_assets(folder)[0]
    cached = assets.file_path(ref)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(mesh_bytes("ar4_visual", 6000))
    monkeypatch.setattr(_plan, "locate", lambda held, folder=None: cached)
    fake = FakeRecords(token)
    pushed(folder, fake)
    assert ref.sha256 in fake.stored_sha256s(), "the planted choice's upload of the forbidden mesh is not seen"
    shutil.rmtree(cached.parent, ignore_errors=True)


# --- resumed from a push killed at any request ---------------------------------------------------------------------


FILES = st.dictionaries(st.sampled_from(["a.mimebundle", "NOTICE", "t.npz", "u.npz"]),
                        st.binary(min_size=1, max_size=300), min_size=1, max_size=4)


def resumed_problems(files: dict[str, bytes], kill_at: int, after: bool, verify: bool = True) -> list[str]:
    """The resume property's failures for one draw: a push killed at request ``kill_at`` (before or after the service
    acts), then pushed again until done. The end: one draft push knows, holding exactly the planned files with their
    bytes; a file verified before the stop not sent again; a second draft only where the stop lost the answer to the
    draft's making, and then push said so."""
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
        root = Path(directory)
        isolate(patch, root / "home")
        if not verify:
            patch.setattr(_draft, "_entry_verified", lambda entry, file: entry.get("status") == "completed")
        token = with_token(patch)
        fake = FakeRecords(token)
        count = [0]

        def at(method: str, path: str) -> bool:
            count[0] += 1
            return count[0] == kill_at + 1

        fake.plant("kill-after" if after else "kill-before", at)
        fake.plant("corrupt", lambda method, path: method == "PUT" and path.endswith("/content"))
        plan = plan_of(fake, files, root / "files")
        said: list[str] = []
        try:
            push.send(plan, None, client(fake), said.append)
        except (Killed, BuildRefused):
            pass
        stopped = len(fake.requests)
        recorded = push.read_state(plan.service, plan.experiment)
        verified_before = set(recorded.files) if recorded is not None else set()
        drafted, _ = run_until_drafted(plan, fake, said)
        record = fake.records[drafted.record]
        if {key: entry.data for key, entry in record.files.items()} != files:
            problems.append(f"the draft holds {sorted(record.files)}, not exactly the planned bytes")
        orphan_possible = after and kill_at < stopped and fake.requests[kill_at][:2] == ("POST", "/records")
        if len(fake.drafts()) != 1 and not (orphan_possible and len(fake.drafts()) == 2 and
                                            any(line.startswith("warning: an earlier push stopped") for line in said)):
            problems.append(f"{len(fake.drafts())} drafts")
        resent = [path for method, path, _, _ in fake.requests[stopped:] if method == "PUT" and
                  path.endswith("/content") and path.split("/")[-2] in verified_before]
        if resent:
            problems.append(f"sent again after it was verified: {resent}")
    return problems


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(FILES, st.integers(0, 30), st.booleans())
def test_a_push_killed_at_any_request_is_resumed_by_running_it_again(files: dict[str, bytes], kill_at: int,
                                                                     after: bool):
    assert resumed_problems(files, kill_at, after) == []


def test_control_without_the_md5_check_the_resume_property_fails_every_run():
    for kill_at in (0, 3, 7, 30):
        problems = resumed_problems({"a.mimebundle": b"a" * 50, "NOTICE": b"n" * 20}, kill_at, False, verify=False)
        assert any("not exactly the planned bytes" in each for each in problems), (kill_at, problems)


def test_a_dropped_answer_to_the_drafts_making_is_said_and_one_draft_is_kept_after(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    plan = plan_of(fake, {"a.bin": b"a"}, tmp_path / "files")
    fake.plant("kill-after", lambda method, path: path == "/records")
    said: list[str] = []
    drafted, stops = run_until_drafted(plan, fake, said)
    assert stops == 1 and any("may hold a draft MIME has no record of" in line for line in said)
    assert len(fake.drafts()) == 2 and drafted.record == "629527"


# --- refusals before anything goes online --------------------------------------------------------------------------


def test_a_folder_without_a_licence_and_a_disagreeing_licence_are_refused_before_a_request(token: str,
                                                                                          tmp_path: Path):
    fake = FakeRecords(token)
    bare = saved_toy(tmp_path / "bare", licence=None)
    write_sharing(bare)
    with pytest.raises(BuildRefused) as refused:
        pushed(bare, fake)
    assert refused.value.codes() == ["push.licence"] and 'exp.licence(\'CC-BY-4.0\')' in \
        refused.value.refusals[0].action
    other = saved_toy(tmp_path / "mit", licence="MIT")
    path = write_sharing(other)
    path.write_text(path.read_text(encoding="utf-8").replace("licence: MIT", "licence: CC-BY-4.0"), encoding="utf-8")
    with pytest.raises(BuildRefused) as refused:
        pushed(other, fake)
    assert refused.value.codes() == ["push.licence"] and "and the experiment declares MIT" in         refused.value.refusals[0].message
    assert fake.requests == []


def test_no_token_is_refused_after_the_plan_and_before_a_request(toy: Path, tmp_path: Path):
    fake = FakeRecords("unused")
    seen = Seen(fake)
    with pytest.raises(BuildRefused) as refused:
        pushed(copy_of(toy, tmp_path), fake, seen=seen)
    assert refused.value.codes() == ["push.token"] and "mime credentials set zenodo-sandbox" in \
        refused.value.refusals[0].action
    assert "request" not in seen.events and any(event.startswith("said Next: ") for event in seen.events)


def test_too_many_files_and_too_large_are_refused_in_the_plan(toy: Path, token: str, tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_plan, "MOST_FILES", 1)
    monkeypatch.setattr(_plan, "MOST_BYTES", 100)
    fake = FakeRecords(token)
    assert sorted(codes(lambda: pushed(copy_of(toy, tmp_path), fake))) == ["push.too-large", "push.too-many-files"]
    assert fake.requests == []


def test_a_sandbox_location_in_a_lock_pushed_to_production_is_refused(toy: Path, monkeypatch: pytest.MonkeyPatch,
                                                                      tmp_path: Path):
    fake = FakeRecords(with_token(monkeypatch, "zenodo"))
    folder = copy_of(toy, tmp_path)
    ref = AssetRef("table", "ab" * 32, 10, "CC0-1.0", ("doi:10.5072/zenodo.42/table.npz",))
    monkeypatch.setattr(_plan, "folder_assets", lambda root: (ref,))
    assert codes(lambda: pushed(folder, fake, service=fake.service("zenodo", production=True))) == \
        ["push.sandbox-location"]


def test_a_permission_whose_grantee_is_not_the_uploader_is_refused(token: str, tmp_path: Path):
    shell = write_mesh(tmp_path / "made", "shell", 900)
    from terms_cases import sha256_of

    from mime.terms import CustomLicence

    own = CustomLicence("LicenseRef-Maker-1.0", "Maker Licence", "https://example.org/maker")
    permission = Permission("Maker Ltd", "Carberry, Josiah", (sha256_of(shell),), Scope(True, True, True),
                            "2026-11-01", Evidence("public_statement", "https://example.org/permission"))
    with same_reference_run():
        layered("granted", [(shell, Terms(own, "Maker Ltd", "presentation"))], licence="CC-BY-4.0",
                permissions=[permission]).build().save(tmp_path / "granted")
    folder = tmp_path / "granted"
    write_sharing(folder)
    fake = FakeRecords(token)
    assert codes(lambda: pushed(folder, fake)) == ["push.permission"]
    sharing = folder / "sharing.yaml"
    sharing.write_text(sharing.read_text(encoding="utf-8") + 'uploader: {name: "Somebody, Else"}\n',
                       encoding="utf-8")
    assert codes(lambda: pushed(folder, fake)) == ["push.permission"]
    sharing.write_text(sharing.read_text(encoding="utf-8").replace("Somebody, Else", "Carberry, Josiah"),
                       encoding="utf-8")
    seen = Seen(fake)
    pushed(folder, fake, seen=seen)
    said = "\n".join(seen.events)
    assert "You are declaring publicly, under your name, that Maker Ltd permitted you to redistribute this file" in said
    rights = fake.records["629525"].body["metadata"]["rights"]  # type: ignore[index]
    assert {"id": "cc-by-4.0"} in rights and any(each.get("link") == "https://example.org/maker" and "id" not in each
                                                 for each in rights)


# --- the token -----------------------------------------------------------------------------------------------------


def captured_with_locals(error: BaseException) -> str:
    return "".join(traceback.TracebackException.from_exception(error, capture_locals=True).format())


def every_refusal(folder: Path, fake: FakeRecords, tmp_path: Path) -> list[BaseException]:
    """A refusal from each path push takes with a token in hand: the service's answers, a transfer that fails every
    time, the state, a draft changed on the site, a published record, no terminal."""
    found: list[BaseException] = []
    plantings: list[tuple[str, Callable[[str, str], bool], dict[str, object]]] = [
        ("status", lambda m, p: p == "/records", {"status": 401}),
        ("status", lambda m, p: p == "/records", {"status": 403}),
        ("status", lambda m, p: p == "/records", {"status": 400}),
        ("status", lambda m, p: p == "/records", {"status": 500, "times": 9}),
        ("status", lambda m, p: p == "/records", {"status": 429, "times": 9, "headers": {"Retry-After": "1"}}),
        ("drop-before", lambda m, p: p == "/records", {"times": 9}),
        ("corrupt", lambda m, p: p.endswith("/content"), {"times": 99}),
        ("cut", lambda m, p: p.endswith("/content"), {"times": 99}),
    ]
    for kind, matches, given in plantings:
        fake.faults.clear()
        fake.plant(kind, matches, **given)
        try:
            pushed(folder, fake)
        except BuildRefused as error:
            found.append(error)
        push.discard(folder, fake.service(), yes=True, transport=FakeTransport(fake), pause=0)
    fake.faults.clear()
    for action in (lambda: push.push(folder, fake.service(), transport=FakeTransport(fake), pause=0),
                   lambda: push.discard(folder, fake.service(), transport=FakeTransport(fake), pause=0)):
        try:
            pushed(folder, fake)
            action()
        except BuildRefused as error:
            found.append(error)
    fake.publish(max(fake.records))
    for action in (lambda: pushed(folder, fake),
                   lambda: push.discard(folder, fake.service(), yes=True, transport=FakeTransport(fake), pause=0)):
        try:
            action()
        except BuildRefused as error:
            found.append(error)
    return found


def test_no_refusal_line_file_or_request_holds_the_token(toy: Path, token: str, tmp_path: Path,
                                                         capsys: pytest.CaptureFixture[str]):
    folder = copy_of(toy, tmp_path)
    fake = FakeRecords(token)
    refusals = every_refusal(folder, fake, tmp_path)
    found = sorted({code for error in refusals for code in error.codes()})  # type: ignore[attr-defined]
    assert {"push.token", "push.token-scope", "push.rejected", "push.service", "push.checksum", "push.unconfirmed",
            "push.published"} <= set(found), found
    held = Token("zenodo-sandbox", token)
    for error in refusals:
        assert not held.occurs_in(captured_with_locals(error).encode("utf-8")), error
    written = [path for path in (Path(_state.state_root())).rglob("*") if path.is_file()]
    assert written and not any(held.occurs_in(path.read_bytes()) for path in written)
    assert not held.occurs_in(json.dumps(fake.requests).encode("utf-8"))
    assert not held.occurs_in(capsys.readouterr().out.encode("utf-8"))


def test_control_the_capture_finds_a_token_a_refusal_holds(token: str):
    held = Token("zenodo-sandbox", token)

    def leaky() -> None:
        text = held.reveal()  # noqa: F841 - the planted defect: the text under a name in a frame
        raise BuildRefused([push.Client.__init__.__globals__["_refuse"](
            "push.service", push._client.RefusalClass.ENVIRONMENT, "planted", "`mime push <folder> --to x`")])

    with pytest.raises(BuildRefused) as refused:
        leaky()
    assert held.occurs_in(captured_with_locals(refused.value).encode("utf-8"))


# --- who owns a refused field; a stand-in never touches a real service's records ----------------------------------


def test_the_draft_push_sends_holds_every_field_datacite_needs_for_its_doi(toy: Path, token: str, tmp_path: Path):
    from mime.push._plan import DOI_FIELDS

    fake = FakeRecords(token)
    plan = _plan.make_plan(copy_of(toy, tmp_path), fake.service(), push.read_sharing(toy), tmp_path / "staged")
    metadata = plan.payload["metadata"]
    assert all(metadata.get(field) for field in DOI_FIELDS)  # type: ignore[union-attr]
    assert metadata["publisher"] == "Zenodo"  # type: ignore[index]


def without(plan: push.Plan, field: str) -> push.Plan:
    from dataclasses import replace

    metadata = dict(plan.payload["metadata"])  # type: ignore[arg-type]
    metadata.pop(field)
    return replace(plan, payload={**plan.payload, "metadata": metadata})


def test_a_refused_field_mime_builds_is_mimes_defect_and_one_the_author_writes_is_theirs(token: str,
                                                                                         tmp_path: Path):
    from mime.build import RefusalClass

    fake = FakeRecords(token)
    with pytest.raises(BuildRefused) as refused:
        push.send(without(plan_of(fake, {"a.bin": b"a"}, tmp_path / "a"), "publisher"), None, client(fake))
    held = refused.value.refusals
    assert [(r.code, r.category) for r in held] == [("push.rejected", RefusalClass.INTERNAL)]
    assert "metadata.publisher" in held[0].message and "MIME's defect, not yours" in held[0].message
    assert "sharing.yaml" not in held[0].action
    fake = FakeRecords(token)
    with pytest.raises(BuildRefused) as refused:
        push.send(without(plan_of(fake, {"a.bin": b"a"}, tmp_path / "b"), "title"), None, client(fake))
    held = refused.value.refusals
    assert [(r.code, r.category) for r in held] == [("push.rejected", RefusalClass.USER)]
    assert "correct it in sharing.yaml" in held[0].action


def test_each_field_of_the_metadata_has_its_owner():
    owner = _plan.field_owner
    assert [owner(field) for field in ("metadata.publisher", "metadata.publication_date", "metadata.resource_type",
                                       "metadata.description", "metadata.related_identifiers.1.identifier",
                                       "metadata.something_new")] == ["mime"] * 6
    assert [owner(field, 3) for field in ("metadata.title", "metadata.creators.0.person_or_org.family_name",
                                          "metadata.contributors.2.role", "metadata.related_identifiers.2",
                                          "metadata.subjects.1")] == ["sharing"] * 5
    assert [owner(field, 3) for field in ("metadata.contributors.3.role", "metadata.subjects.0.subject",
                                          "metadata.identifiers.0.identifier")] == ["mime"] * 3
    assert owner("metadata.contributors.0.role") == "mime", "with no contributor in sharing.yaml, every one is MIME's"
    assert owner("metadata.rights.0.id") == "experiment"


def test_a_fake_cannot_take_a_real_services_address():
    for api in ("https://zenodo.org/api", "https://sandbox.zenodo.org/api"):
        with pytest.raises(ValueError, match="a fake at a real service's address"):
            FakeRecords("t", api=api)


def test_a_stand_ins_records_are_never_read_or_written_for_the_real_service(toy: Path, token: str,
                                                                            tmp_path: Path):
    """A dry run's fake, named as the sandbox and even at its address, keeps its records apart: the real sandbox's
    push (here a stand-in at the real address, in this process) makes its own draft and never names the fake's."""
    folder = copy_of(toy, tmp_path)
    dry = FakeRecords(token, api=push.ZENODO_SANDBOX.api, site=push.ZENODO_SANDBOX.site, real_address=True)
    dry_service = push.Service("zenodo-sandbox", push.ZENODO_SANDBOX.api, push.ZENODO_SANDBOX.site, "zenodo-sandbox",
                               False, "a dry run of the sandbox")
    pushed(folder, dry, service=dry_service)
    assert _state.namespace(dry_service) != _state.namespace(push.ZENODO_SANDBOX)
    assert not (_state.state_root() / "zenodo-sandbox").exists()
    real = FakeRecords(token, api=push.ZENODO_SANDBOX.api, site=push.ZENODO_SANDBOX.site, first=700001,
                       real_address=True)
    drafted = pushed(folder, real, service=push.ZENODO_SANDBOX)
    assert drafted is not None and drafted.record == "700001"
    assert not any("629525" in path for _, path, _, _ in real.requests)


def test_control_records_kept_by_the_services_name_alone_let_a_dry_run_steer_the_real_push(
        toy: Path, token: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_state, "namespace", lambda service: service.name)
    folder = copy_of(toy, tmp_path)
    dry = FakeRecords(token, api=push.ZENODO_SANDBOX.api, site=push.ZENODO_SANDBOX.site, real_address=True)
    pushed(folder, dry, service=push.Service("zenodo-sandbox", push.ZENODO_SANDBOX.api, push.ZENODO_SANDBOX.site,
                                             "zenodo-sandbox", False, "a dry run of the sandbox"))
    real = FakeRecords(token, api=push.ZENODO_SANDBOX.api, site=push.ZENODO_SANDBOX.site, first=700001,
                       real_address=True)
    pushed(folder, real, service=push.ZENODO_SANDBOX)
    assert ("GET", "/records/629525/draft") in [(m, p) for m, p, _, _ in real.requests], \
        "with the namespace planted away, the real push takes the dry run's draft id"


def lane() -> object:
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "zenodo_sandbox_lane.py"
    spec = importlib.util.spec_from_file_location("zenodo_sandbox_lane", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def own_temporary_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    import tempfile as temporary

    folder = tmp_path / "tmp"
    folder.mkdir()
    monkeypatch.setattr(temporary, "tempdir", str(folder))
    return folder


def test_the_lanes_dry_run_never_writes_a_record_of_the_real_sandbox(token: str, tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch):
    own_temporary_folder(monkeypatch, tmp_path)
    held = lane()
    with held.sandbox_side(True, Token("zenodo-sandbox", token)) as (service, _):  # type: ignore[attr-defined]
        assert service != push.ZENODO_SANDBOX and _state.namespace(service).startswith("other/")
    assert held.main(["--creator", "Carberry, Josiah", "--demo", "toy", "--dry-run",  # type: ignore[attr-defined]
                      "--log", str(tmp_path / "log")]) == 0
    assert not (_state.state_root() / "zenodo-sandbox").exists()
    assert list((tmp_path / "log").glob("zenodo_sandbox_lane_*.jsonl"))


def test_control_a_dry_run_as_the_real_sandbox_is_caught_writing_its_records(token: str, tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch):
    import contextlib

    own_temporary_folder(monkeypatch, tmp_path)
    held = lane()
    monkeypatch.setattr(held, "sandbox_side", lambda dry_run, token: contextlib.nullcontext((
        push.ZENODO_SANDBOX, FakeTransport(FakeRecords(token, api=push.ZENODO_SANDBOX.api,
                                                       site=push.ZENODO_SANDBOX.site, real_address=True)))))
    held.main(["--creator", "Carberry, Josiah", "--demo", "toy", "--dry-run",  # type: ignore[attr-defined]
               "--log", str(tmp_path / "log")])
    assert (_state.state_root() / "zenodo-sandbox").exists(), "the planted dry run's records are not seen"
    import shutil

    shutil.rmtree(_state.state_root() / "zenodo-sandbox")


def listing(folder: Path) -> list[tuple[str, bytes]]:
    return sorted((str(path.relative_to(folder)), path.read_bytes()) for path in folder.rglob("*") if path.is_file())


def test_the_lanes_dry_run_needs_no_token_says_what_it_compared_and_removes_what_it_writes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """No token in the environment or stored (the conftest's home holds none): the dry run holds one of its own. It
    compares the fake behind the real transport with the fake in this process, and says it asked no sandbox; it
    leaves no folder in the temporary directory, no record in the state directory, no log where none was asked for,
    and nothing written into a folder it was given."""
    temporary = own_temporary_folder(monkeypatch, tmp_path)
    given = saved_toy(tmp_path / "given")
    before = listing(given)
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    assert lane().main(["--creator", "Carberry, Josiah", "--demo", "toy", "--folder", str(given),  # type: ignore[attr-defined]
                        "--dry-run"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[-1].startswith("the fake behind the real transport answers as the fake in this process does (a dry "
                              "run: the sandbox was not asked")
    assert listing(given) == before and not (given / "sharing.yaml").exists()
    assert list(temporary.iterdir()) == [] and list((tmp_path / "cwd").iterdir()) == []
    assert not _state.state_root().exists() or not [p for p in _state.state_root().rglob("*") if p.is_file()]


def test_control_the_lanes_dry_run_comparison_finds_a_difference_planted_on_one_side(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    own_temporary_folder(monkeypatch, tmp_path)
    real = push.HttpTransport.send

    def changed(self, method, url, headers, body, token):  # noqa: ANN001, ANN202
        answer = real(self, method, url, headers, body, token)
        held = json.loads(answer.body) if answer.body.strip() else None
        if isinstance(held, dict) and "parent" in held:
            held["is_published"] = None
            return push.Answer(answer.status, answer.headers, json.dumps(held).encode())
        return answer

    monkeypatch.setattr(push.HttpTransport, "send", changed)
    assert lane().main(["--creator", "Carberry, Josiah", "--demo", "toy", "--dry-run"]) == 1  # type: ignore[attr-defined]
    out = capsys.readouterr().out
    assert "differences:" in out and "is_published" in out


def test_mime_offline_closes_push_after_the_plan(toy: Path, token: str, tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MIME_OFFLINE", "1")
    fake = FakeRecords(token)
    with pytest.raises(BuildRefused) as refused:
        pushed(copy_of(toy, tmp_path), fake)
    assert refused.value.codes() == ["push.service"] and "MIME_OFFLINE" in refused.value.refusals[0].message
    assert fake.requests == []



# --- the draft push reads is the one it made --------------------------------------------------------------------------


def two_studies(root: Path) -> tuple[Path, Path]:
    from push_cases import saved_named

    return saved_named(root, "other_study"), saved_named(root, "my_study")


def record_naming(service: push.Service, experiment: str, other: str) -> None:
    """Push's record of ``experiment`` edited to name the draft of ``other`` (its id and page, and its mark too: a
    whole record copied over, with its experiment's name changed)."""
    theirs = push.read_state(service, other)
    assert theirs is not None
    from dataclasses import replace

    _state.write_state(service, replace(theirs, experiment=experiment))


def test_a_record_naming_another_experiments_draft_changes_and_deletes_nothing_in_it(tmp_path: Path, token: str):
    fake = FakeRecords(token)
    other, mine = two_studies(tmp_path)
    theirs = pushed(other, fake)
    assert theirs is not None
    record = fake.records[theirs.record]
    before = (record.revision, dict(record.body["metadata"]), sorted(record.files))  # type: ignore[arg-type]
    record_naming(fake.service(), "my_study", "other_study")
    assert codes(lambda: pushed(mine, fake)) == ["push.draft-foreign"]
    assert codes(lambda: push.discard(mine, fake.service(), yes=True, transport=FakeTransport(fake), pause=0)) == \
        ["push.draft-foreign"]
    assert theirs.record in fake.records
    assert (record.revision, dict(record.body["metadata"]), sorted(record.files)) == before  # type: ignore[arg-type]
    said: list[str] = []
    push.status(mine, fake.service(), say=said.append, transport=FakeTransport(fake), pause=0)
    assert any("does not carry the identifier push wrote" in line for line in said)


def test_control_without_the_identifier_check_another_experiments_draft_is_rewritten_and_deleted(
        tmp_path: Path, token: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_draft, "_ours", lambda draft, service, state: True)
    fake = FakeRecords(token)
    other, mine = two_studies(tmp_path)
    theirs = pushed(other, fake)
    assert theirs is not None
    record_naming(fake.service(), "my_study", "other_study")
    try:
        pushed(mine, fake)
    except BuildRefused:
        pass
    assert fake.records[theirs.record].body["metadata"]["title"] == "my_study", \
        "the planted record's push is not seen rewriting the other experiment's draft"  # type: ignore[index]
    push.discard(mine, fake.service(), yes=True, transport=FakeTransport(fake), pause=0)
    assert theirs.record not in fake.records, "the planted discard is not seen deleting it"


def test_names_that_differ_only_in_case_have_records_apart_on_any_file_system(tmp_path: Path, token: str):
    from push_cases import saved_named

    fake = FakeRecords(token)
    upper, lower = _state.state_directory(fake.service(), "Toy"), _state.state_directory(fake.service(), "toy")
    assert str(upper).casefold() != str(lower).casefold()
    first = pushed(saved_named(tmp_path / "a", "Toy"), fake)
    second = pushed(saved_named(tmp_path / "b", "toy"), fake)
    assert first is not None and second is not None and first.record != second.record
    assert fake.records[first.record].body["metadata"]["title"] == "Toy"  # type: ignore[index]
    assert sorted(fake.records[first.record].files) == ["NOTICE", "Toy.mimebundle"]


def test_a_record_whose_experiment_differs_even_in_case_is_refused(tmp_path: Path, token: str):
    """What a case-folding file system does to the lookup (``Toy``'s directory found for ``toy``), emulated by copying
    the record across: read_state compares the name it holds."""
    from push_cases import saved_named

    fake = FakeRecords(token)
    pushed(saved_named(tmp_path / "a", "Toy"), fake)
    held = _state.state_directory(fake.service(), "Toy") / _state.STATE_FILE
    folded = _state.state_directory(fake.service(), "toy") / _state.STATE_FILE
    folded.parent.mkdir(parents=True)
    folded.write_bytes(held.read_bytes())
    assert codes(lambda: pushed(saved_named(tmp_path / "b", "toy"), fake)) == ["push.state"]


def test_a_record_from_before_push_marked_its_drafts_is_refused_by_name(toy: Path, token: str, tmp_path: Path):
    fake = FakeRecords(token)
    old = _state.state_root() / _state.namespace(fake.service()) / "full_toy" / _state.STATE_FILE
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"format": _state.OLD_FORMAT, "service": fake.service().name, "api": fake.api,
                               "experiment": "full_toy", "record": "629525"}), encoding="utf-8")
    with pytest.raises(BuildRefused) as refused:
        pushed(copy_of(toy, tmp_path), fake)
    assert refused.value.codes() == ["push.state"] and _state.OLD_FORMAT in refused.value.refusals[0].message
    assert fake.requests == []


# --- metadata edited on the site --------------------------------------------------------------------------------------


def test_a_title_edited_on_the_site_stops_the_push_naming_both_values(toy: Path, token: str, tmp_path: Path):
    from mime.build import RefusalClass

    fake = FakeRecords(token)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    record = fake.records["629525"]
    record.body["metadata"]["title"] = "Edited on the site"  # type: ignore[index]
    before = record.revision
    with pytest.raises(BuildRefused) as refused:
        pushed(folder, fake)
    held = refused.value.refusals
    assert [(r.code, r.category) for r in held] == [("push.metadata-edited", RefusalClass.USER)]
    assert "metadata.title on the site: Edited on the site; from sharing.yaml: full_toy" in held[0].message
    assert "sharing.yaml's `title` and push again" in held[0].action and push.OVERWRITE_FLAG in held[0].action
    assert record.revision == before, "nothing was written to the draft"
    said: list[str] = []
    push.status(folder, fake.service(), say=said.append, transport=FakeTransport(fake), pause=0)
    assert any("edited on the site since push last wrote it: title" in line for line in said)


def test_an_edit_copied_into_sharing_yaml_goes_on_and_the_flag_writes_over_one_that_is_not(toy: Path, token: str,
                                                                                         tmp_path: Path):
    fake = FakeRecords(token)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    record = fake.records["629525"]
    record.body["metadata"]["title"] = "Edited on the site"  # type: ignore[index]
    sharing = folder / "sharing.yaml"
    sharing.write_text(sharing.read_text(encoding="utf-8").replace('title: "full_toy"', 'title: "Edited on the site"'),
                       encoding="utf-8")
    assert pushed(folder, fake) is not None, "the site's title, copied into sharing.yaml, is no longer at stake"
    record.body["metadata"]["version"] = "v2 (the site's)"  # type: ignore[index]
    assert codes(lambda: pushed(folder, fake)) == ["push.metadata-edited"]
    seen = Seen(fake)
    assert pushed(folder, fake, seen=seen, overwrite_site_edits=True) is not None
    assert "version" not in record.body["metadata"]  # type: ignore[operator]
    assert any("wrote over what was edited on the site (--overwrite-site-edits): version" in event
               for event in seen.events)
    assert pushed(folder, fake) is not None, "after the flag, what push wrote is the draft's again"


def test_control_without_the_check_of_what_push_last_wrote_a_site_edit_is_overwritten(
        toy: Path, token: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_draft, "edited_fields", lambda draft, state: [])
    fake = FakeRecords(token)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    fake.records["629525"].body["metadata"]["title"] = "Edited on the site"  # type: ignore[index]
    pushed(folder, fake)
    assert fake.records["629525"].body["metadata"]["title"] == "full_toy", \
        "the planted push is not seen writing over the site's title"  # type: ignore[index]


# --- the link, last ---------------------------------------------------------------------------------------------------


def test_every_push_status_and_discard_ends_with_the_drafts_link(toy: Path, token: str, tmp_path: Path):
    fake = FakeRecords(token)
    folder = copy_of(toy, tmp_path)
    page = "https://zenodo-records.invalid/uploads/629525"
    seen = Seen(fake)
    pushed(folder, fake, seen=seen)
    assert seen.events[-2].startswith("said The draft 629525") and seen.events[-1] == f"said {page}"
    said: list[str] = []
    push.status(folder, fake.service(), say=said.append, transport=FakeTransport(fake), pause=0)
    assert said[-1] == page
    said.clear()
    push.push(folder, fake.service(), confirm=lambda question: False, say=said.append, transport=FakeTransport(fake),
              pause=0)
    assert said[-2:] == ["Not made: nothing was sent.", page]
    fake.plant("status", lambda method, path: path.endswith("/content"), status=500, times=99)
    fake.records["629525"].files.clear()
    said.clear()
    with pytest.raises(BuildRefused):
        push.push(folder, fake.service(), yes=True, say=said.append, transport=FakeTransport(fake), pause=0)
    assert said[-1] == page, "a refused push names the draft it left"
    fake.faults.clear()
    said.clear()
    push.discard(folder, fake.service(), confirm=lambda question: False, say=said.append, transport=FakeTransport(fake),
                 pause=0)
    assert said[-2:] == ["Not deleted.", page]
    said.clear()
    push.discard(folder, fake.service(), yes=True, say=said.append, transport=FakeTransport(fake), pause=0)
    assert said[-1] == "https://zenodo-records.invalid/me/uploads"


# --- the tests' own isolation -----------------------------------------------------------------------------------------


def test_control_a_record_written_outside_the_tests_state_directory_is_caught_by_the_guard(
        toy: Path, token: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outside_home: Outside):
    """The conftest's guard: with this test's isolation planted away (its directories set back to the module's, as a
    shell's XDG_STATE_HOME once named the real ones), a push's records land in the module's state directory, and the
    guard finds them. They are removed here, so the guard's own check after this test passes."""
    assert outside_home.written() == []
    isolate(monkeypatch, outside_home.home)
    pushed(copy_of(toy, tmp_path), FakeRecords(with_token(monkeypatch)))
    found = outside_home.written()
    assert any(path.name == _state.STATE_FILE for path in found), "the planted write is not seen by the guard"
    import shutil

    shutil.rmtree(outside_home.state)
