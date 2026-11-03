"""A publish's ``--confirm`` bound to what was shown (design #212 (5); the P2b review's F5, its stale case): the DOI
alone confirms the draft as the push that reserved that DOI, the first to print it, left it, as #171 asked; once a push
has changed the draft, the DOI alone is refused (``push.confirm-stale``) and ``--confirm <DOI>@<fingerprint>``, with the
fingerprint every push prints beside the DOI, publishes exactly that draft. A refusal sends nothing.

A draft whose DOI was reserved before push kept that record (``shown`` absent: #100's pushes) is confirmed by its DOI and
fingerprint alone, changed or not: no later push stands for the one that printed its DOI (the review of #112, F1). And
the fingerprint is held once more immediately before the publish request, computed from what is about to be published:
a push that lands while the publish reads the draft is caught there (F2).

Controls, each caught on every run:

- **a stale ``--confirm`` with the check planted away**: the folder changed and pushed again, the DOI alone the first
  push printed is caught publishing the second push's bundle;
- **a fingerprint that does not bind**: with the fingerprint planted to a constant, a fingerprint printed for the first
  draft is caught publishing the second;
- **``shown`` filled by any push** (the rule before the fix): a draft pushed before the binding, changed and pushed
  again, is caught published by the DOI alone the old push printed;
- **the check before the request planted away**: a push landing between the publish's first check and its read of the
  draft is caught published under the confirmation printed for the draft before it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import given as drawn
from hypothesis import settings
from hypothesis import strategies as st
from push_cases import copy_of, with_token
from test_push_publish import NUMBER, Terminal, _replace_content, changed, codes, published, publishes, pushed, toy
from zenodo_records import FakeRecords, FakeTransport

from mime import push
from mime.build import BuildRefused
from mime.push import _draft, _publish, _state

__all__ = ["changed", "toy"]  # the module's fixtures, from the publishing tests

DOI = f"10.5072/zenodo.{NUMBER}"


def printed(person: Terminal) -> str:
    """The ``--confirm`` a push printed, beside the DOI."""
    found = re.findall(r"--confirm (\S+@[0-9a-f]+)`", person.text())
    assert found, person.text()
    return found[-1]


def held_bundle(fake: FakeRecords) -> bytes:
    data = fake.records[NUMBER].files["full_toy.mimebundle"].data
    assert data is not None
    return data


def resaved(folder: Path, fake: FakeRecords, service: push.Service, changed: Path) -> tuple[str, str, bytes]:
    """The draft pushed, the folder changed (one reference step more) and pushed again, as P2b's review's p01 d2 did:
    what the first push printed, what the second printed, and the second push's bundle."""
    first = Terminal()
    pushed(folder, fake, service, first)
    _replace_content(folder, changed)
    second = Terminal()
    pushed(folder, fake, service, second)
    return printed(first), printed(second), held_bundle(fake)


def test_every_push_prints_the_doi_with_its_fingerprint_and_the_status_does_too(toy: Path, tmp_path: Path,
                                                                                  monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    first = Terminal()
    pushed(folder, fake, service, first)
    state = push.read_state(service, "full_toy")
    assert state is not None and printed(first) == f"{DOI}@{push.fingerprint(state)}"
    assert len(push.fingerprint(state)) == _state.FINGERPRINT_LENGTH
    again = Terminal()
    pushed(folder, fake, service, again)
    assert len(again.said) == 1 and printed(again) == printed(first), "an unchanged push says it in one line, with it"
    status = Terminal()
    push.status(folder, service, say=status.say, transport=FakeTransport(fake), pause=0)
    assert printed(status) == printed(first)


def test_the_doi_alone_publishes_a_draft_unchanged_since_the_push_that_printed_it(toy: Path, tmp_path: Path,
                                                                                    monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    pushed(folder, fake, service)  # nothing changed: the DOI alone still names what was shown
    assert published(folder, fake, service, None, confirm=DOI) is not None and publishes(fake) == 1


def test_a_stale_doi_alone_is_refused_before_any_request_and_the_last_fingerprint_publishes(
        toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    first, second, bundle = resaved(folder, fake, service, changed)
    assert first != second and first.partition("@")[0] == second.partition("@")[0] == DOI
    before = len(fake.requests)
    for confirm in (DOI, f"https://doi.org/{DOI}", first):
        assert codes(lambda: published(folder, fake, service, None, confirm=confirm)) == ["push.confirm-stale"], confirm
    assert len(fake.requests) == before, "a stale confirmation sends nothing"
    assert published(folder, fake, service, None, confirm=second.upper().replace("ZENODO", "zenodo")) is not None
    assert publishes(fake) == 1 and fake.records[NUMBER].published and held_bundle(fake) == bundle


def test_a_stale_refusal_names_no_fingerprint_a_script_could_retry_with_unseen(toy: Path, changed: Path,
                                                                                 tmp_path: Path,
                                                                                 monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    _, second, _ = resaved(folder, fake, service, changed)
    with pytest.raises(BuildRefused) as refused:
        published(folder, fake, service, None, confirm=DOI)
    text = str(refused.value)
    assert second.partition("@")[2] not in text and "--status" in text, text


def test_a_wrong_doi_is_still_not_confirmed_with_or_without_a_fingerprint(toy: Path, tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    person = Terminal()
    pushed(folder, fake, service, person)
    fingerprint = printed(person).partition("@")[2]
    for confirm in ("10.5072/zenodo.1", f"10.5072/zenodo.1@{fingerprint}", f"{DOI}@{fingerprint}x"):
        assert codes(lambda: published(folder, fake, service, None, confirm=confirm)) == ["push.not-confirmed"]
    assert publishes(fake) == 0


def test_a_publish_with_no_terminal_and_no_confirm_names_the_form_that_would_publish(
        toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    with pytest.raises(BuildRefused) as unchanged:
        push.publish(folder, service, say=Terminal().say, transport=FakeTransport(fake), pause=0)
    assert unchanged.value.codes() == ["push.unconfirmed"] and f"--confirm {DOI}`" in str(unchanged.value), \
        "the DOI alone, while it still names what was shown"
    _replace_content(folder, changed)
    person = Terminal()
    pushed(folder, fake, service, person)
    with pytest.raises(BuildRefused) as stale:
        push.publish(folder, service, say=Terminal().say, transport=FakeTransport(fake), pause=0)
    assert f"--confirm {printed(person)}`" in str(stale.value), "the DOI and its fingerprint, once a push changed it"
    assert publishes(fake) == 0


_HEX = "0123456789abcdefABCDEF"


@st.composite
def marks(draw: st.DrawFn) -> tuple[push.DraftState, str]:
    """A draft's state and a fingerprint to confirm it with: its own (in either case), its own with one character
    changed, or any other. Drawn so, the acceptance branch is reached as often as the refusal."""
    state = push.DraftState("zenodo-sandbox", "https://x.invalid/api", "toy", "1", "0", "10.5072/zenodo.1", "p",
                            files={draw(st.sampled_from(["a", "full_toy.mimebundle", "NOTICE.txt"])): push.Uploaded(
                                draw(st.integers(0, 1 << 40)), draw(st.text(_HEX, min_size=32, max_size=32)),
                                draw(st.text(_HEX, min_size=64, max_size=64)))},
                            sent=draw(st.text(_HEX, min_size=64, max_size=64)))
    own = push.fingerprint(state)
    at = draw(st.integers(0, _state.FINGERPRINT_LENGTH - 1))
    other = own[:at] + draw(st.sampled_from([each for each in "0123456789abcdef" if each != own[at]])) + own[at + 1:]
    return state, draw(st.one_of(st.just(own), st.just(own.upper()), st.just(other),
                                 st.text(_HEX, min_size=_state.FINGERPRINT_LENGTH, max_size=_state.FINGERPRINT_LENGTH)))


@settings(deadline=None, max_examples=200)
@drawn(drawn_mark=marks())
def test_any_fingerprint_but_the_drafts_own_is_stale(drawn_mark: tuple[push.DraftState, str]):
    state, mark = drawn_mark
    now = push.fingerprint(state)
    service = push.ZENODO_SANDBOX
    if mark.lower() == now:
        _publish._stale(state, f"10.5072/zenodo.1@{mark}", service)  # pyright: ignore[reportPrivateUsage]
    else:
        with pytest.raises(BuildRefused):
            _publish._stale(state, f"10.5072/zenodo.1@{mark}", service)  # pyright: ignore[reportPrivateUsage]


def test_control_with_the_check_planted_away_a_stale_doi_publishes_the_draft_it_was_not_printed_for(
        toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    _, _, bundle = resaved(folder, fake, service, changed)
    monkeypatch.setattr(_publish, "_stale", lambda state, confirm, service: None)
    monkeypatch.setattr(_publish, "_bound_at_post", lambda plan, state, confirm, service, sent: None)
    assert published(folder, fake, service, None, confirm=DOI) is not None
    assert publishes(fake) == 1 and held_bundle(fake) == bundle, "the planted defect (no binding) is not seen"


def test_control_with_a_fingerprint_that_binds_nothing_the_first_drafts_fingerprint_publishes_the_second(
        toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_state, "fingerprint", lambda state: "0" * _state.FINGERPRINT_LENGTH)
    monkeypatch.setattr(_publish, "fingerprint", _state.fingerprint)
    monkeypatch.setattr(_publish, "fingerprint_of", lambda sent, files: "0" * _state.FINGERPRINT_LENGTH)
    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    first, _, _ = resaved(folder, fake, service, changed)
    assert published(folder, fake, service, None, confirm=first) is not None and publishes(fake) == 1


# --- a draft whose DOI a push reserved before the binding (F1) ---------------------------------------------------------


def forget_shown(service: push.Service) -> None:
    """Push's record of the draft as #100's push left it: no ``shown``, which it did not keep."""
    path = _state.state_directory(service, "full_toy") / _state.STATE_FILE
    held = json.loads(path.read_text(encoding="utf-8"))
    del held["shown"]
    path.write_text(json.dumps(held, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def pushed_before_the_binding(folder: Path, fake: FakeRecords, service: push.Service, change: Path | None) -> str:
    """The draft pushed as #100's push left it, the folder changed by ``change`` (or not) and pushed again by this
    one: the ``--confirm`` the second push printed."""
    pushed(folder, fake, service)
    forget_shown(service)
    if change is not None:
        _replace_content(folder, change)
    person = Terminal()
    pushed(folder, fake, service, person)
    return printed(person)


@pytest.mark.parametrize("change", [True, False], ids=["changed", "unchanged"])
def test_a_draft_pushed_before_the_binding_is_confirmed_by_its_doi_and_fingerprint_alone(
        change: bool, toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    second = pushed_before_the_binding(folder, fake, service, changed if change else None)
    state = push.read_state(service, "full_toy")
    assert state is not None and state.shown == "", "a later push does not stand for the one that printed the DOI"
    before = len(fake.requests)
    for confirm in (DOI, f"https://doi.org/{DOI}"):
        with pytest.raises(BuildRefused) as refused:
            published(folder, fake, service, None, confirm=confirm)
        assert refused.value.codes() == ["push.confirm-stale"] and "no record of the draft" in str(refused.value)
    assert len(fake.requests) == before, "a refusal sends nothing"
    assert published(folder, fake, service, None, confirm=second) is not None and publishes(fake) == 1


def test_control_with_shown_filled_by_any_push_a_pre_binding_drafts_doi_alone_publishes_the_changed_draft(
        toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    real = _draft.Drafting.draft

    def any_push(self: _draft.Drafting) -> push.DraftState:
        found = real(self)
        self.reserved = not found.shown.startswith(f"{found.doi}@")  # the rule before the fix: any push fills it
        return found

    monkeypatch.setattr(_draft.Drafting, "draft", any_push)
    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    pushed_before_the_binding(folder, fake, service, changed)
    bundle = held_bundle(fake)
    assert published(folder, fake, service, None, confirm=DOI) is not None
    assert publishes(fake) == 1 and held_bundle(fake) == bundle, "the planted defect (shown filled late) is not seen"


# --- a push that lands while the publish reads the draft (F2) ----------------------------------------------------------


def with_a_push_landing(folder: Path, fake: FakeRecords, service: push.Service, changed: Path,
                        monkeypatch: pytest.MonkeyPatch) -> tuple[str, Callable[[], object]]:
    """The draft pushed, and the folder changed; the publish then made with the confirmation the push printed, a second
    push of the changed folder landing between the publish's first check and its read of the draft (standing in for
    another ``mime push`` process): that confirmation, and the publish."""
    person = Terminal()
    pushed(folder, fake, service, person)
    _replace_content(folder, changed)
    real = _publish.planned

    def landing(root: Path, service: push.Service, state: push.DraftState) -> object:
        monkeypatch.setattr(_publish, "planned", real)
        pushed(folder, fake, service)
        return real(root, service, state)

    monkeypatch.setattr(_publish, "planned", landing)
    confirm = printed(person)
    return confirm, lambda: published(folder, fake, service, None, confirm=confirm)


@pytest.mark.parametrize("form", ["fingerprint", "doi"])
def test_a_push_landing_while_the_publish_reads_the_draft_is_caught_before_the_request(
        form: str, toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    confirm, publish = with_a_push_landing(folder, fake, service, changed, monkeypatch)
    if form == "doi":
        confirm = DOI
        publish = (lambda: published(folder, fake, service, None, confirm=DOI))
    with pytest.raises(BuildRefused) as refused:
        publish()
    assert refused.value.codes() == ["push.confirm-stale"] and "immediately before the publish" in str(refused.value)
    assert publishes(fake) == 0 and not fake.records[NUMBER].published, confirm


def test_control_with_the_check_before_the_request_planted_away_the_landed_push_is_published(
        toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_publish, "_bound_at_post", lambda plan, state, confirm, service, sent: None)
    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    _, publish = with_a_push_landing(folder, fake, service, changed, monkeypatch)
    assert publish() is not None and publishes(fake) == 1, "the planted defect (no check before the request) is not seen"
