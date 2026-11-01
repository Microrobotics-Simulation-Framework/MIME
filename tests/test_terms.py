"""Licences as SPDX expressions and the verdict each gives a file (``mime.terms``, designs #198, #199).

A licence is an SPDX expression in its canonical spelling, or ``NOASSERTION``; free text is refused, and so is another
spelling of a real one, with the canonical spelling named. The verdict is in OpenMaterica's words: ``permitted`` only
for a licence on MIME's open or conditional list, or by a declared permission; every other real licence is
``unknown``, and travels as a reference; ``AND`` takes the most restrictive, ``OR`` the freest, ``WITH`` keeps its
licence's verdict.

The strategies draw real ids from the SPDX list ``packaging`` ships, the expected verdicts are ``terms_cases``' own
table, and the controls plant what each test guards against (an unknown licence judged permitted; the AND rule made
the freest) and must be caught on every one of twenty runs.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
from hypothesis import given
from hypothesis import strategies as st
from terms_cases import CONDITIONAL_IDS, NO_REDISTRIBUTION, OPEN_IDS, UNKNOWN_IDS

from mime import terms
from mime.assets import AssetRef, known
from mime.build import BuildRefused
from mime.terms import (
    CustomLicence,
    Evidence,
    Permission,
    Scope,
    Verdict,
    licence_problem,
    licence_verdict,
    permission_problems,
    permission_verdict,
)

open_ids = st.sampled_from(OPEN_IDS)
conditional_ids = st.sampled_from(sorted(CONDITIONAL_IDS))
unknown_ids = st.sampled_from(UNKNOWN_IDS)
any_ids = st.one_of(open_ids, conditional_ids, unknown_ids)
SHA = "a" * 64


def expected(identifier: str) -> Verdict:
    """This file's reading of a single id's verdict, written apart from the code."""
    if identifier in OPEN_IDS:
        return Verdict("permitted", "open")
    if identifier in CONDITIONAL_IDS:
        return Verdict("permitted", "conditional", CONDITIONAL_IDS[identifier])
    return Verdict("unknown", "referenced")


# --- the expression ----------------------------------------------------------------------------------------------


@given(any_ids)
def test_every_real_id_in_its_canonical_spelling_is_a_licence(identifier: str):
    assert licence_problem(identifier) is None


@given(any_ids)
def test_another_spelling_of_a_real_id_is_refused_naming_the_canonical_one(identifier: str):
    spelt = identifier.lower() if identifier.lower() != identifier else identifier.upper()
    problem = licence_problem(spelt)
    assert problem is not None and repr(identifier) in problem


@pytest.mark.parametrize("text", ["MIT License", "CC BY 4.0", "not stated", "public domain", "", "Apache 2",
                                  "GPL", "MIT,Apache-2.0", "MIT AND", "(MIT", "LicenseRef-", "LicenseRef-a_b"])
def test_free_text_is_refused(text: str):
    assert licence_problem(text) is not None


@pytest.mark.parametrize("text", [None, 3, b"MIT", ["MIT"]])
def test_a_licence_that_is_not_text_is_refused(text: object):
    assert licence_problem(text) is not None


def test_noassertion_alone_is_a_licence_that_says_nobody_stated_one():
    assert licence_problem("NOASSERTION") is None
    assert licence_verdict("NOASSERTION") == Verdict("unknown", "referenced")
    assert licence_problem("MIT AND NOASSERTION") is not None


def test_an_expression_is_held_to_its_canonical_spelling():
    assert licence_problem("MIT OR Apache-2.0") is None
    assert licence_problem("MIT  OR  Apache-2.0") is not None
    assert licence_problem("mit or apache-2.0") is not None


@given(any_ids, any_ids)
def test_the_ids_of_an_expression_are_its_licences_never_an_operator_or_an_exception(one: str, two: str):
    """An id's ``+`` ("or later", which a few deprecated ids of the list carry in their name) is read as the id."""
    expression = f"({one} WITH LLVM-exception) OR {two}"
    assert terms.ids_of(expression) == tuple(dict.fromkeys((one.removesuffix("+"), two.removesuffix("+"))))


# --- the verdict -------------------------------------------------------------------------------------------------


@given(any_ids)
def test_a_single_licence_has_the_verdict_of_its_list(identifier: str):
    assert licence_verdict(identifier) == expected(identifier)


@given(unknown_ids)
def test_no_unknown_licence_is_permitted(identifier: str):
    verdict = licence_verdict(identifier)
    assert verdict.status == "unknown" and verdict.tier == "referenced" and not verdict.carried


@given(st.lists(any_ids, min_size=2, max_size=5))
def test_and_holds_every_licence_at_once(identifiers: list[str]):
    verdict = licence_verdict(" AND ".join(identifiers))
    singles = [expected(each) for each in identifiers]
    if any(single.tier == "referenced" for single in singles):
        assert verdict.tier == "referenced" and verdict.status == "unknown"
    else:
        conditions = {condition for single in singles for condition in single.conditions}
        assert verdict.status == "permitted" and set(verdict.conditions) == conditions
        assert verdict.tier == ("conditional" if conditions else "open")


@given(st.lists(any_ids, min_size=2, max_size=5))
def test_or_lets_the_recipient_take_the_freest(identifiers: list[str]):
    verdict = licence_verdict(" OR ".join(identifiers))
    singles = [expected(each) for each in identifiers]
    if any(single.tier == "open" for single in singles):
        assert verdict == Verdict("permitted", "open")
    elif any(single.tier == "conditional" for single in singles):
        fewest = min(len(single.conditions) for single in singles if single.tier == "conditional")
        assert verdict.tier == "conditional" and len(verdict.conditions) == fewest
    else:
        assert verdict.tier == "referenced"


@given(any_ids)
def test_an_exception_keeps_its_licences_verdict(identifier: str):
    assert licence_verdict(f"{identifier} WITH Classpath-exception-2.0") == expected(identifier)


@given(any_ids, any_ids, any_ids)
def test_and_binds_tighter_than_or(one: str, two: str, three: str):
    assert licence_verdict(f"{one} OR {two} AND {three}") == licence_verdict(f"{one} OR ({two} AND {three})")


def test_a_custom_licence_is_unknown_unless_a_person_says_it_forbids_redistribution():
    custom = {NO_REDISTRIBUTION.id: NO_REDISTRIBUTION}
    assert licence_verdict(NO_REDISTRIBUTION.id, custom) == Verdict("not_permitted", "referenced")
    assert licence_verdict("LicenseRef-Somebody", custom) == Verdict("unknown", "referenced")
    assert licence_verdict(f"MIT AND {NO_REDISTRIBUTION.id}", custom).status == "not_permitted"
    assert licence_verdict(f"MIT OR {NO_REDISTRIBUTION.id}", custom) == Verdict("permitted", "open")


def test_a_custom_licence_cannot_say_it_permits_redistribution():
    held = CustomLicence("LicenseRef-Mine", "Mine", "https://example.org/mine", "permitted")
    assert held.problem() is not None and "Permission" in (held.problem() or "")
    assert CustomLicence("LicenseRef-Mine", "Mine", "https://example.org/mine").problem() is None
    for bad in (CustomLicence("Mine", "Mine", "https://example.org/mine"),
                CustomLicence("LicenseRef-Mine", " ", "https://example.org/mine"),
                CustomLicence("LicenseRef-Mine", "Mine", "http://example.org/mine")):
        assert bad.problem() is not None


@given(st.booleans(), st.booleans())
def test_a_permission_that_grants_redistribution_is_permitted_with_the_conditions_it_leaves(modify: bool,
                                                                                            commercial: bool):
    verdict = permission_verdict(Scope(True, modify, commercial))
    conditions = {*(() if commercial else ("non-commercial",)), *(() if modify else ("no-derivatives",))}
    assert verdict.status == "permitted" and set(verdict.conditions) == conditions
    assert verdict.tier == ("conditional" if conditions else "open")
    assert permission_verdict(Scope(False, modify, commercial)).tier == "referenced"


@given(unknown_ids, st.booleans(), st.booleans())
def test_a_permission_only_ever_raises_a_files_tier(identifier: str, modify: bool, commercial: bool):
    raised = terms.freest([licence_verdict(identifier), permission_verdict(Scope(True, modify, commercial))])
    assert raised.status == "permitted"
    assert terms.freest([licence_verdict("CC0-1.0"), permission_verdict(Scope(True, modify, commercial))]) == \
        Verdict("permitted", "open")


# --- a permission's form ----------------------------------------------------------------------------------------


def a_permission(**changes: object) -> Permission:
    held: dict[str, object] = {"grantor": "Ada Maker, founder, Robot Maker Ltd", "grantee": "A. Lab, Uni of Somewhere",
                               "files": (SHA,), "scope": Scope(True, False, False), "date": "2026-11-20",
                               "evidence": Evidence("written_permission", "an email of 2026-11-20, held by the grantee")}
    held.update(changes)
    return Permission(**held)  # type: ignore[arg-type]


def test_a_complete_permission_has_no_problem():
    assert permission_problems(a_permission(), [SHA]) == []
    statement = a_permission(evidence=Evidence("public_statement", "https://example.org/statement"))
    assert permission_problems(statement, [SHA]) == []


@pytest.mark.parametrize("change, named", [
    ({"grantor": ""}, "grantor"), ({"grantee": " "}, "grantee"), ({"date": "20 Nov 2026"}, "date"),
    ({"date": "2026-02-30"}, "date"), ({"scope": Scope(False, True, True)}, "redistribution"),
    ({"scope": "redistribute"}, "scope"), ({"evidence": None}, "evidence"),
    ({"evidence": Evidence("an email", "held")}, "evidence"),  # type: ignore[arg-type]
    ({"evidence": Evidence("written_permission", "")}, "pointer"),
    ({"evidence": Evidence("public_statement", "a letter")}, "https"),
])
def test_an_incomplete_permission_is_refused_naming_the_part(change: dict[str, object], named: str):
    problems = permission_problems(a_permission(**change), [SHA])
    assert problems and any(named in problem for problem in problems), problems


def test_a_permission_names_its_files_by_the_sha256_of_their_bytes():
    assert any("no file" in each for each in permission_problems(a_permission(), []))
    assert any("not the sha256" in each for each in permission_problems(a_permission(), ["scene/layers/x.usda"]))


# --- an asset's licence ------------------------------------------------------------------------------------------


def test_the_shipped_references_hold_spdx_licences():
    held = known()
    assert len(held) >= 13
    assert all(licence_problem(ref.licence) is None for ref in held.values())


@pytest.mark.parametrize("licence", ["CC0", "MIT License", "not stated", "cc0-1.0", "MIT  OR Apache-2.0"])
def test_an_asset_reference_with_free_text_for_its_licence_is_refused(licence: str):
    with pytest.raises(BuildRefused) as refused:
        AssetRef("table", SHA, 10, licence, ("https://example.org/table.npz",))
    assert refused.value.codes() == ["asset.reference"]


@pytest.mark.parametrize("licence", ["CC0-1.0", "NOASSERTION", "LicenseRef-Robot-Maker", "MIT OR Apache-2.0"])
def test_an_asset_reference_takes_an_spdx_licence(licence: str):
    assert AssetRef("table", SHA, 10, licence, ("https://example.org/table.npz",)).licence == licence


# --- the controls ------------------------------------------------------------------------------------------------

RUNS = 20


@contextmanager
def unknown_judged_permitted() -> Iterator[None]:
    """The plant: a licence on neither list judged open, as a reading that took every real licence for an open one
    would."""
    real = terms._leaf
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(terms, "_leaf", lambda identifier, custom: real(identifier, custom)
                      if identifier in terms.OPEN or identifier in terms.CONDITIONAL else Verdict("permitted", "open"))
        yield


@contextmanager
def and_made_freest() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(terms, "_all", terms._any)
        yield


def unknown_is_never_permitted(identifier: str) -> None:
    verdict = licence_verdict(identifier)
    assert verdict.status != "permitted" and not verdict.carried, (identifier, verdict)


def and_is_most_restrictive(identifier: str) -> None:
    assert licence_verdict(f"MIT AND {identifier}").tier == "referenced"


def planted_runs(scenario: Callable[[str], None], plant: Callable[[], object]) -> int:
    """How many of twenty runs catch the plant: each run takes the next real unknown id, and is caught when the
    scenario's assertion fails."""
    caught = 0
    for run in range(RUNS):
        identifier = UNKNOWN_IDS[(run * 37) % len(UNKNOWN_IDS)]
        with plant():  # type: ignore[attr-defined]
            try:
                scenario(identifier)
            except AssertionError:
                caught += 1
    return caught


def test_control_an_unknown_licence_judged_permitted_is_caught_every_run():
    assert planted_runs(unknown_is_never_permitted, unknown_judged_permitted) == RUNS


def test_control_and_made_the_freest_is_caught_every_run():
    assert planted_runs(and_is_most_restrictive, and_made_freest) == RUNS


def test_the_control_scenarios_pass_unplanted():
    for identifier in UNKNOWN_IDS[:40]:
        unknown_is_never_permitted(identifier)
        and_is_most_restrictive(identifier)
