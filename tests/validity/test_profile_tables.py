"""The profiles' two tables and the rule that holds them together (design #190 items 2 and 3, #184 item 1).

Classes as data: each finding code's class, each class with what covers it; the profile rules, (profile, class) to an
action. The code that applies them reads a finding's class, never a code. **Profiles act on waivers alone**: every rule
that covers a finding covers it with a waiver, and :func:`waivers_alone` finds any that does not. Its controls plant a
rule that generates an extrapolation, and one that waives a class no waiver covers; a property plants a bad row in
random tables and is found on every draw. a1 adds a class and a row as data: a test adds one (a waiver for an
unchecked condition, which MIME refuses today) and the build waives it with no other edit."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from profile_cases import Paperwork, settling_ball

from mime.build import BuildRefused
from mime.materials import Extrapolation
from mime.validity import (
    FINDING_CLASSES,
    PROFILE_RULES,
    Action,
    Cover,
    Finding,
    FindingClasses,
    Judge,
    Profile,
    ProfileRules,
    Rule,
    Waiver,
    pending_waiver,
    profile_of,
    rule_problems,
    using_tables,
    waivers_alone,
)


def test_mime_s_tables_class_the_codes_mime_meets_today():
    assert FINDING_CLASSES.class_of("condition_unstated:shear_rate") == "assumption"
    assert FINDING_CLASSES.class_of("condition_unstated:temperature") == "assumption"
    assert FINDING_CLASSES.class_of("stated_without_origin") == "assumption"
    # every other code keeps the behaviour it had before profiles: no class, so no rule
    for code in ("condition_unchecked:test_method", "condition_uncertain:temperature", "no_pick", "outside_range",
                 "condition_outside:shear_rate"):
        assert FINDING_CLASSES.class_of(code) is None, code
        assert Judge(Profile.EXPLORATORY).rule(code) is None


def test_the_rules_are_exploratory_s_and_research_s_for_an_assumption():
    assert Judge(Profile.EXPLORATORY).action("stated_without_origin") is Action.GENERATE
    assert Judge(Profile.RESEARCH).action("stated_without_origin") is Action.BASIS
    assert Judge(Profile.STRICT).action("stated_without_origin") is None


def test_profiles_act_on_waivers_alone():
    assert waivers_alone(FINDING_CLASSES, PROFILE_RULES) == []


def test_the_control_a_rule_that_generates_an_extrapolation_is_found():
    planted = PROFILE_RULES.with_rule(Profile.EXPLORATORY, "assumption", Rule(
        Action.GENERATE, make=lambda finding: Extrapolation("constant", basis="planted", reason="planted")))
    found = waivers_alone(FINDING_CLASSES, planted)
    assert found and "generates Extrapolation" in found[0], found


def test_the_control_a_rule_that_waives_what_no_waiver_covers_is_found():
    classes = FINDING_CLASSES.with_class("not_covered", Cover.MODEL_CHANGE, ["outside_range"])
    planted = PROFILE_RULES.with_rule(Profile.EXPLORATORY, "not_covered", Rule(Action.GENERATE))
    found = waivers_alone(classes, planted)
    assert found and "covered by model_change" in found[0], found
    listed = PROFILE_RULES.with_rule(Profile.EXPLORATORY, "not_covered", Rule(Action.LIST))
    assert waivers_alone(classes, listed) == []


def test_the_control_holds_in_the_build_a_planted_extrapolation_is_refused():
    """End to end: under the planted rule, the build's own product is not a waiver, and the build refuses it."""
    planted = PROFILE_RULES.with_rule(Profile.EXPLORATORY, "assumption", Rule(
        Action.GENERATE, make=lambda finding: Extrapolation("constant", basis="planted", reason="planted")))
    experiment = settling_ball(Paperwork(density="steel's density", spacing=None))
    with using_tables(FINDING_CLASSES, planted), pytest.raises(BuildRefused) as refused:
        experiment.plan()
    assert {"profile.not-a-waiver", "profile.tables"} <= set(refused.value.codes())
    plans = settling_ball(Paperwork(density="steel's density")).plan()
    assert all(isinstance(w, Waiver) for w in plans.profile.waivers)


def test_the_control_a_rule_whose_product_passes_the_table_check_and_not_the_build_is_refused():
    """A rule whose product is a waiver on the table check's sample and something else on a real finding passes the
    check: the build holds every product it makes, at every site, and refuses it by name."""
    def sly(finding: Finding) -> object:
        return pending_waiver(finding) if finding.text == "a planted finding" else Extrapolation(
            "constant", basis="planted", reason="planted")

    planted = PROFILE_RULES.with_rule(Profile.EXPLORATORY, "assumption", Rule(Action.GENERATE, make=sly))
    assert rule_problems(FINDING_CLASSES, planted) == []
    with using_tables(FINDING_CLASSES, planted), pytest.raises(BuildRefused) as refused:
        settling_ball(Paperwork(density="steel's density")).plan()
    assert "profile.not-a-waiver" in refused.value.codes() and "profile.tables" not in refused.value.codes()


actions = st.sampled_from(list(Action))
covers = st.sampled_from(list(Cover))
names = st.text(alphabet="abcdefgh", min_size=1, max_size=6)


@st.composite
def tables(draw: st.DrawFn) -> tuple[FindingClasses, ProfileRules, list[tuple[Profile, str]]]:
    """Random tables, and the rows in them a waiver cannot cover or whose product is not a waiver."""
    classes = FindingClasses({}, {})
    rules = ProfileRules({})
    bad: list[tuple[Profile, str]] = []
    for index, name in enumerate(draw(st.lists(names, min_size=1, max_size=6, unique=True))):
        cover = draw(covers)
        classes = classes.with_class(name, cover, [f"code_{index}"])
        for profile in draw(st.lists(st.sampled_from(list(Profile)), unique=True, max_size=3)):
            action = draw(actions)
            planted = action is Action.GENERATE and draw(st.booleans())
            make = (lambda finding: Extrapolation("linear", basis="planted", reason="planted")) if planted \
                else pending_waiver
            rules = rules.with_rule(profile, name, Rule(action, make))
            allowed = {Action.GENERATE: {Cover.WAIVER}, Action.BASIS: {Cover.WAIVER, Cover.SOURCE}}.get(action)
            if planted or (allowed is not None and cover not in allowed):
                bad.append((profile, name))
    return classes, rules, bad


@given(tables())
def test_the_check_finds_every_planted_row_and_nothing_else(drawn):
    classes, rules, bad = drawn
    found = waivers_alone(classes, rules)
    named = {line.split(":")[0] for line in found}
    assert named == {f"{profile.value} {name}" for profile, name in bad}


def test_a1_adds_a_class_and_a_row_with_no_other_edit():
    """An unchecked condition (water read with no test method stated) has no class today and is refused; a table
    with one more class and one more row waives it, in the same build code."""
    def unrequired(profile: str = "exploratory"):
        return settling_ball(Paperwork(profile=profile, density="steel's density", acceptance="Newtonian",
                                       require=False))

    with pytest.raises(BuildRefused) as refused:
        unrequired().plan()
    assert "record.unknown" in refused.value.codes()
    classes = FINDING_CLASSES.with_class("omission", Cover.WAIVER, ["condition_unchecked:*"])
    rules = PROFILE_RULES.with_rule(Profile.EXPLORATORY, "omission", Rule(Action.GENERATE))
    with using_tables(classes, rules):
        plans = unrequired().plan()
        with pytest.raises(BuildRefused):
            unrequired("research").plan()
    waived = {w.target: w.reasons for w in plans.profile.waivers if w.generated and w.target.startswith("material")}
    assert waived == {"material 'water' bulk.density": ("condition_unchecked:test_method",),
                      "material 'water' fluid.dynamic_viscosity": ("condition_unchecked:test_method",)}


def test_a_table_that_repeats_a_row_or_names_no_cover_is_refused():
    with pytest.raises(ValueError, match="has a row already"):
        FindingClasses.read("a waiver x\na waiver y\n")
    with pytest.raises(ValueError, match="is in the class"):
        FindingClasses.read("a waiver x\nb waiver x\n")
    with pytest.raises(ValueError):
        FindingClasses.read("a nowhere x\n")
    with pytest.raises(ValueError, match="has a rule"):
        ProfileRules.read("exploratory a generate\nexploratory a list\n")


def test_strict_is_refused_by_name_and_an_unknown_profile_with_the_profiles():
    assert profile_of("research") == (Profile.RESEARCH, None)
    assert profile_of(Profile.EXPLORATORY) == (Profile.EXPLORATORY, None)
    assert profile_of("strict")[1][0] == "profile.reserved"
    unknown = profile_of("lax")[1]
    assert unknown[0] == "profile.unknown" and "'exploratory', 'research'" in unknown[2]


def test_a_generated_waiver_covers_its_finding_with_a_pending_basis():
    made = pending_waiver(Finding("material 'water' fluid.dynamic_viscosity", "condition_unstated:shear_rate", "x"))
    assert made.generated and made.reasons == ("condition_unstated:shear_rate",) and str(made.basis) == "pending"
