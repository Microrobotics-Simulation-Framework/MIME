"""The profiles in a build (design #190 items 1, 4 and 6, #191 items 2 to 4): exploratory waives what the experiment
leaves unstated with a pending reason, records it and stamps it; research needs each reason and refuses a pending one
by name, where a ``"TODO"`` text passes the check before it. Each refusal's control is the same experiment sealed, each
pending reason replaced by a text, which builds in research. A setting's source is given on its quantity or by
``exp.source``, once; a path naming nothing is refused with the candidates. Validity limits and input errors are refused
in every profile, and a profile changes no value and no graph."""

from dataclasses import replace

import pytest
from profile_cases import BARE_PENDING, SHEAR, Paperwork, settling_ball

from mime.build import BuildRefused
from mime.units import kg_per_m3, m, mm, standard_gravity
from mime.validity import chosen, pending

SEALED = Paperwork(profile="research", acceptance="water is Newtonian at a settling ball's shear rates",
                   density="a steel ball's density", spacing=mm(0.2, reason="converged at half this spacing"),
                   blob=0.5, quadrature_refinement=4,
                   sources={"drag:ball.blob": "the regularised method's usual blob",
                            "drag:ball.quadrature_refinement": "the quadrature's spacing half the blob"})


def codes(paperwork: Paperwork) -> list[str]:
    with pytest.raises(BuildRefused) as refused:
        settling_ball(paperwork).plan()
    return sorted(refused.value.codes())


def test_exploratory_waives_what_a_new_user_leaves_unstated_and_says_so():
    plans = settling_ball().plan()
    outcome = plans.profile
    assert [item.target for item in outcome.pending] == list(BARE_PENDING)
    generated = [w for w in outcome.waivers if w.generated]
    assert {w.reasons for w in generated} == {(SHEAR,), ("default_numerical",)}
    taken = {d.path: (d.kind.value, d.value) for d in outcome.defaults}
    assert taken["drag:ball.spacing"] == ("numerical", m(0.000125)), "a compact body's 338 samples (D9)"
    assert taken["drag:ball.blob"] == ("numerical", 0.1), "a compact body's blob (drag 0.4.0)"
    assert taken["drag:ball.quadrature_refinement"] == ("numerical", 20), "two over the blob"
    assert taken["drag:ball.wall_kind"] == ("determined", "free_space")
    assert taken["gravity"][0] == "convention"
    assert outcome.stamp().text().startswith("profile exploratory: 5 items pending, not sealed")


def test_research_refuses_each_by_name():
    assert codes(Paperwork(profile="research")) == ["record.unknown", *["setting.numerical-default"] * 3,
                                                    "value.reason-pending"]


def test_the_control_the_same_experiment_sealed_builds_in_research():
    plans = settling_ball(SEALED).plan()
    assert plans.profile.pending == ()
    assert plans.profile.stamp().text() == "profile research: nothing pending"


def test_a_pending_reason_is_refused_by_name_where_a_todo_text_passes():
    """Today's check takes any text, so a placeholder passes it; ``pending()`` is the placeholder research refuses."""
    settling_ball(replace(SEALED, density="TODO")).plan()
    assert codes(replace(SEALED, density=pending())) == ["value.reason-pending"]


def test_an_acceptance_with_a_pending_reason_is_stamped_in_exploratory_and_refused_in_research():
    plans = settling_ball(Paperwork(acceptance=pending())).plan()
    kinds = {item.target: item.kind for item in plans.profile.pending}
    assert kinds["material 'water' fluid.dynamic_viscosity"] == "accepted, its reason pending"
    assert codes(replace(SEALED, acceptance=pending())) == ["material.accept-pending"]


def test_a_source_given_twice_or_naming_nothing_is_refused_with_the_candidates():
    twice = Paperwork(density="steel", radius_reason="a datasheet", sources={"ball.parts.shell.radius": "also"})
    assert "source.duplicate" in codes(twice)
    with pytest.raises(BuildRefused) as refused:
        settling_ball(Paperwork(density="steel", sources={"ball.parts.shell.radus": "a datasheet"})).plan()
    unknown = [r for r in refused.value.refusals if r.code == "source.unknown"]
    assert unknown and "ball.parts.shell.radius" in unknown[0].action
    declared_twice = settling_ball(Paperwork(density=None))
    declared_twice.declare("ball.parts.shell.density", kg_per_m3(7850.0, reason="a"), reason="b")
    with pytest.raises(BuildRefused) as refused:
        declared_twice.plan()
    assert "source.duplicate" in refused.value.codes()


def test_a_pending_source_is_stamped_in_exploratory_and_refused_in_research():
    plans = settling_ball(Paperwork(density="steel", sources={"dt": pending()})).plan()
    assert "dt" in [item.target for item in plans.profile.pending]
    assert codes(replace(SEALED, sources={**SEALED.sources, "dt": pending()})) == ["source.pending"]


def test_research_lists_unsourced_physical_and_unevidenced_numerical_values_and_refuses_neither():
    listed = {item.target: item.kind for item in settling_ball(SEALED).plan().profile.listed}
    assert listed["ball.parts.shell.radius"] == "no source"
    assert listed["conditions.temperature"] == "no source"
    assert listed["dt"] == "no convergence evidence"
    assert "drag:ball.spacing" not in listed  # its reason states its evidence
    assert settling_ball(Paperwork(density="steel")).plan().profile.listed == ()


def test_a_value_chosen_by_the_experimenter_is_a_declared_change_of_its_role():
    sealed = replace(SEALED, radius_reason=chosen("the radius this study sweeps"))
    plans = settling_ball(sealed).plan()
    outcome = plans.profile
    assert outcome.changes == {"ball.parts.shell.radius": "physical"}
    assert "  declared change: ball.parts.shell.radius: declared physical, given here as the experimenter's choice " \
           "(chosen: the radius this study sweeps)" in outcome.report_lines()
    assert "ball.parts.shell.radius" not in [item.target for item in outcome.listed]
    assert outcome.sources["ball.parts.shell.radius"] == chosen("the radius this study sweeps")


def test_gravity_is_standard_by_convention_and_stated_when_written():
    exploratory = settling_ball(Paperwork(density="steel"))
    assert "gravity" not in exploratory.document()
    stated = settling_ball(SEALED)
    stated._settings = replace(stated._settings, gravity=standard_gravity())
    assert stated.document()["gravity"] == {"value": 9.80665, "unit": "m/s^2"}
    assert "gravity" in [item.target for item in stated.plan().profile.listed]


def test_gravity_at_its_convention_is_listed_in_research_and_takes_a_source():
    """N2: ``exp.source("gravity", ...)`` names gravity whether it is written or left to its convention; research lists
    the convention taken until it has one (a physical convention, which may not hold for the setup)."""
    conventional = settling_ball(SEALED).plan().profile
    (listed,) = [item for item in conventional.listed if item.target == "gravity"]
    assert listed.kind == "a convention, no source"
    sourced = settling_ball(replace(SEALED, sources={**SEALED.sources, "gravity": "the lab is at sea level"})).plan()
    assert "gravity" not in [item.target for item in sourced.profile.listed]
    assert sourced.profile.sources["gravity"] == "the lab is at sea level"


def test_a_choice_of_a_computed_default_is_no_choice_and_research_refuses_it():
    """W2: ``exp.source(spacing, reason=chosen(...))`` on a spacing left to its computed default chooses a value nobody
    wrote, so research refuses it by name, as it does a text reason there. The control: the value written, with the same
    choice, builds."""
    left = replace(SEALED, spacing=None, sources={**SEALED.sources, "drag:ball.spacing": chosen("TODO")})
    assert codes(left) == ["setting.numerical-default"]
    assert codes(replace(left, sources={**SEALED.sources, "drag:ball.spacing": "a text"})) == \
        ["setting.numerical-default"]
    written = replace(SEALED, spacing=mm(0.2, reason=chosen("the spacing this study sweeps")))
    outcome = settling_ball(written).plan().profile
    assert outcome.changes == {"drag:ball.spacing": "numerical"}


def test_research_lists_a_physical_convention_taken_and_not_a_design_one():
    """K6: a convention on a physical setting (a tube centred on the origin, a segment at its frame's origin, no
    servo lag) is what anyone assumes untold and may not hold: research lists each one taken, with the line that states
    its source. A screw's starts are required (the maintainer's decision), so script A writes them."""
    from script_a import script_a

    listed = {item.target: item for item in _research(script_a()).listed}
    assert listed["tube.centre"].kind == "a convention, no source"
    assert listed["umr.parts.shell.starts"].kind == "no source", "a screw's starts are required, and written"
    assert listed["tube.centre"].python == 'exp.source("tube.centre", reason=<reason>)'
    assert not [target for target in listed if target.endswith(".rate")], "a numerical convention is not listed"
    from chain_cases import kinematic_rpm

    arm = {item.target: item.kind for item in _research(kinematic_rpm()).listed}
    conventions = {target for target, kind in arm.items() if kind == "a convention, no source"}
    assert {"rpm.arm.servo", "rpm.motor.at", "rpm.motor.orientation"} <= conventions, sorted(arm)


def _research(experiment):
    experiment._settings = replace(experiment._settings, profile="research")
    try:
        return experiment.plan().profile
    except BuildRefused as refused:
        return refused.value.report.profile


def test_a_lab_s_undeclared_default_is_reported_stamped_in_exploratory_and_refused_in_research():
    """W5 (D11 reversed): a lab's setting with a written default and no declared role is reported among the defaults
    as kind undeclared, stamped pending in exploratory and refused by name in research, the refusal naming both ways
    out. The control: the same value with its source builds in research."""
    from test_dispatch import LabSpeed, lab_speed

    from mime.scene import Z
    from mime.units import s

    def labbed(paperwork):
        experiment = settling_ball(paperwork)
        experiment.observe(LabSpeed("ball", along=-Z, after=s(0.004)))
        return experiment

    path = "labdispatch.speed:ball.smoothing"
    with lab_speed():
        explored = labbed(replace(SEALED, profile="exploratory")).plan().profile
        assert {d.path: d.kind.value for d in explored.defaults}[path] == "undeclared"
        assert path in [item.target for item in explored.pending]
        with pytest.raises(BuildRefused) as refused:
            labbed(SEALED).plan()
        (found,) = [r for r in refused.value.refusals if r.code == "setting.undeclared-default"]
        assert path in found.message and "setting_roles" in found.action and "exp.source" in found.action
        sourced = labbed(replace(SEALED, sources={**SEALED.sources, path: "the lab's usual smoothing"})).plan()
        assert path not in {d.path for d in sourced.profile.defaults}


@pytest.mark.parametrize("profile", ["exploratory", "research"])
def test_input_errors_and_validity_limits_are_refused_in_every_profile(profile):
    wrong_units = settling_ball(replace(SEALED, profile=profile, density=None))
    wrong_units.declare("ball.parts.shell.density", m(7850.0), reason="a slip of the unit")
    with pytest.raises(BuildRefused) as refused:
        wrong_units.plan()
    assert "value.units" in refused.value.codes()
    unpicked = settling_ball(replace(SEALED, profile=profile))
    unpicked._settings = replace(unpicked._settings, conditions=None)
    with pytest.raises(BuildRefused) as refused:
        unpicked.plan()
    assert "record.no-pick" in refused.value.codes()


def test_strict_is_refused_by_name():
    assert "profile.reserved" in codes(replace(SEALED, profile="strict"))


def test_a_profile_changes_no_value_and_no_graph():
    """The per-step cost is the graph's, and a profile acts at build time: the sealed experiment's graph is the same
    under both profiles, hash for hash."""
    research = settling_ball(SEALED).build()
    exploratory = settling_ball(replace(SEALED, profile="exploratory")).build()
    assert research.record.hashes()["derived"] == exploratory.record.hashes()["derived"]
    bare = settling_ball().build()
    # the defaults' values as the rule states them, written and built in research: a default nudged in exploratory's
    # builds alone moves the bare graph from this one
    written = settling_ball(replace(SEALED, spacing=m(0.000125, reason="the default's value, written"),
                                    blob=0.1, quadrature_refinement=20)).build()
    assert bare.record.hashes()["derived"] == written.record.hashes()["derived"]


def test_a_source_whose_reason_is_none_is_refused():
    for reason in ("", "   ", 3.0):
        assert "source.reason" in codes(Paperwork(density="steel", sources={"dt": reason}))
    assert "source.reason" in codes(Paperwork(density="steel", radius_reason=chosen("")))
