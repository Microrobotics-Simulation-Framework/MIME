"""A determined default (design #191 item 4): the mrs drag's ``wall_kind``, the one wall kind that matches the region
the medium fills, resolved in every profile and reported; refused with the candidates where several match, and the
one written kept. The control registers a second wall kind for a medium that fills no region, which makes the default
ambiguous."""

import dataclasses

import pytest
from profile_cases import Paperwork, settling_ball

from mime.build import BuildRefused
from mime.registry import EntryKind, WallForm, WallKind, register_wall_kind, registered
from mime.registry._registry import _by_class, _by_key  # noqa: PLC2701 - the control unregisters what it planted
from mime.gradients import GradientStatus
from mime.vocabulary import register_namespace


def test_the_wall_kind_is_determined_in_every_profile():
    """In exploratory the build takes it; in research too, where the build is refused for the drag's numerical defaults
    and its report still carries what the profile made of the experiment."""
    exploratory = settling_ball(Paperwork(density="steel", acceptance="Newtonian")).plan().profile
    with pytest.raises(BuildRefused) as refused:
        settling_ball(Paperwork(profile="research", density="steel", acceptance="Newtonian")).plan()
    assert refused.value.codes() == ["setting.numerical-default"] * 3
    research = refused.value.report.profile
    for outcome in (exploratory, research):
        taken = {d.path: d for d in outcome.defaults}
        assert taken["drag:ball.wall_kind"].value == "free_space"
        assert taken["drag:ball.wall_kind"].kind.value == "determined"


def test_the_control_two_matching_wall_kinds_refuse_with_the_candidates():
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Unbounded(WallKind):
        name = "determined.unbounded"
        version = "0.0.1"
        differentiability = GradientStatus.EXACT
        form = WallForm.ANALYTIC
        matches = None
        reads = ()

    register_namespace("determined", "0.0.1")
    register_wall_kind(Unbounded)
    try:
        with pytest.raises(BuildRefused) as refused:
            settling_ball(Paperwork(density="steel", acceptance="Newtonian")).plan()
        (found,) = [r for r in refused.value.refusals if r.code == "drag.wall-kind"]
        assert "which 2 wall kinds match (determined.unbounded 0.0.1, free_space 0.1.0)" in found.message
        written = settling_ball(Paperwork(density="steel", acceptance="Newtonian", wall_kind="free_space")).plan()
        assert "drag:ball.wall_kind" not in {d.path for d in written.profile.defaults}
    finally:
        record = next(r for r in registered(EntryKind.WALL_KIND) if r.cls is Unbounded)
        del _by_key[(record.kind, record.name, record.version)]
        del _by_class[Unbounded]
