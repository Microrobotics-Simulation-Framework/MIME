"""A unit bug in the camera's route lookup is the entry's defect, never a kind the camera cannot trace (usability bundle 2,
the re-check's minor). ``subjects_of`` asks ``route_of`` for each body part's and each structure's route and takes a refusal
for a kind with none: a body part is left out of the picture with its reason, a structure with a drawn mesh is traced from
the mesh. A ``UnitError`` is a ``MimeError``, so that handler saw one raised inside the lookup as the lookup's own refusal:
``call.units``, the user's mistake, for a body, and a tube that built unseen. Each case plants the re-check's unit bug
(a time read as a length) under one handler and gives the code the base gave, ``entry.defect``. **The controls:** the same
plant raising a plain ``ValueError`` gives the same codes (the handler never saw it), and a real refusal of the lookup is
still the handler's (a body's ``camera.route``; a drawn tube is traced from its mesh)."""

import pytest
from in_loop_cases import ballistic

import mime.sensing._scene as scene
from mime.build import ArgumentRefused, MimeError, Refusal, RefusalClass
from mime.units import mm, ms, s


def unit_bug() -> None:
    s(1.0).to(mm)


def plain_bug() -> None:
    raise ValueError("a bug of the lookup's that is no unit error")


def a_kinds_refusal() -> None:
    raise ArgumentRefused([Refusal("camera.route", RefusalClass.USER, "this kind has no route", "leave it out")])


def planted(monkeypatch: pytest.MonkeyPatch, kind: str, failure):
    real = scene.route_of

    def route_of(asked: str, *rest, **named):
        if asked == kind:
            failure()
        return real(asked, *rest, **named)

    monkeypatch.setattr(scene, "route_of", route_of)


def codes_of_building(monkeypatch: pytest.MonkeyPatch, kind: str, failure) -> list[str] | None:
    """The codes the build is refused with, or ``None`` where it builds."""
    planted(monkeypatch, kind, failure)
    try:
        ballistic(ms(3)).build()
    except MimeError as error:
        return error.codes()
    return None


@pytest.mark.parametrize("kind", ["sphere", "tube"])
def test_a_unit_bug_in_the_route_lookup_is_the_entrys_defect_as_at_the_base(monkeypatch, kind):
    assert codes_of_building(monkeypatch, kind, unit_bug) == ["entry.defect"]


@pytest.mark.parametrize("kind", ["sphere", "tube"])
def test_the_control_a_plain_error_there_gives_the_same_codes(monkeypatch, kind):
    assert codes_of_building(monkeypatch, kind, plain_bug) == ["entry.defect"]


def test_the_control_a_kinds_own_refusal_for_a_body_is_still_the_handlers():
    """A body part of a kind with no route is left out with its reason, which the build refuses with."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        assert codes_of_building(monkeypatch, "sphere", a_kinds_refusal) == ["camera.route"]


def test_the_control_a_kinds_own_refusal_for_a_drawn_structure_is_traced_from_its_mesh():
    with pytest.MonkeyPatch.context() as monkeypatch:
        assert codes_of_building(monkeypatch, "tube", a_kinds_refusal) is None
