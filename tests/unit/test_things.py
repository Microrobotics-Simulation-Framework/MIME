import pytest

from mime.testing.toys import ToyBody, ToyField
from mime.things import (
    Apparatus,
    ApparatusHandle,
    Body,
    BodyHandle,
    FieldPath,
    Medium,
    State,
    ThingKind,
    did_you_mean,
)
from mime.units import m


def test_things_are_declarations_of_their_kind():
    assert Body("probe", dynamics=ToyBody(), initial=State(position=m((0, 0, 0)))).kind is ThingKind.BODY
    assert Medium("bath").dynamics is None
    assert dict(Apparatus("source", fields={"push": ToyField()}).fields) == {"push": ToyField()}


def test_a_handle_carries_the_name_and_kind():
    handle = BodyHandle("probe")
    assert (handle.name, handle.kind, str(handle)) == ("probe", ThingKind.BODY, "probe")


def test_an_apparatus_handle_gives_typed_paths_to_its_fields():
    source = ApparatusHandle("source", ["push", "pull"])
    assert source.push == FieldPath("source", "push")
    assert (source.push.owner, source.push.field, str(source.pull)) == ("source", "push", "source.pull")
    with pytest.raises(AttributeError, match=r"has no field 'puhs'. Did you mean 'push'\?"):
        source.puhs  # noqa: B018


def test_did_you_mean_lists_the_candidates():
    assert did_you_mean("prob", ["probe", "bath"]) == "Did you mean 'probe'? The candidates are: bath, probe."
    assert did_you_mean("zzz", ["probe"]) == "The candidates are: probe."
    assert did_you_mean("x", []) == "There are no candidates."
