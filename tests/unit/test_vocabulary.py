import pytest

from mime import vocabulary
from mime.testing.toys import LOAD, TORQUE
from mime.vocabulary import Interface, ReadKind, TargetKind


def test_the_six_target_kinds_and_how_they_combine():
    assert [kind.value for kind in TargetKind] == [
        "load", "kinematic", "operator", "equality constraint", "inequality constraint", "registration"]
    assert {k for k in TargetKind if k.additive} == {TargetKind.LOAD, TargetKind.KINEMATIC, TargetKind.OPERATOR}
    assert {k for k in TargetKind if k.exclusive} == {TargetKind.EQUALITY_CONSTRAINT}
    assert not TargetKind.INEQUALITY_CONSTRAINT.delivered_by_edge and not TargetKind.REGISTRATION.delivered_by_edge


def test_a_reserved_name_fixes_kind_units_frame_and_sign_and_leaves_its_shape_open():
    assert TORQUE.reserved and TORQUE.shape is None
    assert (TORQUE.kind, TORQUE.units, TORQUE.frame, TORQUE.sign) == (TargetKind.LOAD, "N m", "world", "on the body")
    with pytest.raises(ValueError, match="stays open"):
        Interface("toy.body.spin", TargetKind.LOAD, "N m", "world", (3,), reserved=True)
    with pytest.raises(ValueError, match="only a reserved entry"):
        Interface("toy.body.spin", TargetKind.LOAD, "N m", "world", None, magnitude=1.0)


def test_an_entry_that_carries_values_declares_a_typical_magnitude():
    with pytest.raises(ValueError, match="typical magnitude"):
        Interface("toy.body.spin", ReadKind.STATE, "rad/s", "world", (3,))


def test_names_are_namespaced_and_versioned():
    assert vocabulary.lookup("toy.body.load") is LOAD
    assert LOAD.namespace == "toy" and vocabulary.namespace_of("body.applied_wrench") == "mime"
    assert vocabulary.versions()["toy"] == "0.1.0" and "mime" in vocabulary.versions()
    with pytest.raises(ValueError, match="not 'mime'"):
        vocabulary.register_namespace("mime", "1")
    with pytest.raises(ValueError, match="registered at version 0.1.0"):
        vocabulary.register_namespace("toy", "9.9.9")


def test_redefining_a_name_differently_is_refused():
    assert vocabulary.define(LOAD) is LOAD
    with pytest.raises(ValueError, match="already defined differently"):
        vocabulary.define(Interface(LOAD.name, TargetKind.OPERATOR, "N", "world", (3,), magnitude=1.0))
