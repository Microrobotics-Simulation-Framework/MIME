"""The drag's resolved record, and so the input hash of every program on a rung that does not read ``coefficients``, is
the released drag 0.4.0's: the setting the rft rung added is left out of a record whose rung reads none (``Entry.
unrecorded_settings``), so a folder saved before it loads.

The pinned hash is the input hash of ``mrs_free_space`` as the tree before the rft rung computed it, in saved form 10,
the form it was measured in: both tests hash the program in that form (``hashes(version=10)``), since a later form's
schema line moves every input hash and would make the control pass whatever the hook does. The control: with the
setting recorded (the hook removed) the same program's input hash moves, so the pin can fail."""

from unittest import mock

from mime.interactions import Drag
from mrs_cases import mrs_free_space
from rft_cases import rft_free_space

MRS_FREE_SPACE_INPUT_HASH = "sha256:0898c45a1399359eb07d20898effdb996b027a81e0abb405f9ad25a9fc754636"
#: The saved form the pin was measured in.
PINNED_FORM = 10


def _drag_settings(experiment):
    found = [e["settings"] for e in experiment.build().record.resolved["entries"].values() if e["name"] == "drag"]
    assert found
    return found


def test_a_released_program_keeps_its_input_hash():
    assert mrs_free_space().build().record.hashes(version=PINNED_FORM)["input"] == MRS_FREE_SPACE_INPUT_HASH


def test_the_control_with_the_setting_recorded_the_hash_moves():
    with mock.patch.object(Drag, "unrecorded_settings", lambda self: frozenset()):
        assert mrs_free_space().build().record.hashes(version=PINNED_FORM)["input"] != MRS_FREE_SPACE_INPUT_HASH


def test_the_record_holds_coefficients_only_for_the_rung_that_reads_it():
    assert all("coefficients" not in settings for settings in _drag_settings(mrs_free_space()))
    assert all("coefficients" in settings for settings in _drag_settings(rft_free_space()))
