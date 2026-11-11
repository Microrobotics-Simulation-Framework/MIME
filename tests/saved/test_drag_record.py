"""The drag's resolved record, and so the input hash of every program on a rung that does not read ``coefficients``, is
the released drag 0.4.0's: the setting the rft rung added to 0.4.0 is left out of a record whose rung reads none
(``Entry.unrecorded_settings``), so a folder saved before it loads. Drag 0.5.0, which had no release before the setting,
records every setting on every rung (its hook is empty): a program on 0.5.0 holds ``coefficients``, null off the rft
rung.

The pinned hash is the input hash of ``mrs_free_space`` (drag 0.4.0) as the tree before the rft rung computed it, in
the saved form it was pinned in (the tenth: a later form's own key moves every hash, so the input is written back into
that form, as a load of a folder of it does). The controls: with 0.4.0's setting recorded (its hook emptied) the same
program's input hash moves, so the pin can fail; with 0.5.0's hook set to 0.4.0's rule, its programs off the rft rung
lose the setting, so the 0.5.0 assertion can fail."""

from unittest import mock

from cases import script_a_unblended
from mrs_cases import mrs_free_space
from rft_cases import rft_free_space, rft_kept

from mime.interactions import Drag
from mime.interactions.hydrodynamic import UncoupledDrag
from mime.saved._record import input_hash
from mime.saved._versions import SOURCE_SCALE_ADDED, document_in_form

MRS_FREE_SPACE_INPUT_HASH = "sha256:0898c45a1399359eb07d20898effdb996b027a81e0abb405f9ad25a9fc754636"


def _drag_settings(experiment, version: str):
    found = [e["settings"] for e in experiment.build().record.resolved["entries"].values()
             if e["name"] == "drag" and e["version"] == version]
    assert found
    return found


def _pinned_form_hash(experiment) -> str:
    """The experiment's input hash in the form the pin was taken in."""
    record = experiment.build().record
    assert record.document is not None
    return input_hash(document_in_form(record.document, SOURCE_SCALE_ADDED), record.resolved, list(record.entries))


def test_a_released_program_keeps_its_input_hash():
    assert _pinned_form_hash(mrs_free_space()) == MRS_FREE_SPACE_INPUT_HASH


def test_the_control_with_the_setting_recorded_the_hash_moves():
    with mock.patch.object(UncoupledDrag, "unrecorded_settings", lambda self: frozenset()):
        assert _pinned_form_hash(mrs_free_space()) != MRS_FREE_SPACE_INPUT_HASH


def test_on_0_4_0_the_record_holds_coefficients_only_for_the_rung_that_reads_it():
    assert all("coefficients" not in settings for settings in _drag_settings(mrs_free_space(), "0.4.0"))
    assert all("coefficients" in settings for settings in _drag_settings(rft_kept(), "0.4.0"))


def test_on_0_5_0_the_record_holds_coefficients_on_every_rung():
    for experiment in (script_a_unblended(), rft_free_space()):
        assert all("coefficients" in settings for settings in _drag_settings(experiment, "0.5.0"))


def test_the_control_with_0_4_0s_rule_on_0_5_0_a_program_off_the_rft_rung_loses_it():
    with mock.patch.object(Drag, "unrecorded_settings", UncoupledDrag.unrecorded_settings):
        assert all("coefficients" not in settings for settings in _drag_settings(script_a_unblended(), "0.5.0"))
