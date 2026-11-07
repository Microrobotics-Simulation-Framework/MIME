"""The sweep takes no saved-form step: a study's points and a run's overrides live in run folders and results, never in
the saved form, so ``CURRENT`` stays the form of the tip this branch last merged (8, from #182)."""

from mime.saved._versions import CURRENT


def test_the_saved_form_is_the_one_the_branch_merged():
    assert CURRENT == 8
