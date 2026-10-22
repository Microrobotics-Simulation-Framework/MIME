"""The planted failure: CI's control that every lane runs its tests and can go red.

The ``plant_failure`` input of a manual CI run sets ``MIME_CI_PLANT_FAILURE=1``; every lane must then fail here.
"""

import os


def test_the_canary_fails_when_a_failure_is_planted():
    assert os.environ.get("MIME_CI_PLANT_FAILURE") != "1", "planted failure: MIME_CI_PLANT_FAILURE=1"
