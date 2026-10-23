"""CI's planted failures: the controls that every lane runs its tests and can go red, and that the time budget can.

The ``plant_failure`` input of a manual CI run sets ``MIME_CI_PLANT_FAILURE=1``; every lane must then fail here. The
``plant_slow_tests`` input sets ``MIME_CI_PLANT_SLOW_TESTS=1``: the two tests below then take 21 s and 6 s, neither is
on ``tests/duration_allowlist.txt``, and every lane's ``Test durations`` step must fail the job on the first (still
over the hard line when it is re-run alone) and warn on the second. Otherwise both return at once.
"""

import os
import time

PLANTED_SLOW = os.environ.get("MIME_CI_PLANT_SLOW_TESTS") == "1"


def test_the_canary_fails_when_a_failure_is_planted():
    assert os.environ.get("MIME_CI_PLANT_FAILURE") != "1", "planted failure: MIME_CI_PLANT_FAILURE=1"


def test_a_planted_test_over_the_hard_line():
    if PLANTED_SLOW:
        time.sleep(21.0)


def test_a_planted_test_over_the_policy_line():
    if PLANTED_SLOW:
        time.sleep(6.0)
