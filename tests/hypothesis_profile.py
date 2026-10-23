"""The Hypothesis profile every test run uses, CI's included: the profile Hypothesis would load here (its ``ci``
profile on CI, which derandomises, and ``default`` elsewhere) with ``print_blob`` set, so a failing property prints
the ``@reproduce_failure`` blob that replays it. It changes nothing else: no example count, no derandomisation, no
deadline. ``tests/conftest.py`` loads it."""

from hypothesis import settings

#: What Hypothesis would have used: read before the profile below is loaded.
BASELINE = settings.default
settings.register_profile("mime", parent=BASELINE, print_blob=True)
settings.load_profile("mime")
