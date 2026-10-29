"""The corrected unit-tube table as the tests read it: the shipped asset ``unit_tube_R0.997`` (209 MB), fetched into the
cache before the run (``mime assets fetch unit_tube_R0.997``; CI does it once per job into a cache keyed by the asset's
sha256, ``MIME_ASSETS_DIR``). A test never downloads: a missing asset fails by name, with the command that stores it."""

import functools
from pathlib import Path

from mime import assets

NAME = "unit_tube_R0.997"


@functools.cache
def path() -> Path:
    """The cached file, or ``BuildRefused`` (``asset.missing``) naming the asset and the command that fetches it."""
    return assets.locate(assets.reference(NAME))
