"""``mime scaffold``: a new experiment, bundle, study, block or plugin from templates (:func:`scaffold`), every value from
``--set``, the person's ``defaults.toml`` or scaffold's own, never overwriting, never asking under ``--yes``."""

from mime.scaffold._scaffold import Scaffolded, scaffold
from mime.scaffold._values import KEYS, KINDS

__all__ = ["KEYS", "KINDS", "Scaffolded", "scaffold"]
