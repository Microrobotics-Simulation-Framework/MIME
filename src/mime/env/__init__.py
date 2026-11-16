"""The environment a run is made in, pinned where it can be and recorded where it cannot.

MIME ships, for each variant (``cpu``, ``cuda12``), the hashed requirements its ``uv.lock`` gives, and for each Python
series the base image it is tested on, pinned by digest. :func:`check`, which ``mime env check`` runs, holds an
environment to them before a run and returns an :class:`EnvReport` (its ``rows`` are :class:`DistributionRow`), or
raises :class:`EnvRefused`. What no pin can hold (the GPU and its driver, the CPU, XLA's flags, the runner) is recorded
in the experiment's lock, and :func:`apply_lock_precision` applies a saved folder's recorded precision before it is
loaded. :func:`deterministic_mode` turns XLA's deterministic GPU mode on before a run;
:func:`user_xla_cache` and :func:`use_xla_cache` keep XLA's compilation cache, in :func:`user_xla_cache_root`, which
:func:`prune_to_cap` caps; :data:`EXACT_GPUS` are the GPU names Modal serves exactly. ``mime modal`` writes the script
that runs a saved experiment on Modal; nothing in MIME imports ``modal``.
"""

from mime.env._check import DistributionRow, EnvRefused, EnvReport, apply_lock_precision, check
from mime.env._gpus import EXACT_GPUS
from mime.env._host import deterministic_mode
from mime.env._xla_cache import prune_to_cap, use_xla_cache, user_xla_cache, user_xla_cache_root

__all__ = [
    "EXACT_GPUS",
    "DistributionRow",
    "EnvRefused",
    "EnvReport",
    "apply_lock_precision",
    "check",
    "deterministic_mode",
    "prune_to_cap",
    "use_xla_cache",
    "user_xla_cache",
    "user_xla_cache_root",
]
