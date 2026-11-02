"""The environment a run is made in, pinned where it can be and recorded where it cannot.

MIME ships, for each variant (``cpu``, ``cuda12``), the hashed requirements its ``uv.lock`` gives (:func:`pins`), and
for each Python series the base image it is tested on, pinned by digest (:func:`bases`). :func:`check`, which ``mime
env check`` runs, holds an environment to them before a run and applies the precision the experiment's lock records.
What no pin can hold (the GPU and its driver, the CPU, XLA's flags, the runner) is recorded in the lock
(:func:`gpu_record`, :func:`cpu_record`, :func:`runner_record`), and :func:`xla_cache_dir` names the directory XLA's
compilation cache is kept in. :func:`modal_script` writes the script that runs a saved experiment on Modal; nothing in
MIME imports ``modal``.
"""

from mime.env._check import (
    INSTALLERS,
    EnvRefused,
    EnvReport,
    System,
    apply_lock_precision,
    check,
    environment_modules,
    fingerprint,
    gpu_precision_refusals,
    installed_distributions,
    lab_pins_of,
    lock_matmul_precision,
    lock_x64,
    shadow_refusals,
)
from mime.env._gpus import EXACT_GPUS, LOOSE_GPUS, gpu_type, is_kind_of
from mime.env._host import (
    DETERMINISTIC_FLAG,
    RUNNER_INFO_VARIABLE,
    RUNNER_VARIABLE,
    CpuRecord,
    GpuRecord,
    cpu_record,
    deterministic,
    gpu_record,
    matmul_precision,
    runner_info,
    runner_record,
    with_deterministic,
    xla_flags,
)
from mime.env._modal_script import ModalScript, modal_script
from mime.env._pins import (
    EXPORT_ARGUMENTS,
    VARIANTS,
    Base,
    Pin,
    applicable,
    bases,
    parse_requirements,
    pins,
    requirements_bytes,
    requirements_name,
    series_of,
)
from mime.env._xla_cache import (
    USER_CACHE_CAP_BYTES,
    clear_user_xla_cache,
    prune_to_cap,
    use_xla_cache,
    user_xla_cache,
    user_xla_cache_root,
    xla_cache_dir,
)

__all__ = [
    "DETERMINISTIC_FLAG",
    "EXACT_GPUS",
    "EXPORT_ARGUMENTS",
    "INSTALLERS",
    "LOOSE_GPUS",
    "RUNNER_INFO_VARIABLE",
    "RUNNER_VARIABLE",
    "USER_CACHE_CAP_BYTES",
    "VARIANTS",
    "Base",
    "CpuRecord",
    "EnvRefused",
    "EnvReport",
    "GpuRecord",
    "ModalScript",
    "Pin",
    "System",
    "applicable",
    "apply_lock_precision",
    "bases",
    "check",
    "clear_user_xla_cache",
    "cpu_record",
    "deterministic",
    "environment_modules",
    "fingerprint",
    "gpu_precision_refusals",
    "gpu_record",
    "gpu_type",
    "installed_distributions",
    "is_kind_of",
    "lab_pins_of",
    "lock_matmul_precision",
    "lock_x64",
    "matmul_precision",
    "modal_script",
    "parse_requirements",
    "pins",
    "prune_to_cap",
    "requirements_bytes",
    "requirements_name",
    "runner_info",
    "runner_record",
    "series_of",
    "shadow_refusals",
    "use_xla_cache",
    "user_xla_cache",
    "user_xla_cache_root",
    "with_deterministic",
    "xla_cache_dir",
    "xla_flags",
]
