"""Experiment hook infrastructure for the MIME runner.

Hooks allow experiments to provide custom logic (mesh generation,
scalar extraction, flow visualization, scene setup) without modifying
framework code. The runner loads hooks from the experiment's config
and dispatches to them during the simulation loop.

Hook signatures:
    mesh_generator:    (params: dict) -> (verts, tris) | None
    scalar_extractor:  (ctx: HookContext) -> dict[str, float]
    flow_extractor:    (ctx: HookContext) -> np.ndarray | None
    scene_setup:       (bridge, params, actor_config, env_config) -> None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class HookContext:
    """Per-step context passed to hook callables.

    Extensible without breaking existing hooks — new fields can be
    added without changing the call signature.
    """
    state: dict[str, dict[str, Any]]
    params: dict
    dt: float
    step: int
    ext_inputs: dict | None


@dataclass
class ExperimentHooks:
    """Collection of experiment-provided hook callables.

    All hooks are optional — None means the feature is skipped.
    """
    mesh_generator: Optional[Callable] = None
    scalar_extractor: Optional[Callable] = None
    flow_extractor: Optional[Callable] = None
    scene_setup: Optional[Callable] = None


def load_hooks(config: dict, experiment_dir: Path) -> ExperimentHooks:
    """Load experiment hooks from config['hooks'] section.

    Each hook entry is 'module.py:attribute_name', resolved relative
    to experiment_dir. Missing hooks or import failures produce
    warnings, not errors — features degrade gracefully.

    Parameters
    ----------
    config : dict
        Parsed experiment.yaml.
    experiment_dir : Path
        Root directory of the experiment.

    Returns
    -------
    ExperimentHooks
        Loaded hooks (None for any that failed or weren't specified).
    """
    hooks_config = config.get("hooks", {})
    if not hooks_config:
        return ExperimentHooks()

    hooks = ExperimentHooks()
    for hook_name in ("mesh_generator", "scalar_extractor",
                      "flow_extractor", "scene_setup"):
        spec = hooks_config.get(hook_name)
        if spec is None:
            continue
        loaded = _load_hook(spec, experiment_dir, hook_name)
        if loaded is not None:
            setattr(hooks, hook_name, loaded)

    return hooks


def _load_hook(
    spec: str,
    experiment_dir: Path,
    hook_name: str,
) -> Optional[Callable]:
    """Load a single hook from 'module.py:attribute_name' spec."""
    try:
        if ":" not in spec:
            logger.warning(
                "Hook '%s' spec '%s' missing ':attribute' — skipping",
                hook_name, spec,
            )
            return None

        module_path_str, attr_name = spec.rsplit(":", 1)
        module_path = experiment_dir / module_path_str

        if not module_path.exists():
            logger.warning(
                "Hook '%s' module not found: %s — skipping",
                hook_name, module_path,
            )
            return None

        import importlib.util
        module_name = f"experiment_hook_{hook_name}"
        loader_spec = importlib.util.spec_from_file_location(
            module_name, str(module_path),
        )
        module = importlib.util.module_from_spec(loader_spec)
        loader_spec.loader.exec_module(module)

        hook = getattr(module, attr_name, None)
        if hook is None:
            logger.warning(
                "Hook '%s': attribute '%s' not found in %s — skipping",
                hook_name, attr_name, module_path,
            )
            return None

        if not callable(hook):
            logger.warning(
                "Hook '%s': %s.%s is not callable — skipping",
                hook_name, module_path, attr_name,
            )
            return None

        logger.info("Hook '%s' loaded: %s:%s", hook_name, module_path_str, attr_name)
        return hook

    except Exception as e:
        logger.warning("Hook '%s' failed to load: %s — skipping", hook_name, e)
        return None


def serializable_state(state: dict) -> dict:
    """Strip framework-internal keys (prefixed with _) for serialization.

    State dicts may contain non-serializable framework objects (e.g.
    _fluid_field_provider). This utility filters them at both the
    top-level (node names) and per-node (state keys) level, producing
    a dict safe for jax.tree_util, HDF5 export, pickle, etc.
    """
    return {
        node_name: {
            k: v for k, v in node_state.items()
            if not k.startswith("_")
        }
        for node_name, node_state in state.items()
        if not node_name.startswith("_")
    }
