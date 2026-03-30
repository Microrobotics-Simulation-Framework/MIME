"""Experiment hook re-exports for UMR confinement.

This module makes the experiment's contract with the runner scannable:
all hook callables are re-exported here from their implementation
modules. The experiment.yaml hooks section points into this module.

Stateful hooks (ScalarExtractor) are instantiated here — the runner
receives ready-to-call objects.
"""

import sys
from pathlib import Path

# Ensure experiment subdirectories are importable
_exp_dir = str(Path(__file__).parent)
if _exp_dir not in sys.path:
    sys.path.insert(0, _exp_dir)

from analysis.scalars import ScalarExtractor
from analysis.flow_extract import extract_flow
from geometry.mesh import generate_mesh
from scene.setup_viz import setup_scene

# Stateful — lazy-inits on first call via ctx.params
scalar_extractor = ScalarExtractor()

# Stateless function hooks
mesh_generator = generate_mesh
flow_extractor = extract_flow
scene_setup = setup_scene
