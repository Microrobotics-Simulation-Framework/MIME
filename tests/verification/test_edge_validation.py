"""§4 (v0.2 fit-up) — edge-validation cleanliness.

MADDENING v0.2's ``GraphManager.validate()`` flags graph edges whose
connected fields disagree in shape, dtype, or physical unit — the issues
``compile()`` raises as ``ShapeMismatchWarning`` / ``DtypeMismatchWarning`` /
``UnitMismatchWarning`` (subclasses of ``EdgeValidationWarning``).

This test pins MIME's representative experiment graphs as edge-validation
clean: each must report no shape/dtype/unit edge issue. If a future node
change introduces a mismatched edge it fails loudly here, rather than
letting MADDENING v0.3 escalate it to a hard error.

It calls ``validate()`` directly rather than ``compile()``: ``validate()``
*is* the edge check — ``compile()`` merely runs it, translates the result
into warnings, then JIT-compiles. Skipping the JIT keeps the test cheap and
side-steps the compile-time memory cost.
"""

import importlib.util
from pathlib import Path

import pytest

from mime.experiments.dejongh import default_mlp_weights_path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The dejongh graphs build an MLPResistanceNode from the trained Cholesky-MLP
# weights — a gitignored artifact (too large for the repo) absent in CI / fresh
# checkouts. Skip only those parametrisations there; the non-MLP experiments
# (e.g. umr_confinement) still exercise edge validation in CI.
_MLP_DEPENDENT = {"dejongh", "dejongh_new_chain"}
_requires_mlp_weights = pytest.mark.skipif(
    not default_mlp_weights_path().exists(),
    reason=f"MLP weights artifact absent (gitignored): {default_mlp_weights_path()}",
)

# validate() reports edge problems as strings with these prefixes; they are
# exactly what compile() routes to the EdgeValidationWarning subclasses.
_EDGE_ISSUE_PREFIXES = ("WARNING[shape]", "WARNING[dtype]", "WARNING[units]")


def _load_module(path, name):
    """Import a non-package experiment module (params.py / setup.py) by path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Experiment-graph registry -------------------------------------------
#
# Each builder returns an uncompiled GraphManager. To cover another
# experiment, add a builder and a registry entry. LBM/FVM experiments are
# built at a reduced resolution: edge-spec consistency does not depend on
# lattice size, and a full-resolution lattice would exceed the test
# environment's capped GPU memory (Cat B in the v0.2 fit-up notes). That
# minimal-build recipe is encapsulated per-experiment in its builder.

def _build_dejongh():
    from mime.experiments.dejongh import build_graph
    return build_graph()


def _build_dejongh_new_chain():
    from mime.experiments.dejongh_new_chain import build_graph
    return build_graph()


def _build_umr_confinement():
    physics = _REPO_ROOT / "experiments" / "umr_confinement" / "physics"
    params_mod = _load_module(physics / "params.py", "_umr_edgeval_params")
    params = {k: v for k, v in vars(params_mod).items() if not k.startswith("_")}
    params["RESOLUTION"] = 24  # edge specs are independent of lattice size
    setup_mod = _load_module(physics / "setup.py", "_umr_edgeval_setup")
    return setup_mod.build_graph(params)


_EXPERIMENT_BUILDERS = {
    "dejongh": _build_dejongh,                      # Stokeslet + MLP + lubrication
    "dejongh_new_chain": _build_dejongh_new_chain,  # magnetics + motor + actuation
    "umr_confinement": _build_umr_confinement,      # IBLBM fluid-structure (LBM)
}


@pytest.mark.parametrize("experiment", [
    pytest.param(name, marks=_requires_mlp_weights) if name in _MLP_DEPENDENT
    else name
    for name in sorted(_EXPERIMENT_BUILDERS)
])
def test_experiment_graph_edge_validation_clean(experiment):
    """The experiment graph carries no shape/dtype/unit edge mismatch."""
    gm = _EXPERIMENT_BUILDERS[experiment]()
    edge_issues = [i for i in gm.validate()
                   if i.startswith(_EDGE_ISSUE_PREFIXES)]
    assert not edge_issues, (
        f"{experiment}: graph has edge-validation issue(s):\n"
        + "\n".join(f"  {i}" for i in edge_issues)
    )
