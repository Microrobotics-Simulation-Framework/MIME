"""experiments/dejongh_confined starts the body pre-sunk, as its params.py says.

Before the MADDENING 0.4.1 port, physics/setup.py looked for
``gm.set_initial_state`` (which MADDENING never had) and then for a private
``_initial_state`` on ``gm._nodes["body"]`` (a node spec, which has none): the
start silently stayed at the origin although params.py sets INIT_Y_M = -1 mm.
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parents[2] / "experiments" / "dejongh_confined"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _params() -> dict:
    namespace: dict = {}
    exec((EXP_DIR / "physics" / "params.py").read_text(), namespace)
    return {k: v for k, v in namespace.items() if not k.startswith("_")}


def test_body_starts_where_params_put_it():
    params = _params()
    expected = [params["INIT_X_M"], params["INIT_Y_M"], params["INIT_Z_M"]]
    assert expected[1] == -1.0e-3

    gm = _load("dejongh_setup", EXP_DIR / "physics" / "setup.py").build_graph(params)
    position = np.asarray(gm.get_node_state("body")["position"])
    np.testing.assert_array_equal(position, np.asarray(expected, dtype=position.dtype))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    np.testing.assert_array_equal(np.asarray(gm.get_node_state("body")["position"]), position)
