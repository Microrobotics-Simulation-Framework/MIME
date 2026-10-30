"""The Cholesky MLP as a type (phase 7i): ``dejongh_mlp_v2`` is a ``Surrogate`` record naming its evaluator
(``cholesky_mlp 0.1.0``), its outputs, the features it reads, its accuracy per use and its provenance, and the drag's
node reads the weights through the evaluator its record names.

**The gate:** the node, reading the weights through the type, reproduces the 4c1 MLP's golden outputs (the 0.2 port's
``MLPResistanceNode`` and ``LubricationCorrectionNode`` on fixed poses, ``references/port_drag*``) within the 1e-12
``test_port_drag.py`` declares, off the axis, with and without lubrication. **The control:** a record naming another
evaluator, the same network with its diagonal floored at 1e-4 instead of 1e-5, misses them, on every run.

The record's own rules: a surrogate outputs a resistance, a flow load only where its evaluator computes one; one
accuracy per use, and an unmeasured one is ``None``, not a bound; an evaluator that is not registered is refused by
name, and so is a record whose features are not the ones the node builds."""

import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mime.nodes.hydro._surrogate_drag as drag_module
from mime.data import asset
from mime.nodes.hydro import SurrogateDragNode
from mime.surrogates import (
    CHOLESKY_MLP,
    DEJONGH_MLP_V2,
    SCREW_FEATURES,
    Evaluator,
    Provenance,
    Surrogate,
    _cholesky_mlp,
    evaluator_named,
    evaluators,
    predict_resistance,
    read_cholesky_mlp,
    register_evaluator,
)

REFERENCES = Path(__file__).parent / "references"
PORT = np.load(REFERENCES / "port_drag.npz")
KINDS = list(json.loads((REFERENCES / "port_drag_provenance.json").read_text())["kinds"])
#: The poses ``test_port_drag.py`` compares off the axis: the kinds inside, between, past the clamp and near the axis.
OFF_AXIS = np.isin(PORT["kind"], [KINDS.index(name) for name in ("inside", "between", "past_clamp", "near_axis")])
RTOL = 1e-12
NU, LENGTH, RADIUS, TUBE, MU, EPSILON = 2.33, 7.47e-3, 1.56e-3, 3.175e-3, 1e-3, 1.5e-4


def planted_resistance(network: _cholesky_mlp.CholeskyMLP, features: object) -> object:
    """``predict_resistance`` with the factor's diagonal floored at 1e-4 instead of 1e-5: the planted evaluator's."""
    dtype = jnp.asarray(features).dtype
    layers = tuple((jnp.asarray(w, dtype=dtype), jnp.asarray(b, dtype=dtype)) for w, b in network.layers)
    normalised = (jnp.asarray(features) - network.feature_mean) / network.feature_std
    entries = _cholesky_mlp.mlp_forward(layers, normalised) * network.entry_std + network.entry_mean
    lifted = jnp.where(jnp.asarray(_cholesky_mlp._ON_DIAGONAL), jnp.logaddexp(entries, 0.0) + 1e-4, entries)
    factor = jnp.where(jnp.asarray(_cholesky_mlp._IN_LOWER_TRIANGLE), lifted[_cholesky_mlp._ENTRY_AT], 0.0)
    return factor @ factor.T


PLANTED = Evaluator("planted_floor", "0.0.1", read_cholesky_mlp, planted_resistance)
if PLANTED.label() not in evaluators():
    register_evaluator(PLANTED)


def node(epsilon: float | None) -> SurrogateDragNode:
    return SurrogateDragNode("drag", 5e-4, weights="dejongh_mlp_v2", weights_sha256=asset("dejongh_mlp_v2").sha256,
                             viscosity=MU, helix_wavenumber=NU, length=LENGTH, radius=RADIUS,
                             outer_radius=1.33 * RADIUS, vessel_shape="cylinder", vessel_centre=(0.0, 0.0, 0.0),
                             vessel_axis=(0.0, 0.0, 1.0), vessel_radius=TUBE, lubrication_epsilon=epsilon,
                             reference_point="end")


def resistances(held: SurrogateDragNode) -> np.ndarray:
    return np.array([np.asarray(held.update(held.initial_state(), {"pose": jnp.concatenate(
        [jnp.asarray(p), jnp.asarray(q)])}, 5e-4)["resistance"]) for p, q in zip(PORT["position"], PORT["orientation"],
                                                                               strict=True)])


def blockwise(found: np.ndarray, expected: np.ndarray) -> float:
    worst = 0.0
    for rows in (slice(0, 3), slice(3, 6)):
        for columns in (slice(0, 3), slice(3, 6)):
            scale = float(np.max(np.abs(expected[:, rows, columns])))
            worst = max(worst, float(np.max(np.abs(found[:, rows, columns] - expected[:, rows, columns]))) / scale)
    return worst


@pytest.mark.parametrize("lubricated", [False, True], ids=["the surrogate", "with lubrication"])
def test_the_type_reproduces_the_4c1_mlps_golden_outputs(lubricated: bool):
    held = node(EPSILON if lubricated else None)
    assert held._evaluator is evaluator_named(DEJONGH_MLP_V2.evaluator) is CHOLESKY_MLP
    golden = PORT["lub_resistance"] if lubricated else PORT["resistance"]
    worst = blockwise(resistances(held)[OFF_AXIS], golden[OFF_AXIS])
    print(f"\nworst blockwise difference from the golden outputs {worst:.2e}")
    assert worst <= RTOL


def test_control_a_record_naming_another_evaluator_misses_the_golden_outputs(monkeypatch: pytest.MonkeyPatch):
    planted = replace(DEJONGH_MLP_V2, evaluator=PLANTED.label())
    monkeypatch.setattr(drag_module, "SURROGATES", {"dejongh_mlp_v2": planted})
    held = node(None)
    assert held._evaluator is PLANTED
    worst = blockwise(resistances(held)[OFF_AXIS], PORT["resistance"][OFF_AXIS])
    print(f"\nthe planted evaluator misses by {worst:.2e}")
    assert worst > 1e3 * RTOL


def test_the_planted_evaluator_with_the_packaged_floor_is_the_cholesky_mlp():
    """The control's evaluator differs from the packaged one only in its floor: at 1e-5 it agrees to rounding."""
    network = read_cholesky_mlp(asset_bytes_v2())
    features = jnp.asarray((2.33, 4.79, 0.49, 0.6, 0.0, -1.0, 2.33 ** 2, 0.49 ** 2))
    packaged = np.asarray(predict_resistance(network, features))
    planted = np.asarray(planted_resistance(network, features))
    assert 0.0 < np.max(np.abs(planted - packaged)) / np.max(np.abs(packaged)) < 1e-2


def asset_bytes_v2() -> bytes:
    from mime.data import asset_bytes

    return asset_bytes(asset("dejongh_mlp_v2"))


def test_the_records_accuracy_is_per_use_and_an_unmeasured_use_is_none():
    swim = DEJONGH_MLP_V2.accuracy_for("swim_speed")
    entries = DEJONGH_MLP_V2.accuracy_for("resistance")
    assert swim is not None and swim.value == 0.162 and swim.anomaly == "MIME-ANO-114"
    assert "not against the flow" in swim.claim
    assert entries is not None and entries.value is None and "unknown, not bounded" in entries.claim
    assert DEJONGH_MLP_V2.accuracy_for("flow_load") is None


def test_the_records_provenance_is_the_lateral_surface_about_the_end():
    held = DEJONGH_MLP_V2.provenance
    assert held.patches == ("lateral",) and held.reference_point == "end" and held.derivation == "dejongh_mlp_v2"
    assert DEJONGH_MLP_V2.outputs == ("resistance",) and DEJONGH_MLP_V2.features == SCREW_FEATURES


def record(**changes: object) -> Surrogate:
    held: dict[str, object] = {"asset": "x", "evaluator": CHOLESKY_MLP.label(), "features": SCREW_FEATURES,
                               "accuracy": (), "provenance": Provenance("x", ("lateral",), "end", "x")}
    held.update(changes)
    return Surrogate(**held)  # type: ignore[arg-type]


@pytest.mark.parametrize("changes", [{"outputs": ()}, {"outputs": ("flow_load",)}, {"outputs": ("resistance", "drag")},
                                     {"provenance": Provenance("x", ("lateral",), "middle", "x")},
                                     {"accuracy": (DEJONGH_MLP_V2.accuracy[0], DEJONGH_MLP_V2.accuracy[0])}])
def test_a_record_breaking_the_types_rules_is_refused(changes: dict[str, object]):
    with pytest.raises(ValueError):
        record(**changes)


def test_a_flow_load_needs_an_evaluator_that_computes_one_and_an_evaluator_must_be_registered():
    with pytest.raises(ValueError, match="flow load"):
        record(outputs=("resistance", "flow_load")).resolve()
    with pytest.raises(KeyError, match="cholesky_mlp 0.1.0"):
        record(evaluator="nothing 0.0.0").resolve()


def test_a_record_whose_features_the_node_does_not_build_is_refused(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(drag_module, "SURROGATES",
                        {"dejongh_mlp_v2": replace(DEJONGH_MLP_V2, features=SCREW_FEATURES[:6])})
    with pytest.raises(ValueError, match="features"):
        node(None)
