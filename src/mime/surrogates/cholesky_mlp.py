"""Cholesky-parameterised MLP surrogate for SPD matrix prediction.

Predicts a 6×6 symmetric positive-definite (SPD) matrix by outputting
the 21 lower-triangular entries of its Cholesky factor L. The diagonal
of L is passed through softplus so R = L Lᵀ is SPD by construction.

Core primitives (model architecture only; training is in
``scripts/retrain_mlp_v2.py``):

    :func:`mlp_forward`        — stateless MLP forward pass
    :func:`L_flat_to_R_jax`    — Cholesky reconstruction, softplus diag
    :func:`R_to_L_flat_numpy`  — inverse mapping for training targets
    :func:`load_weights`       — load .npz weights + normalization stats

Used by :class:`mime.nodes.environment.stokeslet.mlp_resistance_node.MLPResistanceNode`
at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import jax
import jax.numpy as jnp


# ── Cholesky <-> R conversion ────────────────────────────────────────

def R_to_L_flat_numpy(R: np.ndarray) -> np.ndarray:
    """R (SPD) → L_flat (21,) with log-scaled diagonal.

    Used to prepare training targets. The diagonal of L goes through
    an inverse softplus: x = log(expm1(L_ii)), so that at inference
    the predictor output x is mapped back via softplus(x) → L_ii > 0.
    """
    L = np.linalg.cholesky(R)
    flat = np.zeros(21)
    idx = 0
    for i in range(6):
        for j in range(i + 1):
            val = L[i, j]
            if i == j:
                flat[idx] = np.log(np.expm1(max(val, 1e-10)))
            else:
                flat[idx] = val
            idx += 1
    return flat


def L_flat_to_R_jax(flat):
    """21 raw Cholesky entries → 6×6 SPD R = LLᵀ. Softplus on diagonal.

    SPD is guaranteed by construction — the diagonal of L after softplus
    is strictly positive, and LLᵀ is always positive semi-definite.
    """
    L = jnp.zeros((6, 6))
    idx = 0
    for i in range(6):
        for j in range(i + 1):
            val = flat[idx]
            if i == j:
                val = jax.nn.softplus(val) + 1e-5
            L = L.at[i, j].set(val)
            idx += 1
    return L @ L.T


L_flat_to_R_vmap = jax.vmap(L_flat_to_R_jax)


# ── MLP model ────────────────────────────────────────────────────────

def mlp_forward(params_list, x):
    """Stateless forward pass. ``params_list`` is [(W, b), ...] per layer.

    All hidden layers use SiLU activation. Last layer is linear (21 outputs).
    """
    for w, b in params_list[:-1]:
        x = jax.nn.silu(jnp.dot(x, w) + b)
    w, b = params_list[-1]
    return jnp.dot(x, w) + b


def mlp_init(key, layers, in_dim: int, out_dim: int = 21):
    """He-initialised weights for SiLU activations."""
    from jax import random
    params = []
    dims = [in_dim] + list(layers) + [out_dim]
    for i in range(len(dims) - 1):
        key, sub = random.split(key)
        w = random.normal(sub, (dims[i], dims[i + 1])) * np.sqrt(2.0 / dims[i])
        b = jnp.zeros(dims[i + 1])
        params.append((w, b))
    return params


# ── Weights loader ───────────────────────────────────────────────────

@dataclass
class CholeskyMLPWeights:
    """Bundle of trained weights + normalization stats for inference."""
    layers: List[Tuple[jnp.ndarray, jnp.ndarray]]  # (W, b) per layer
    X_mean: jnp.ndarray
    X_std: jnp.ndarray
    L_mean: jnp.ndarray
    L_std: jnp.ndarray
    layer_sizes: Tuple[int, ...]
    use_squared_features: bool
    R_cyl_UMR: float


def load_weights(path) -> CholeskyMLPWeights:
    """Load CholeskyMLP weights + normalization from .npz file.

    The .npz is produced by ``scripts/retrain_mlp_v2.py``. Fields:
        n_layers, layer_sizes, W0..Wn, b0..bn,
        X_mean, X_std, L_mean, L_std,
        R_CYL_UMR, use_squared_features
    """
    d = np.load(path)
    n_layers = int(d["n_layers"])
    layers = [(jnp.asarray(d[f"W{i}"]), jnp.asarray(d[f"b{i}"]))
              for i in range(n_layers)]
    layer_sizes_arr = d["layer_sizes"]
    layer_sizes = tuple(int(s) for s in np.atleast_1d(layer_sizes_arr))
    use_sq = bool(d["use_squared_features"]) if "use_squared_features" in d.files else True
    return CholeskyMLPWeights(
        layers=layers,
        X_mean=jnp.asarray(d["X_mean"]),
        X_std=jnp.asarray(d["X_std"]),
        L_mean=jnp.asarray(d["L_mean"]),
        L_std=jnp.asarray(d["L_std"]),
        layer_sizes=layer_sizes,
        use_squared_features=use_sq,
        R_cyl_UMR=float(d["R_CYL_UMR"]),
    )


def predict_R_from_features(weights: CholeskyMLPWeights, X_raw: jnp.ndarray) -> jnp.ndarray:
    """Full inference pipeline: raw features → R (SPD).

    Normalize → MLP → denormalize L → reconstruct R = LLᵀ.
    """
    X_n = (X_raw - weights.X_mean) / weights.X_std
    L_flat_n = mlp_forward(weights.layers, X_n)
    L_flat = L_flat_n * weights.L_std + weights.L_mean
    return L_flat_to_R_jax(L_flat)
