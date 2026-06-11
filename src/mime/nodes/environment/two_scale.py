"""Two-scale Schwarz coupling builder (C1).

Promotes the hand-wiring of the A3 de-risk slice
(``tests/verification/test_two_scale_slice_a3.py``) into a reusable helper:
couples an FVM far-field node ⊗ a Stokeslet-BEM near-field node into a
``GraphManager`` coupling group, with the A1/A2 interface transfers

    far → near :  far.velocity_at_points  →  near.background_flow
    near → far :  near.body_traction × weights (N)  →  far.forcing_values

This is the clean reusable builder the in-flux ``umr_confinement`` Schwarz path
referenced but never had (``make_schwarz_coupling_edges``). The far node works
with *any* FVM backend (Cartesian today; body-fitted once the unstructured
projection is stable — see the B3 note in ``plans/MIME_v0.3.0_PLAN.md``).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def make_two_scale_coupling(
    gm,
    far_node,
    near_node,
    *,
    body_points: jnp.ndarray,
    body_weights: jnp.ndarray,
    coupling_kwargs: dict[str, Any] | None = None,
) -> dict:
    """Wire a far-field ⊗ near-field two-scale Schwarz coupling into ``gm``.

    Parameters
    ----------
    gm : GraphManager
        Target graph. ``far_node`` / ``near_node`` are added if not present.
    far_node : FVMFluidNode
        The inertial far-field, built with
        ``n_sample_points = n_forcing_points = body_points.shape[0]`` so it
        exposes ``velocity_at_points`` / ``forcing_*`` ports (A1/A2).
    near_node : StokesletFluidNode
        The confined-Stokes near-field in **Schwarz mode**
        (``interface_mesh`` set), exposing ``background_flow`` /
        ``body_traction``.
    body_points : (N_body, 3)
        The body-surface points where the far field is sampled and the near
        field's reaction is applied (the BEM body mesh points).
    body_weights : (N_body,)
        BEM quadrature weights, converting traction (Pa) → point force (N).
    coupling_kwargs : dict, optional
        Forwarded to ``add_coupling_group``. Sensible Schwarz defaults are
        applied: ``convergence_norm="interface"``, ``acceleration="iqn-ils"``,
        ``max_iterations=8``, ``diagnostics=True``. When IQN acceleration is
        used, ``accelerated_fields`` defaults to the float coupling fields
        ``{far:("u",), near:("body_traction",)}`` and ``atol/rtol`` are relaxed
        to the field magnitudes — without this MADDENING's full-state
        flatten promotes the FVM int32 ``i_step`` and breaks the coupling carry
        (see memory ``fvm-coupling-group-int-state``).

    Returns
    -------
    dict
        ``{"far": far_node.name, "near": near_node.name,
           "geometry_inputs": {far_node.name: {"sample_points": ...,
                                                "forcing_points": ...}}}``
        — the constant geometry external inputs the caller must merge into the
        per-step ``external_inputs`` (alongside the body-kinematics drive).
    """
    far, near = far_node.name, near_node.name
    existing = set(getattr(gm, "_nodes", {}))
    if far not in existing:
        gm.add_node(far_node)
    if near not in existing:
        gm.add_node(near_node)

    # far → near: sampled fluid velocity at the body points → background flow.
    gm.add_edge(far, near, "velocity_at_points", "background_flow",
                source_units="m/s", target_units="m/s")

    # near → far: BEM traction (Pa) → reaction point forces (N) = traction·w.
    w = jnp.asarray(body_weights)

    def _traction_to_force(tr, _w=w):
        return tr * _w[:, None]

    gm.add_edge(near, far, "body_traction", "forcing_values",
                transform=_traction_to_force, source_units="Pa", target_units="N")

    # constant geometry external inputs (the sampling/forcing locations).
    nb = int(body_points.shape[0])
    for fld in ("sample_points", "forcing_points"):
        gm.add_external_input(far, fld, shape=(nb, 3), dtype=body_points.dtype)

    # coupling group with Schwarz defaults + the int-state-safe IQN settings.
    kw = dict(max_iterations=8, tolerance=1e-4,
              convergence_norm="interface", acceleration="iqn-ils",
              diagnostics=True)
    kw.update(coupling_kwargs or {})
    if kw.get("acceleration") in ("iqn-ils", "iqn-imvj"):
        kw.setdefault("accelerated_fields", {far: ("u",), near: ("body_traction",)})
        kw.setdefault("atol", 1e-7)
        kw.setdefault("rtol", 1e-3)
    gm.add_coupling_group([far, near], **kw)

    pts = jnp.asarray(body_points)
    return {
        "far": far,
        "near": near,
        "geometry_inputs": {far: {"sample_points": pts, "forcing_points": pts}},
    }
