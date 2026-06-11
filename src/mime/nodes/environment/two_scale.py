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

from mime.nodes.environment.fvm.integrator import regularized_stokeslet_velocity


def make_two_scale_coupling(
    gm,
    far_node,
    near_node,
    *,
    body_points: jnp.ndarray,
    body_weights: jnp.ndarray,
    coupling_kwargs: dict[str, Any] | None = None,
    subtract_self_wake: bool = False,
    self_wake_mu: float | None = None,
    self_wake_epsilon: float | None = None,
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

    w = jnp.asarray(body_weights)
    pts = jnp.asarray(body_points)

    # far → near: sampled fluid velocity at the body points → background flow.
    # NOTE: the raw FVM sample includes the swimmer's OWN wake (the FVM builds it
    # from the near-field reaction forcing), so the confined BEM sees its own
    # disturbance as "background" — a self-wake overlap that spuriously reduces
    # the drag (zero-flow asymptote ~0.4× the BEM-alone confined value; see plan
    # "E-phase step 5"). ``subtract_self_wake`` was an attempt to remove it by
    # subtracting the near-field's free-space regularized-Stokeslet self-velocity
    # — but it **DIVERGES** (a KNOWN BLOCKER, 2026-06-11): the analytic Stokeslet
    # is instantaneous-steady while the FVM self-wake is *transient* (develops
    # over many timesteps), so per-step subtraction over-subtracts ~90× early and
    # the coupling blows up exponentially. The proper fix needs the FVM's actual
    # transient self-wake (a far-field quantity, second FVM solve), not a
    # free-space near-field Stokeslet. **Left OFF by default; do not enable until
    # the transient-matched subtraction is implemented.** Until then the coupled
    # path is used DIFFERENTIALLY (the clean linear counter-flow response).
    gm.add_edge(far, near, "velocity_at_points", "background_flow",
                additive=subtract_self_wake,
                source_units="m/s", target_units="m/s")

    def _traction_to_force(tr, _w=w):
        return tr * _w[:, None]

    # near → far: BEM traction (Pa) → reaction point forces (N) = traction·w.
    gm.add_edge(near, far, "body_traction", "forcing_values",
                transform=_traction_to_force, source_units="Pa", target_units="N")

    if subtract_self_wake:  # KNOWN-DIVERGENT (see note above); opt-in for the record
        mu = self_wake_mu if self_wake_mu is not None else getattr(near_node, "_mu")
        eps = (self_wake_epsilon if self_wake_epsilon is not None
               else getattr(far_node, "_forcing_sigma"))

        def _self_wake(tr, _pts=pts, _w=w, _mu=float(mu), _eps=float(eps)):
            forces = tr * _w[:, None]
            return -regularized_stokeslet_velocity(
                _pts, _pts, forces, mu=_mu, epsilon=_eps)

        gm.add_edge(near, near, "body_traction", "background_flow",
                    transform=_self_wake, additive=True,
                    source_units="Pa", target_units="m/s")

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

    return {
        "far": far,
        "near": near,
        "geometry_inputs": {far: {"sample_points": pts, "forcing_points": pts}},
    }
