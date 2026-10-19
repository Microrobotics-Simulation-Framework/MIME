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
    extra_coupling_members: tuple[str, ...] = (),
    subtract_self_wake: bool = False,
    self_wake_mu: float | None = None,
    self_wake_epsilon: float | None = None,
    transfers: str = "hand",
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
        ``{far:("u",), near:("body_traction",)}`` (``point_force`` in place
        of ``body_traction`` with ``transfers="mapped"``) and ``atol/rtol`` are relaxed
        to the field magnitudes — without this MADDENING's full-state
        flatten promotes the FVM int32 ``i_step`` and breaks the coupling carry
        (see memory ``fvm-coupling-group-int-state``).

    transfers : {"hand", "sampling_mapped", "mapped"}
        Where the two transfers are computed. ``"hand"``: in the far node
        (trilinear sampling, Gaussian spreading). ``"sampling_mapped"``: the
        sampling on a ``multilinear_grid`` gather edge (far built with
        ``coupling_points = N_body``, without ``n_sample_points``), the
        spreading by hand. ``"mapped"``: the spreading too, on a
        ``multilinear_grid`` conservative scatter edge from the near node's
        ``point_force`` (near built with ``coupling_points=True``, far with
        ``cell_forcing=True``, without ``n_forcing_points``): a trilinear
        kernel, not the hand path's Gaussian. Both edges anchor their
        geometry at their source, whose copy of the points is the one the
        caller supplies for this step.

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
    nb = int(pts.shape[0])
    if transfers not in ("hand", "sampling_mapped", "mapped"):
        raise ValueError(f"transfers={transfers!r}")
    if transfers != "hand" and subtract_self_wake:
        raise ValueError("subtract_self_wake needs transfers='hand'")

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
    if transfers == "hand":
        gm.add_edge(far, near, "velocity_at_points", "background_flow",
                    additive=subtract_self_wake,
                    source_units="m/s", target_units="m/s")
    else:
        gm.add_edge(far, near, "u_phys", "background_flow",
                    mapping=_lattice_mapping(far_node, nb, "consistent"),
                    geometry=("source", "coupling_points"),
                    source_units="m/s", target_units="m/s")

    def _traction_to_force(tr, _w=w):
        return tr * _w[:, None]

    # near → far: BEM traction (Pa) → reaction point forces (N) = traction·w.
    if transfers == "mapped":
        gm.add_edge(near, far, "point_force", "cell_force",
                    mapping=_lattice_mapping(far_node, nb, "conservative"),
                    geometry=("source", "coupling_points"),
                    source_units="N", target_units="N")
    else:
        gm.add_edge(near, far, "body_traction", "forcing_values",
                    transform=_traction_to_force, source_units="Pa", target_units="N")

    if subtract_self_wake:  # KNOWN-DIVERGENT (see note above); opt-in for the record
        mu = self_wake_mu if self_wake_mu is not None else getattr(near_node, "_mu")
        eps = (self_wake_epsilon if self_wake_epsilon is not None
               else far_node.params["forcing_sigma"])

        def _self_wake(tr, _pts=pts, _w=w, _mu=float(mu), _eps=float(eps)):
            forces = tr * _w[:, None]
            return -regularized_stokeslet_velocity(
                _pts, _pts, forces, mu=_mu, epsilon=_eps)

        gm.add_edge(near, near, "body_traction", "background_flow",
                    transform=_self_wake, additive=True,
                    source_units="Pa", target_units="m/s")

    # constant geometry external inputs (the sampling/forcing locations).
    geometry_inputs = {
        "hand": {far: ("sample_points", "forcing_points")},
        "sampling_mapped": {far: ("coupling_points", "forcing_points")},
        "mapped": {far: ("coupling_points",), near: ("coupling_points",)},
    }[transfers]
    for node_name, fields in geometry_inputs.items():
        for fld in fields:
            gm.add_external_input(node_name, fld, shape=(nb, 3), dtype=pts.dtype)

    # coupling group with Schwarz defaults + the int-state-safe IQN settings.
    kw = dict(max_iterations=8, tolerance=1e-4,
              convergence_norm="interface", acceleration="iqn-ils",
              diagnostics=True)
    kw.update(coupling_kwargs or {})
    if kw.get("acceleration") in ("iqn-ils", "iqn-imvj"):
        # The scope must hold the field the near->far edge reads: with the spreading on a
        # mapping that is point_force, and IQN scoped to body_traction then accelerates a field
        # no edge reads, leaving the loop a plain Gauss-Seidel sweep, which diverges for a
        # narrow (trilinear) spreading kernel.
        near_field = "point_force" if transfers == "mapped" else "body_traction"
        kw.setdefault("accelerated_fields", {far: ("u",), near: (near_field,)})
        kw.setdefault("atol", 1e-7)
        kw.setdefault("rtol", 1e-3)
    # Extra members (e.g. the swimmer body + magnetic chain) join the same implicit
    # group so an OVERDAMPED body's force balance (and the magnetic-orientation
    # feedback) are resolved self-consistently with the Schwarz iteration each step —
    # not staggered (which step-0-blows-up the overdamped body and lags the magnetic
    # torque into negative damping).
    gm.add_coupling_group([far, near, *extra_coupling_members], **kw)

    return {
        "far": far,
        "near": near,
        "geometry_inputs": {n: {f: pts for f in fs} for n, fs in geometry_inputs.items()},
    }


def _lattice_mapping(far_node, n_points: int, mode: str):
    """``multilinear_grid`` between the far node's cell centres and ``n_points`` points.

    The lattice's origin is the first cell centre and its shape the number of cells; the
    field is the flat C-order cell array, which is the Cartesian mesh's cell order.
    """
    from maddening.core.coupling.grid_mapping import multilinear_grid_mapping

    mesh = far_node.mesh
    if mesh.cartesian_shape is None:
        raise ValueError("a mapped two-scale transfer needs a Cartesian far mesh")
    origin = tuple(o + 0.5 * h for o, h in zip(mesh.cartesian_origin, mesh.cartesian_spacing))
    return multilinear_grid_mapping(origin, mesh.cartesian_spacing, mesh.cartesian_shape,
                                    n_points=n_points, mode=mode, layout="flat")
