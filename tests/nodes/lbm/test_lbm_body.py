"""Tests for the IBLBMFluidNode geometry hook (mime.nodes.environment.lbm.body).

Covers the three things the geometry refactor had to deliver:

1. the ``LatticeBodyFactory`` hook itself (sphere + UMR geometries),
2. a body that **translates** (``body_position`` consumed, jit-traceable),
3. **no regression**: the legacy ``body_geometry_params`` UMR path is
   byte-identical to the explicit factory, and an unwired ``body_position``
   leaves the body pinned where the pre-hook node froze it.

The heavier UMR regression guards (5-step manual-path equivalence, the 64^3
Bouzidi drag-torque baseline) live in ``test_iblbm_fluid_node.py``; the
sphere-drag physics benchmark is ``tests/verification/test_lbm_sphere_drag.py``.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.environment.lbm.body import (
    LatticeBody,
    LatticeGrid,
    make_sphere_body_factory,
    make_umr_body_factory,
    rigid_body_velocity_field,
)
from mime.nodes.environment.lbm.fluid_node import (
    IBLBMFluidNode,
    make_iblbm_rigid_body_edges,
)

UMR_GEOM_MM = dict(
    body_radius=0.87, body_length=4.1, cone_length=1.9,
    cone_end_radius=0.255, fin_outer_radius=1.42,
    fin_length=2.03, fin_width=0.55, fin_thickness=0.15,
    helix_pitch=8.0,
)
VESSEL_DIAMETER_MM = 9.4


def _geom_lu(N):
    dx = VESSEL_DIAMETER_MM / N
    return {k: v / dx for k, v in UMR_GEOM_MM.items()}


def _pose(position, velocity=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0),
          angle=0.0):
    return {
        "position": jnp.asarray(position, dtype=jnp.float32),
        "velocity": jnp.asarray(velocity, dtype=jnp.float32),
        "angular_velocity": jnp.asarray(omega, dtype=jnp.float32),
        "orientation": jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        "angle": jnp.asarray(angle, dtype=jnp.float32),
    }


def _sphere_node(N=24, R_sph=4.0, R_pipe=10.0, use_bouzidi=False, **kw):
    return IBLBMFluidNode(
        name="lbm", timestep=1.0, nx=N, ny=N, nz=N, tau=0.8,
        vessel_radius_lu=R_pipe,
        body_factory=make_sphere_body_factory("sphere", radius=R_sph),
        use_bouzidi=use_bouzidi,
        **kw,
    )


def _umr_node_legacy(N=24, ratio=0.30, use_bouzidi=False):
    geom = _geom_lu(N)
    R_vessel = geom["body_radius"] / ratio
    return IBLBMFluidNode(
        name="lbm", timestep=1.0, nx=N, ny=N, nz=N, tau=0.8,
        vessel_radius_lu=R_vessel,
        body_geometry_params=dict(nx=N, ny=N, nz=N, **geom),
        use_bouzidi=use_bouzidi,
    )


def _umr_node_factory(N=24, ratio=0.30, use_bouzidi=False):
    geom = _geom_lu(N)
    R_vessel = geom["body_radius"] / ratio
    return IBLBMFluidNode(
        name="lbm", timestep=1.0, nx=N, ny=N, nz=N, tau=0.8,
        vessel_radius_lu=R_vessel,
        body_factory=make_umr_body_factory(**geom),
        use_bouzidi=use_bouzidi,
    )


# -- The sphere geometry -----------------------------------------------------

class TestSphereBody:
    def test_mask_is_a_sphere_of_the_right_volume(self):
        f = make_sphere_body_factory(radius=6.0)
        body = f(_pose((16.0, 16.0, 16.0)), LatticeGrid((32, 32, 32)))

        assert body.mask.shape == (32, 32, 32)
        assert body.mask.dtype == jnp.bool_
        n_solid = int(jnp.sum(body.mask))
        exact = 4.0 / 3.0 * math.pi * 6.0 ** 3
        # Voxelisation of a R=6 sphere: a few % of staircase error.
        assert abs(n_solid - exact) / exact < 0.05, (
            f"voxel count {n_solid} vs exact volume {exact:.1f}")

    def test_mask_and_sdf_describe_the_same_surface(self):
        """create_sphere_mask (dist < R) and sphere_sdf (dist - R < 0) must
        agree — the simple-BB and Bouzidi paths would disagree otherwise."""
        f = make_sphere_body_factory(radius=5.0)
        grid = LatticeGrid((20, 20, 20))
        body = f(_pose((10.0, 10.0, 10.0)), grid)

        ix = jnp.arange(20, dtype=jnp.float32)
        gx, gy, gz = jnp.meshgrid(ix, ix, ix, indexing="ij")
        pts = jnp.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
        sdf_inside = (body.sdf(pts) < 0).reshape(20, 20, 20)

        assert bool(jnp.all(sdf_inside == body.mask))

    def test_mask_tracks_body_position(self):
        f = make_sphere_body_factory(radius=4.0)
        grid = LatticeGrid((32, 32, 32))
        a = f(_pose((16.0, 16.0, 10.0)), grid).mask
        b = f(_pose((16.0, 16.0, 18.0)), grid).mask

        # Same body, shifted 8 cells along z: the mask must shift with it.
        assert not bool(jnp.all(a == b))
        assert int(jnp.sum(a)) == int(jnp.sum(b))
        assert bool(jnp.all(a[:, :, 2:22] == b[:, :, 10:30]))

    def test_full_angular_velocity_passes_through(self):
        """A sphere is rotationally symmetric, so (unlike the UMR) it takes
        the whole omega vector, not just omega_z."""
        f = make_sphere_body_factory(radius=4.0)
        body = f(_pose((16.0,) * 3, omega=(1.0, 2.0, 3.0)),
                 LatticeGrid((32, 32, 32)))
        np.testing.assert_allclose(body.angular_velocity, [1.0, 2.0, 3.0])

    def test_grid_origin_shifts_the_mask(self):
        """origin lets a sharded slab build its piece of the global mask."""
        f = make_sphere_body_factory(radius=4.0)
        full = f(_pose((16.0, 16.0, 16.0)), LatticeGrid((32, 32, 32))).mask
        slab = f(_pose((16.0, 16.0, 16.0)),
                 LatticeGrid((32, 32, 8), origin=(0.0, 0.0, 12.0))).mask
        assert bool(jnp.all(slab == full[:, :, 12:20]))

    def test_factory_is_jit_traceable_in_pose(self):
        f = make_sphere_body_factory(radius=4.0)
        grid = LatticeGrid((24, 24, 24))

        @jax.jit
        def n_solid(pos):
            return jnp.sum(f(_pose(pos), grid).mask)

        # No Python branching on the traced pose.
        assert int(n_solid(jnp.array([12.0, 12.0, 12.0]))) > 0
        assert int(n_solid(jnp.array([12.0, 12.0, 14.5]))) > 0


# -- The UMR geometry --------------------------------------------------------

class TestUMRBody:
    def test_angular_velocity_is_projected_onto_z(self):
        """create_umr_mask is parameterised by a z-angle only, so the body
        must drop the x/y spin rather than impose a wall velocity its mask's
        motion does not reproduce."""
        f = make_umr_body_factory(**_geom_lu(24))
        body = f(_pose((12.0,) * 3, omega=(1.0, 2.0, 3.0)),
                 LatticeGrid((24, 24, 24)))
        np.testing.assert_allclose(body.angular_velocity, [0.0, 0.0, 3.0])

    def test_mask_matches_create_umr_mask(self):
        """The factory is a thin wrapper — it must not perturb the validated
        mask (in particular it must NOT threshold umr_sdf, which gives a
        different staircase)."""
        from mime.nodes.robot.helix_geometry import create_umr_mask

        geom = _geom_lu(24)
        f = make_umr_body_factory(**geom)
        body = f(_pose((12.0, 12.0, 12.0), angle=0.7),
                 LatticeGrid((24, 24, 24)))
        ref = create_umr_mask(
            nx=24, ny=24, nz=24, center=(12.0, 12.0, 12.0),
            rotation_angle=0.7, **geom,
        )
        assert bool(jnp.all(body.mask == ref))

    def test_rejects_grid_and_pose_kwargs(self):
        for bad in ({"nx": 24}, {"center": (1, 2, 3)}, {"rotation_angle": 0.5}):
            with pytest.raises(ValueError, match="supplied per-step"):
                make_umr_body_factory(**_geom_lu(24), **bad)


# -- rigid_body_velocity_field ----------------------------------------------

class TestRigidBodyVelocityField:
    def test_matches_v_plus_omega_cross_r(self):
        shape = (6, 7, 8)
        ref = jnp.array([3.0, 3.5, 4.0])
        v = jnp.array([0.1, -0.2, 0.3])
        w = jnp.array([0.01, 0.02, -0.03])
        got = rigid_body_velocity_field(shape, ref, v, w)

        ix, iy, iz = (jnp.arange(n, dtype=jnp.float32) for n in shape)
        gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing="ij")
        r = jnp.stack([gx - ref[0], gy - ref[1], gz - ref[2]], axis=-1)
        expect = jnp.cross(jnp.broadcast_to(w, r.shape), r) + v
        np.testing.assert_allclose(got, expect, atol=1e-6)

    def test_reduces_to_rotation_only_field(self):
        """v = 0 must reproduce the legacy `_rotation_velocity_field` exactly —
        this is what keeps the umr_confinement wall velocity unchanged."""
        from mime.nodes.environment.lbm.rotating_body import (
            _rotation_velocity_field,
        )
        shape = (10, 10, 10)
        centre = (5.0, 5.0, 5.0)
        omega_z = 0.013
        legacy = _rotation_velocity_field(shape, omega_z, (0, 0, 1), centre)
        new = rigid_body_velocity_field(
            shape, jnp.asarray(centre), jnp.zeros(3),
            jnp.array([0.0, 0.0, omega_z]),
        )
        # Bit-identical, not merely close.
        assert bool(jnp.all(legacy == new))


# -- Node: the hook, and no regression on the legacy path --------------------

class TestNodeGeometryHook:
    def test_legacy_params_equal_explicit_umr_factory(self):
        """body_geometry_params is exactly make_umr_body_factory — the UMR is
        now just one geometry behind the hook."""
        bi = {"body_angular_velocity": jnp.array([0.0, 0.0, 0.01])}
        legacy, factory = _umr_node_legacy(), _umr_node_factory()
        s1, s2 = legacy.initial_state(), factory.initial_state()
        for _ in range(3):
            s1 = legacy.update(s1, bi, 1.0)
            s2 = factory.update(s2, bi, 1.0)
        assert bool(jnp.all(s1["f"] == s2["f"]))
        assert bool(jnp.all(s1["drag_force"] == s2["drag_force"]))
        assert bool(jnp.all(s1["drag_torque"] == s2["drag_torque"]))

    def test_unwired_body_position_pins_body_at_reference(self):
        """REGRESSION: a graph that does not wire body_position must behave
        exactly as the pre-hook node, which froze the body at the grid centre.
        The declared default must therefore be the reference position, not the
        origin."""
        node = _umr_node_legacy(N=24)
        spec = node.boundary_input_spec()
        np.testing.assert_allclose(spec["body_position"].default, [12.0] * 3)

        bi_unwired = {"body_angular_velocity": jnp.array([0.0, 0.0, 0.01])}
        bi_centred = dict(bi_unwired, body_position=jnp.array([12.0] * 3))
        s1 = s2 = node.initial_state()
        for _ in range(3):
            s1 = node.update(s1, bi_unwired, 1.0)
            s2 = node.update(s2, bi_centred, 1.0)
        assert bool(jnp.all(s1["f"] == s2["f"]))

    def test_requires_exactly_one_geometry_source(self):
        common = dict(name="lbm", timestep=1.0, nx=16, ny=16, nz=16,
                      tau=0.8, vessel_radius_lu=6.0)
        with pytest.raises(ValueError, match="exactly one"):
            IBLBMFluidNode(**common)
        with pytest.raises(ValueError, match="exactly one"):
            IBLBMFluidNode(
                **common,
                body_geometry_params=dict(**_geom_lu(16)),
                body_factory=make_sphere_body_factory(radius=3.0),
            )

    def test_bouzidi_without_sdf_raises(self):
        def sdf_less(pose, grid):
            return LatticeBody(
                name="blob",
                mask=jnp.zeros(grid.shape, dtype=bool),
                ref_point=pose["position"],
                sdf=None,
            )
        node = IBLBMFluidNode(
            name="lbm", timestep=1.0, nx=16, ny=16, nz=16, tau=0.8,
            vessel_radius_lu=6.0, body_factory=sdf_less, use_bouzidi=False,
        )
        # Only the Bouzidi path needs an SDF; simple BB is fine.
        node.update(node.initial_state(), {}, 1.0)

        with pytest.raises(ValueError, match="no SDF"):
            IBLBMFluidNode(
                name="lbm2", timestep=1.0, nx=16, ny=16, nz=16, tau=0.8,
                vessel_radius_lu=6.0, body_factory=sdf_less, use_bouzidi=True,
            )

    def test_count_boundary_links_sizes_the_bouzidi_buffer(self):
        node = _sphere_node(N=24, R_sph=4.0, use_bouzidi=True,
                            boundary_links_margin=2.0)
        n_ref = node.count_boundary_links()
        assert n_ref > 0
        assert node._max_boundary_links_per_dir == int(n_ref * 2.0) + 1
        # A translating body's link count varies with sub-cell position —
        # that variation is exactly why the margin exists.
        offset = node.count_boundary_links(position_lu=(12.5, 12.5, 12.37))
        assert offset > 0


class TestNodeTranslation:
    def test_body_position_moves_the_body(self):
        node = _sphere_node(N=32, R_sph=4.0, R_pipe=14.0)
        state = node.initial_state()
        bi = {
            "body_position": jnp.array([16.0, 16.0, 10.0]),
            "body_velocity": jnp.array([0.0, 0.0, 0.005]),
        }
        moved = node.update(state, bi, 1.0)
        held = node.update(state, {"body_velocity": bi["body_velocity"]}, 1.0)
        # Same twist, different place => different flow field.
        assert not bool(jnp.allclose(moved["f"], held["f"]))

    def test_translating_sphere_drag_opposes_motion(self):
        """The node reports the *reaction* (+R*U, force on the FLUID) — the
        documented IBLBM convention that HydrodynamicModel.LBM flips via
        native_drag_sign=-1 (FLUID_NODE_CONTRACT.md). So a sphere pushed in
        +z must push the fluid in +z."""
        node = _sphere_node(N=32, R_sph=4.0, R_pipe=14.0)
        state = node.initial_state()
        U = 0.004
        for i in range(60):
            bi = {
                "body_position": jnp.array([16.0, 16.0, 16.0 + U * i]),
                "body_velocity": jnp.array([0.0, 0.0, U]),
            }
            state = node.update(state, bi, 1.0)

        F = np.asarray(state["drag_force"])
        assert F[2] > 0.0, f"reaction on fluid should be +z, got {F}"
        # Axisymmetric body on the tube axis: no transverse load.
        assert abs(F[0]) < 0.02 * abs(F[2])
        assert abs(F[1]) < 0.02 * abs(F[2])

    def test_drag_is_invariant_along_the_tube_axis(self):
        """The pipe wall is z-invariant, so translating the sphere by a whole
        number of cells along z must reproduce the same drag. This pins the
        translation machinery (mask origin, lever arm, wall velocity) against
        an off-by-one in the pose plumbing."""
        node = _sphere_node(N=32, R_sph=4.0, R_pipe=14.0)
        U = 0.004
        finals = []
        for z0 in (16.0, 20.0):
            state = node.initial_state()
            for _ in range(40):
                bi = {
                    "body_position": jnp.array([16.0, 16.0, z0]),
                    "body_velocity": jnp.array([0.0, 0.0, U]),
                }
                state = node.update(state, bi, 1.0)
            finals.append(np.asarray(state["drag_force"]))
        np.testing.assert_allclose(finals[0], finals[1], atol=1e-6)

    def test_update_is_jit_traceable_with_a_traced_pose(self):
        """The mask now depends on a traced pose — update() must stay
        jit-traceable (no Python branching / .item() on traced values)."""
        node = _sphere_node(N=24, R_sph=3.0, R_pipe=10.0)
        state = node.initial_state()

        step = jax.jit(lambda s, bi: node.update(s, bi, 1.0))
        bi = {
            "body_position": jnp.array([12.0, 12.0, 12.0]),
            "body_velocity": jnp.array([0.0, 0.0, 0.003]),
            "body_angular_velocity": jnp.array([0.0, 0.0, 0.001]),
        }
        out = step(state, bi)
        assert out["f"].shape == state["f"].shape
        assert jnp.all(jnp.isfinite(out["drag_force"]))

        # A different pose must reuse the trace, not retrace into a new body.
        bi2 = dict(bi, body_position=jnp.array([12.0, 12.0, 13.7]))
        out2 = step(state, bi2)
        assert not bool(jnp.allclose(out["f"], out2["f"]))

    def test_umr_body_can_translate_too(self):
        """The UMR is no longer special-cased: the same hook gives it
        translation, which the pre-hook node could not do at all."""
        node = _umr_node_legacy(N=24)
        state = node.initial_state()
        bi_a = {"body_position": jnp.array([12.0, 12.0, 10.0]),
                "body_angular_velocity": jnp.array([0.0, 0.0, 0.01])}
        bi_b = {"body_position": jnp.array([12.0, 12.0, 14.0]),
                "body_angular_velocity": jnp.array([0.0, 0.0, 0.01])}
        a = node.update(state, bi_a, 1.0)
        b = node.update(state, bi_b, 1.0)
        assert not bool(jnp.allclose(a["f"], b["f"]))


# -- Edge wiring -------------------------------------------------------------

class TestTranslationEdges:
    def test_translation_edges_are_opt_in(self):
        """Default wiring must stay 4 edges — umr_confinement calls this
        helper, and wiring position would change its validated results."""
        edges = make_iblbm_rigid_body_edges(
            "lbm", "body", dx_physical=1e-4, dt_physical=1e-6)
        assert len(edges) == 4
        assert not any(e.target_field == "body_position" for e in edges)

    def test_wire_translation_adds_position_and_velocity_edges(self):
        edges = make_iblbm_rigid_body_edges(
            "lbm", "body", dx_physical=1e-4, dt_physical=1e-6,
            wire_translation=True, position_origin_lu=(32.0, 32.0, 32.0),
        )
        assert len(edges) == 6
        pos = next(e for e in edges if e.target_field == "body_position")
        vel = next(e for e in edges if e.target_field == "body_velocity")
        assert pos.source_field == "position" and pos.source_units == "m"
        assert vel.source_field == "velocity" and vel.source_units == "m/s"

        # x_lu = x_SI/dx + origin: the SI origin maps to the grid centre.
        np.testing.assert_allclose(
            pos.transform(jnp.zeros(3)), [32.0, 32.0, 32.0], atol=1e-4)
        np.testing.assert_allclose(
            pos.transform(jnp.array([1e-4, 0.0, 0.0])),
            [33.0, 32.0, 32.0], atol=1e-4)

        # u_lu = u_SI * dt/dx = u_SI * 1e-2
        np.testing.assert_allclose(
            vel.transform(jnp.array([0.0, 0.0, 1.0])),
            [0.0, 0.0, 0.01], atol=1e-6)

    def test_velocity_edge_clamps_at_the_mach_limit(self):
        """Step-0 of FSI coupling carries zero drag, so the body can emit an
        unphysical velocity; the clamp keeps Ma < 0.1 (same rationale as the
        pre-existing omega clamp)."""
        edges = make_iblbm_rigid_body_edges(
            "lbm", "body", dx_physical=1e-4, dt_physical=1e-6,
            wire_translation=True)
        vel = next(e for e in edges if e.target_field == "body_velocity")
        out = vel.transform(jnp.array([0.0, 0.0, 1e6]))
        assert float(out[2]) == pytest.approx(0.0577, abs=1e-4)
