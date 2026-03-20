"""Tests for StageBridge — USD stage management for visualisation.

Tests are split into:
1. API contract tests that use mock stages (always run)
2. USD integration tests that require usd-core (skipped if not installed)
"""

import pytest
import numpy as np

# Check if USD is available
try:
    from pxr import Usd, UsdGeom, Gf, Sdf
    _HAS_USD = True
except ImportError:
    _HAS_USD = False

requires_usd = pytest.mark.skipif(not _HAS_USD, reason="usd-core not installed")

from mime.core.geometry import CylinderGeometry, MeshGeometry


# -- Tests that always run (no USD needed) ---------------------------------

class TestStageBridgeImport:
    def test_module_importable(self):
        """The module should import even without USD installed."""
        from mime.viz import stage_bridge
        assert hasattr(stage_bridge, 'StageBridge')

    def test_missing_usd_raises_clear_error(self):
        """If USD is not installed, StageBridge() should give a clear error."""
        if _HAS_USD:
            pytest.skip("USD is installed — can't test missing-USD path")
        from mime.viz.stage_bridge import StageBridge
        with pytest.raises(ImportError, match="usd-core"):
            StageBridge()

    def test_geometry_source_types(self):
        """CylinderGeometry and MeshGeometry are importable."""
        assert CylinderGeometry(diameter_m=2e-3, length_m=10e-3)
        assert MeshGeometry(mesh_uri="test.usd")


# -- USD integration tests -------------------------------------------------

@requires_usd
class TestStageBridgeCreation:
    def test_creates_in_memory_stage(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        assert bridge.stage is not None

    def test_creates_world_scope(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        world = bridge.stage.GetPrimAtPath("/World")
        assert world.IsValid()

    def test_creates_default_camera(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        cam = bridge.stage.GetPrimAtPath("/World/Camera")
        assert cam.IsValid()
        assert cam.IsA(UsdGeom.Camera)

    def test_accepts_existing_stage(self):
        from mime.viz.stage_bridge import StageBridge
        stage = Usd.Stage.CreateInMemory()
        bridge = StageBridge(stage=stage)
        assert bridge.stage is stage

    def test_stage_up_axis(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge(up_axis="Z")
        assert UsdGeom.GetStageUpAxis(bridge.stage) == UsdGeom.Tokens.z


@requires_usd
class TestStageBridgeRobotRegistration:
    def test_register_robot_creates_prim(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        bridge.register_robot("body", prim_path="/World/Robot")
        prim = bridge.stage.GetPrimAtPath("/World/Robot")
        assert prim.IsValid()

    def test_register_robot_has_xform_ops(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        bridge.register_robot("body", prim_path="/World/Robot")
        prim = bridge.stage.GetPrimAtPath("/World/Robot")
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        op_names = [op.GetOpName() for op in ops]
        assert "xformOp:translate" in op_names
        assert "xformOp:orient" in op_names

    def test_register_robot_default_sphere(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        bridge.register_robot("body", prim_path="/World/Robot", radius=100e-6)
        prim = bridge.stage.GetPrimAtPath("/World/Robot")
        assert prim.IsA(UsdGeom.Sphere)
        sphere = UsdGeom.Sphere(prim)
        assert sphere.GetRadiusAttr().Get() == pytest.approx(100e-6)


@requires_usd
class TestStageBridgeFieldRegistration:
    def test_register_field_creates_cone(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        bridge.register_field("field", prim_path="/World/Arrow")
        prim = bridge.stage.GetPrimAtPath("/World/Arrow")
        assert prim.IsValid()
        assert prim.IsA(UsdGeom.Cone)

    def test_register_field_has_scale_op(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        bridge.register_field("field", prim_path="/World/Arrow")
        prim = bridge.stage.GetPrimAtPath("/World/Arrow")
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        op_names = [op.GetOpName() for op in ops]
        assert "xformOp:scale" in op_names


@requires_usd
class TestStageBridgeStaticGeometry:
    def test_add_parametric_cylinder(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        tube = CylinderGeometry(diameter_m=2e-3, length_m=10e-3, axis="z")
        bridge.add_parametric_geometry(tube, "/World/Tube")
        prim = bridge.stage.GetPrimAtPath("/World/Tube")
        assert prim.IsValid()
        assert prim.IsA(UsdGeom.Cylinder)
        cyl = UsdGeom.Cylinder(prim)
        assert cyl.GetRadiusAttr().Get() == pytest.approx(1e-3)
        assert cyl.GetHeightAttr().Get() == pytest.approx(10e-3)

    def test_add_reference_geometry_payload(self):
        """Default: loads as payload (deferred)."""
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        # We can add a reference to a non-existent file — it just won't resolve
        bridge.add_reference_geometry("fake_mesh.usd", "/World/Anatomy")
        prim = bridge.stage.GetPrimAtPath("/World/Anatomy")
        assert prim.IsValid()
        # Payloads are set
        assert prim.HasPayload()

    def test_add_reference_geometry_hard_ref(self):
        """as_reference=True: loads as hard reference."""
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        bridge.add_reference_geometry("fake_mesh.usd", "/World/Anatomy",
                                     as_reference=True)
        prim = bridge.stage.GetPrimAtPath("/World/Anatomy")
        assert prim.IsValid()
        assert prim.HasAuthoredReferences()


@requires_usd
class TestStageBridgeUpdate:
    def test_update_robot_position(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        bridge.register_robot("body", prim_path="/World/Robot")

        state = {
            "body": {
                "position": np.array([1e-3, 2e-3, 3e-3]),
                "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
            }
        }
        bridge.update(state)

        prim = bridge.stage.GetPrimAtPath("/World/Robot")
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        translate = ops[0].Get()
        assert translate[0] == pytest.approx(1e-3)
        assert translate[1] == pytest.approx(2e-3)
        assert translate[2] == pytest.approx(3e-3)

    def test_update_robot_orientation(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        bridge.register_robot("body", prim_path="/World/Robot")

        # 90-degree rotation around z
        import math
        q = np.array([math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)])
        state = {
            "body": {
                "position": np.zeros(3),
                "orientation": q,
            }
        }
        bridge.update(state)

        prim = bridge.stage.GetPrimAtPath("/World/Robot")
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        orient = ops[1].Get()
        assert orient.GetReal() == pytest.approx(math.cos(math.pi/4), abs=1e-5)

    def test_update_field_vector(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        bridge.register_field("field", prim_path="/World/Arrow")

        state = {
            "field": {
                "field_vector": np.array([0.01, 0.0, 0.0]),  # 10mT along x
            }
        }
        bridge.update(state)

        prim = bridge.stage.GetPrimAtPath("/World/Arrow")
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        # Scale should be non-zero (field is non-zero)
        scale = ops[2].Get()
        assert scale[0] > 0

    def test_update_zero_field_hides_arrow(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        bridge.register_field("field", prim_path="/World/Arrow")

        state = {"field": {"field_vector": np.zeros(3)}}
        bridge.update(state)

        prim = bridge.stage.GetPrimAtPath("/World/Arrow")
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        scale = ops[2].Get()
        assert scale[0] == pytest.approx(0.0)

    def test_update_missing_node_no_crash(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        bridge.register_robot("body", prim_path="/World/Robot")
        # Update with state that doesn't contain "body" — should not crash
        bridge.update({"other_node": {"x": np.array(1.0)}})

    def test_multiple_updates(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        bridge.register_robot("body", prim_path="/World/Robot")

        for i in range(10):
            state = {
                "body": {
                    "position": np.array([float(i) * 1e-4, 0.0, 0.0]),
                    "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
                }
            }
            bridge.update(state)

        prim = bridge.stage.GetPrimAtPath("/World/Robot")
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        final_pos = ops[0].Get()
        assert final_pos[0] == pytest.approx(9e-4)


@requires_usd
class TestStageBridgeObserver:
    def test_as_observer_returns_callable(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        obs = bridge.as_observer()
        assert callable(obs)

    def test_observer_calls_update(self):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        bridge.register_robot("body", prim_path="/World/Robot")
        obs = bridge.as_observer()

        true_state = {
            "body": {
                "position": np.array([5e-4, 0.0, 0.0]),
                "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
            }
        }
        obs(0.0, 0.001, true_state, {}, {}, {})

        prim = bridge.stage.GetPrimAtPath("/World/Robot")
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        assert ops[0].Get()[0] == pytest.approx(5e-4)


@requires_usd
class TestStageBridgeExport:
    def test_export_creates_file(self, tmp_path):
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        bridge.register_robot("body", prim_path="/World/Robot")
        out = str(tmp_path / "test.usda")
        bridge.export(out)
        assert (tmp_path / "test.usda").exists()
        # Verify we can re-open it
        stage = Usd.Stage.Open(out)
        assert stage.GetPrimAtPath("/World/Robot").IsValid()
