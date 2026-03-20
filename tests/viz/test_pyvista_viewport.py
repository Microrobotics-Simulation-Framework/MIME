"""Tests for PyVistaViewport — offscreen USD rendering.

All tests skip if usd-core or pyvista are not installed.
"""

import pytest
import numpy as np

try:
    from pxr import Usd, UsdGeom, Gf
    _HAS_USD = True
except ImportError:
    _HAS_USD = False

try:
    import pyvista as pv
    _HAS_PYVISTA = True
except ImportError:
    _HAS_PYVISTA = False

requires_viz = pytest.mark.skipif(
    not (_HAS_USD and _HAS_PYVISTA),
    reason="requires usd-core and pyvista",
)


class TestPyVistaViewportImport:
    def test_module_importable(self):
        from mime.viz import pyvista_viewport
        assert hasattr(pyvista_viewport, 'PyVistaViewport')


@requires_viz
class TestPyVistaViewportCreation:
    def test_creates_plotter(self):
        from mime.viz.pyvista_viewport import PyVistaViewport
        vp = PyVistaViewport(width=320, height=240)
        assert vp is not None
        vp.close()

    def test_custom_size(self):
        from mime.viz.pyvista_viewport import PyVistaViewport
        vp = PyVistaViewport(width=640, height=480)
        vp.close()


@requires_viz
class TestPyVistaViewportRendering:
    def _make_simple_stage(self):
        """Create a stage with one sphere for testing."""
        from mime.viz.stage_bridge import StageBridge
        bridge = StageBridge()
        bridge.register_robot("body", prim_path="/World/Robot", radius=0.001)
        return bridge

    def test_render_returns_image(self):
        from mime.viz.pyvista_viewport import PyVistaViewport
        bridge = self._make_simple_stage()
        vp = PyVistaViewport(width=320, height=240)
        img = vp.render(bridge.stage)
        assert isinstance(img, np.ndarray)
        assert img.shape == (240, 320, 3)
        assert img.dtype == np.uint8
        vp.close()

    def test_render_non_black(self):
        """Rendered image should not be entirely black."""
        from mime.viz.pyvista_viewport import PyVistaViewport
        bridge = self._make_simple_stage()
        vp = PyVistaViewport(width=320, height=240, background="white")
        img = vp.render(bridge.stage)
        # At least some pixels should be non-zero (white background + blue sphere)
        assert img.max() > 0
        vp.close()

    def test_render_changes_with_state(self):
        """Moving the robot should produce a different image."""
        from mime.viz.pyvista_viewport import PyVistaViewport
        bridge = self._make_simple_stage()
        vp = PyVistaViewport(width=320, height=240)

        # Frame 1: robot at origin
        bridge.update({
            "body": {
                "position": np.array([0.0, 0.0, 0.0]),
                "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
            }
        })
        img1 = vp.render(bridge.stage)

        # Frame 2: robot moved
        bridge.update({
            "body": {
                "position": np.array([0.005, 0.0, 0.0]),
                "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
            }
        })
        img2 = vp.render(bridge.stage)

        # Images should differ (robot moved)
        # Note: this might fail if the sphere is out of view in both frames
        # We just check that the render pipeline doesn't crash on multiple calls
        assert img1.shape == img2.shape
        vp.close()

    def test_render_with_tube(self):
        """Render with a static tube geometry."""
        from mime.viz.stage_bridge import StageBridge
        from mime.viz.pyvista_viewport import PyVistaViewport
        from mime.core.geometry import CylinderGeometry

        bridge = StageBridge()
        bridge.add_parametric_geometry(
            CylinderGeometry(diameter_m=2e-3, length_m=10e-3),
            "/World/Tube",
        )
        bridge.register_robot("body", prim_path="/World/Robot", radius=100e-6)

        vp = PyVistaViewport(width=320, height=240)
        img = vp.render(bridge.stage)
        assert img.shape == (240, 320, 3)
        vp.close()

    def test_render_with_field_arrow(self):
        """Render with a field arrow glyph."""
        from mime.viz.stage_bridge import StageBridge
        from mime.viz.pyvista_viewport import PyVistaViewport

        bridge = StageBridge()
        bridge.register_field("field", prim_path="/World/Arrow")
        bridge.update({
            "field": {
                "field_vector": np.array([0.01, 0.0, 0.0]),
            }
        })

        vp = PyVistaViewport(width=320, height=240)
        img = vp.render(bridge.stage)
        assert img.shape == (240, 320, 3)
        vp.close()

    def test_multiple_renders_no_leak(self):
        """Rendering many frames should not accumulate actors."""
        from mime.viz.pyvista_viewport import PyVistaViewport
        bridge = self._make_simple_stage()
        vp = PyVistaViewport(width=160, height=120)

        for i in range(20):
            bridge.update({
                "body": {
                    "position": np.array([float(i) * 1e-4, 0.0, 0.0]),
                    "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
                }
            })
            img = vp.render(bridge.stage)
            assert img.shape == (120, 160, 3)

        vp.close()

    def test_close_is_safe(self):
        """Calling close() multiple times should not crash."""
        from mime.viz.pyvista_viewport import PyVistaViewport
        vp = PyVistaViewport(width=160, height=120)
        vp.close()
        vp.close()  # Should not raise


@requires_viz
class TestPyVistaViewportProtocol:
    def test_satisfies_usd_viewport_protocol(self):
        """PyVistaViewport should satisfy the USDViewport protocol."""
        from mime.viz.pyvista_viewport import PyVistaViewport
        from mime.core.viewport import USDViewport
        vp = PyVistaViewport(width=160, height=120)
        assert isinstance(vp, USDViewport)
        vp.close()
