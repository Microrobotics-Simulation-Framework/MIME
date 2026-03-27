"""Tests for HydraStormViewport — headless EGL rendering.

EGL context creation can be tested on any NVIDIA GPU.
Actual USD rendering requires UsdImagingGL (OpenUSD built from source).
"""

import pytest
import numpy as np

# Check dependencies
try:
    from OpenGL import GL, EGL
    _HAS_GL = True
except ImportError:
    _HAS_GL = False

try:
    from pxr import UsdImagingGL
    _HAS_USD_GL = True
except ImportError:
    _HAS_USD_GL = False

try:
    from pxr import Usd, UsdGeom
    _HAS_USD = True
except ImportError:
    _HAS_USD = False

requires_gl = pytest.mark.skipif(not _HAS_GL, reason="PyOpenGL not installed")
requires_usd_gl = pytest.mark.skipif(
    not _HAS_USD_GL, reason="UsdImagingGL not available (needs OpenUSD from source)"
)
requires_usd = pytest.mark.skipif(not _HAS_USD, reason="usd-core not installed")


class TestHydraViewportImport:
    def test_module_importable(self):
        from mime.viz import hydra_viewport
        assert hasattr(hydra_viewport, "HydraStormViewport")


@requires_gl
class TestEGLContext:
    def test_egl_context_creation(self):
        from mime.viz.hydra_viewport import _create_egl_context
        display, context = _create_egl_context()
        assert display is not None
        assert context is not None

    def test_fbo_creation(self):
        """FBO can be created after EGL context is active."""
        from mime.viz.hydra_viewport import _create_egl_context
        _create_egl_context()

        from OpenGL.GL import (
            glGenFramebuffers, glBindFramebuffer, glCheckFramebufferStatus,
            glGenTextures, glBindTexture, glTexImage2D,
            glFramebufferTexture2D, glDeleteFramebuffers, glDeleteTextures,
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
            GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, GL_FRAMEBUFFER_COMPLETE,
        )

        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, tex, 0)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        assert status == GL_FRAMEBUFFER_COMPLETE

        glDeleteFramebuffers(1, [fbo])
        glDeleteTextures(1, [tex])


@requires_gl
@requires_usd
class TestHydraViewportNoUsdGL:
    """Tests that work with EGL + usd-core but without UsdImagingGL."""

    def test_constructor_raises_without_usd_gl(self):
        """Without UsdImagingGL, constructor raises ImportError."""
        if _HAS_USD_GL:
            pytest.skip("UsdImagingGL is available — can't test missing path")
        from mime.viz.hydra_viewport import HydraStormViewport
        with pytest.raises(ImportError, match="UsdImagingGL"):
            HydraStormViewport(width=64, height=64)


@requires_gl
@requires_usd_gl
class TestHydraViewportRendering:
    """Full rendering tests — require OpenUSD built from source with GL."""

    def test_render_returns_rgba_array(self):
        from mime.viz.hydra_viewport import HydraStormViewport
        vp = HydraStormViewport(width=128, height=128)
        try:
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.Sphere.Define(stage, "/World/Sphere")
            vp.set_stage(stage)
            pixels = vp.render()
            assert pixels.shape == (128, 128, 4)
            assert pixels.dtype == np.uint8
        finally:
            vp.close()

    def test_render_without_stage_returns_zeros(self):
        from mime.viz.hydra_viewport import HydraStormViewport
        vp = HydraStormViewport(width=64, height=64)
        try:
            pixels = vp.render()
            assert pixels.shape == (64, 64, 4)
            assert np.all(pixels == 0)
        finally:
            vp.close()

    def test_close_is_safe_to_call_twice(self):
        from mime.viz.hydra_viewport import HydraStormViewport
        vp = HydraStormViewport(width=64, height=64)
        vp.close()
        vp.close()  # should not raise
