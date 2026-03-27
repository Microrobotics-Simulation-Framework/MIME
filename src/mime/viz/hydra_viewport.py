"""HydraStormViewport — headless EGL-based USD rendering for cloud deployment.

Renders a live USD stage via UsdImagingGL.Engine into an offscreen
framebuffer, returning numpy RGBA pixels suitable for Selkies WebRTC
streaming or direct image capture.

Requires:
- NVIDIA GPU with EGL support (driver 470+)
- OpenUSD built from source with --opengl (not available in usd-core PyPI)
- PyOpenGL for EGL context management

Usage:
    viewport = HydraStormViewport(width=1280, height=720)
    viewport.set_stage(stage)
    pixels = viewport.render()  # (720, 1280, 4) uint8 RGBA
    viewport.close()

See RENDERING_PLAN.md Step 4 for the architectural specification.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports — fail gracefully with clear messages
_HAS_GL = False
_HAS_USD_GL = False

try:
    from OpenGL import GL
    from OpenGL.GL import (
        glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D,
        glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
        glGenRenderbuffers, glBindRenderbuffer, glRenderbufferStorage,
        glFramebufferRenderbuffer, glCheckFramebufferStatus,
        glViewport, glReadPixels, glClear, glClearColor,
        glDeleteFramebuffers, glDeleteTextures, glDeleteRenderbuffers,
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT,
        GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
        GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
        GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
        GL_FRAMEBUFFER_COMPLETE, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    )
    _HAS_GL = True
except ImportError:
    pass

try:
    from pxr import UsdImagingGL
    _HAS_USD_GL = True
except ImportError:
    pass


def _create_egl_context():
    """Create a headless EGL context with OpenGL 4.x.

    Returns (display, context) tuple. Raises RuntimeError on failure.
    """
    from OpenGL import EGL
    from OpenGL.EGL import (
        eglGetDisplay, eglInitialize, eglChooseConfig,
        eglCreateContext, eglMakeCurrent, eglBindAPI,
        EGL_DEFAULT_DISPLAY, EGL_NONE, EGL_NO_CONTEXT,
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_BLUE_SIZE, EGL_ALPHA_SIZE,
        EGL_DEPTH_SIZE, EGL_NO_SURFACE, EGL_OPENGL_API,
        EGLConfig,
    )
    from ctypes import c_int32

    display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
    if not display:
        raise RuntimeError("EGL: cannot get display")

    major, minor = c_int32(), c_int32()
    if not eglInitialize(display, major, minor):
        raise RuntimeError("EGL: cannot initialize")

    logger.info("EGL %d.%d initialized", major.value, minor.value)

    eglBindAPI(EGL_OPENGL_API)

    attribs = [
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_NONE,
    ]

    configs = (EGLConfig * 1)()
    num = c_int32()
    if not eglChooseConfig(display, attribs, configs, 1, num) or num.value < 1:
        raise RuntimeError("EGL: no suitable config found")

    context = eglCreateContext(display, configs[0], EGL_NO_CONTEXT, None)
    if context == EGL_NO_CONTEXT:
        raise RuntimeError("EGL: cannot create context")

    if not eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context):
        raise RuntimeError("EGL: cannot make context current (surfaceless)")

    logger.info("EGL context created (surfaceless)")
    return display, context


class HydraStormViewport:
    """Headless EGL-based USD rendering via UsdImagingGL.Engine.

    Parameters
    ----------
    width : int
        Framebuffer width in pixels.
    height : int
        Framebuffer height in pixels.
    """

    def __init__(self, width: int = 1280, height: int = 720):
        if not _HAS_GL:
            raise ImportError(
                "HydraStormViewport requires PyOpenGL. "
                "Install with: pip install PyOpenGL PyOpenGL-accelerate"
            )
        if not _HAS_USD_GL:
            raise ImportError(
                "HydraStormViewport requires UsdImagingGL (OpenUSD built "
                "from source with --opengl). Not available in usd-core PyPI."
            )

        self.width = width
        self.height = height
        self._stage = None
        self._engine: Optional[Any] = None

        # Create EGL context
        self._egl_display, self._egl_context = _create_egl_context()

        # Create FBO with colour + depth attachments
        self._fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        # Colour attachment (RGBA8 texture)
        self._color_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._color_tex)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA8,
            width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None,
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
            self._color_tex, 0,
        )

        # Depth attachment (renderbuffer)
        self._depth_rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self._depth_rbo)
        glRenderbufferStorage(
            GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height,
        )
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER,
            self._depth_rbo,
        )

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"FBO incomplete: status {status}")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        logger.info("HydraStormViewport: FBO created (%dx%d)", width, height)

    def set_stage(self, stage: Any) -> None:
        """Set the USD stage to render.

        Creates the UsdImagingGL.Engine on first call.
        """
        self._stage = stage
        if self._engine is None and _HAS_USD_GL:
            self._engine = UsdImagingGL.Engine()
            logger.info("HydraStormViewport: UsdImagingGL.Engine created")

    def render(
        self,
        stage: Optional[Any] = None,
        camera: str = "/World/Camera",
    ) -> np.ndarray:
        """Render the current stage and return RGBA pixels.

        Parameters
        ----------
        stage : Usd.Stage, optional
            If provided, overrides the stage set via set_stage().
        camera : str
            USD prim path of the camera.

        Returns
        -------
        np.ndarray
            (height, width, 4) uint8 RGBA array.
        """
        if stage is not None:
            self._stage = stage
        if self._stage is None:
            return np.zeros((self.height, self.width, 4), dtype=np.uint8)

        if self._engine is None:
            if _HAS_USD_GL:
                self._engine = UsdImagingGL.Engine()
            else:
                return np.zeros((self.height, self.width, 4), dtype=np.uint8)

        from pxr import Gf, UsdGeom

        # Bind FBO
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glViewport(0, 0, self.width, self.height)
        glClearColor(0.18, 0.18, 0.20, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Camera setup
        cam_prim = self._stage.GetPrimAtPath(camera)
        if cam_prim.IsValid() and cam_prim.IsA(UsdGeom.Camera):
            usd_cam = UsdGeom.Camera(cam_prim)
            gf_cam = usd_cam.GetCamera()
            view_matrix = gf_cam.GetViewMatrix()
            proj_matrix = gf_cam.GetProjectionMatrix()
        else:
            # Default camera looking at origin
            aspect = self.width / max(self.height, 1)
            frustum = Gf.Frustum()
            frustum.SetPerspective(45.0, aspect, 0.001, 100.0)
            eye = Gf.Vec3d(0.01, -0.02, 0.01)
            target = Gf.Vec3d(0, 0, 0)
            up = Gf.Vec3d(0, 0, 1)
            frustum.SetPosition(eye)
            from pxr.Gf import Matrix4d, Rotation
            view_matrix = Matrix4d().SetLookAt(eye, target, up)
            proj_matrix = frustum.ComputeProjectionMatrix()

        # Render
        params = UsdImagingGL.RenderParams()
        self._engine.SetCameraState(view_matrix, proj_matrix)
        self._engine.SetRenderViewport(
            Gf.Vec4d(0, 0, self.width, self.height),
        )
        self._engine.Render(self._stage.GetPseudoRoot(), params)

        # Readback
        pixels = glReadPixels(
            0, 0, self.width, self.height,
            GL_RGBA, GL_UNSIGNED_BYTE,
        )
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Convert to numpy (flip vertically — GL origin is bottom-left)
        arr = np.frombuffer(pixels, dtype=np.uint8).reshape(
            self.height, self.width, 4,
        )
        return np.flipud(arr).copy()

    def close(self) -> None:
        """Release GL and EGL resources."""
        if hasattr(self, '_fbo') and self._fbo:
            try:
                glDeleteFramebuffers(1, [self._fbo])
                glDeleteTextures(1, [self._color_tex])
                glDeleteRenderbuffers(1, [self._depth_rbo])
            except Exception:
                pass
            self._fbo = None

        if hasattr(self, '_egl_display') and self._egl_display:
            try:
                from OpenGL.EGL import eglDestroyContext, eglTerminate
                eglDestroyContext(self._egl_display, self._egl_context)
                eglTerminate(self._egl_display)
            except Exception:
                pass
            self._egl_display = None

        self._engine = None
        self._stage = None
        logger.info("HydraStormViewport: resources released")

    def __del__(self):
        self.close()
