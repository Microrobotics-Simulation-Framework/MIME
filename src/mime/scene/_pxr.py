"""OpenUSD's Python modules at MIME's boundary. usd-core ships no type stubs, so a strict checker reads every name in
them as unknown; they are bound here, once, as ``Any``, and the scene's modules call them through these names.
Nothing here changes what is called."""

from __future__ import annotations

import importlib
from typing import Any

Gf: Any = importlib.import_module("pxr.Gf")
Sdf: Any = importlib.import_module("pxr.Sdf")
Usd: Any = importlib.import_module("pxr.Usd")
UsdGeom: Any = importlib.import_module("pxr.UsdGeom")
UsdValidation: Any = importlib.import_module("pxr.UsdValidation")
Vt: Any = importlib.import_module("pxr.Vt")
