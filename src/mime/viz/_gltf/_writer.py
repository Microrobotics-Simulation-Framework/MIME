# SPDX-License-Identifier: Apache-2.0
"""The binary glTF container (glTF 2.0's GLB): the document's JSON and one buffer holding every array it names, each
chunk padded to four bytes. MIME's own, in the place of usd2gltf's dependency on gltflib: a hundred lines are not worth
a runtime dependency that a notebook must install."""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

GLB_MAGIC = 0x46546C67
GLB_VERSION = 2
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
#: glTF's component types, by numpy dtype.
COMPONENTS: dict[str, int] = {"float32": 5126, "uint32": 5125, "uint16": 5123, "uint8": 5121}
#: glTF's element types, by the number of components each element holds.
ELEMENTS: dict[int, str] = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


def _padded(data: bytes, fill: bytes) -> bytes:
    return data + fill * (-len(data) % 4)


class Buffers:
    """The one buffer of a document, and the views and accessors that name its arrays."""

    def __init__(self) -> None:
        self._data = bytearray()
        self.views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []

    def accessor(self, values: np.ndarray, *, target: int | None = None) -> int:
        """An accessor of ``values`` (one row an element; float32 or an unsigned integer type), its view placed at the
        buffer's end, four-byte aligned, with its ``min`` and ``max``; its index."""
        array = np.ascontiguousarray(values)
        if array.dtype.name not in COMPONENTS:
            raise ValueError(f"glTF holds no {array.dtype.name} component")
        width = 1 if array.ndim == 1 else int(np.prod(array.shape[1:]))
        rows = array.reshape(len(array), width)
        self._data.extend(b"\0" * (-len(self._data) % 4))
        view: dict[str, Any] = {"buffer": 0, "byteOffset": len(self._data), "byteLength": int(array.nbytes)}
        if target is not None:
            view["target"] = target
        self._data.extend(array.tobytes())
        self.views.append(view)
        held: dict[str, Any] = {"bufferView": len(self.views) - 1, "componentType": COMPONENTS[array.dtype.name],
                                "count": len(array), "type": ELEMENTS[width]}
        if len(array):
            kind = float if array.dtype.kind == "f" else int
            held["min"] = [kind(v) for v in rows.min(axis=0)]
            held["max"] = [kind(v) for v in rows.max(axis=0)]
        self.accessors.append(held)
        return len(self.accessors) - 1

    @property
    def data(self) -> bytes:
        return bytes(self._data)

    def glb(self, document: dict[str, Any]) -> bytes:
        """The GLB of ``document`` (its buffers, views and accessors are these), its JSON compact and space-padded,
        its buffer zero-padded."""
        whole = {**document, "accessors": self.accessors, "bufferViews": self.views,
                 "buffers": [{"byteLength": len(self._data)}]}
        if not self.accessors:
            del whole["accessors"], whole["bufferViews"]
        text = _padded(json.dumps(whole, separators=(",", ":"), allow_nan=False).encode("utf-8"), b" ")
        binary = _padded(self.data, b"\0")
        length = 12 + 8 + len(text) + 8 + len(binary)
        return (struct.pack("<III", GLB_MAGIC, GLB_VERSION, length) + struct.pack("<II", len(text), JSON_CHUNK) + text
                + struct.pack("<II", len(binary), BIN_CHUNK) + binary)


def read_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    """A GLB's JSON document and its binary chunk; ``ValueError`` where its header or chunks do not hold."""
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != GLB_VERSION or length != len(data):
        raise ValueError("not a glTF 2.0 binary file of its stated length")
    size, kind = struct.unpack_from("<II", data, 12)
    if kind != JSON_CHUNK:
        raise ValueError("a GLB's first chunk is its JSON")
    document: dict[str, Any] = json.loads(data[20:20 + size].decode("utf-8"))
    rest = 20 + size
    binary = b""
    if rest < len(data):
        size, kind = struct.unpack_from("<II", data, rest)
        if kind != BIN_CHUNK:
            raise ValueError("a GLB's second chunk is its buffer")
        binary = data[rest + 8:rest + 8 + size]
    return document, binary


__all__ = ["ARRAY_BUFFER", "ELEMENT_ARRAY_BUFFER", "Buffers", "read_glb"]
