"""``mime.live/1``: the messages of a live session, written independent of their transport.

A request is a JSON object with ``command`` and the command's arguments; a reply is a JSON object with ``protocol``,
``ok`` and, on a refusal, ``errors``, each in the coded shape of MIME's errors (code, class, message, why, action,
where, page). A frame is one chunk's state, the fields the session's schema names in its order: binary (a 32-byte
header, then each field's bytes, little-endian, in the run's own types) or JSON (the same keys), for debugging and web
clients. Within ``mime.live/1`` fields may be added (a new schema id: a client ignores paths it does not know); any
other change is ``mime.live/2``."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from mime.build._refusals import Refusal

PROTOCOL = "mime.live/1"
#: The first four bytes of a binary frame, then its version.
FRAME_MAGIC = b"MIMF"
FRAME_VERSION = 1
#: magic, version (u16), flags (u16), schema id (u32), frame number (u32), step (u64), time in seconds (f64).
HEADER = struct.Struct("<4sHHIIQd")
#: A frame's flags: the session's last frame, sent as it stops.
FINAL = 1
#: The commands a session takes, in the order the protocol's page lists them.
COMMANDS = ("hello", "status", "start", "pause", "step", "stop", "set", "record", "pace")


@dataclass(frozen=True)
class FieldSpec:
    """One field of a frame: its path (``<node>.<field>``, a node of the built graph and a field of its state), its
    node and field, its shape, its dtype as NumPy spells it little-endian (``<f4``), its unit where MIME knows it, and
    its byte offset in the binary frame's payload, after the header."""

    path: str
    node: str
    field: str
    shape: tuple[int, ...]
    dtype: str
    unit: str | None
    offset: int

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * np.dtype(self.dtype).itemsize

    def record(self) -> dict[str, Any]:
        return {"path": self.path, "node": self.node, "field": self.field, "shape": list(self.shape),
                "dtype": self.dtype, "unit": self.unit, "offset": self.offset}


@dataclass(frozen=True)
class Frame:
    """A frame as a subscriber receives it: its number in the session (from zero, one per frame sent), the base steps
    taken at it, the simulated time there (s), each field's value by path, and whether it is the session's last."""

    number: int
    step: int
    time: float
    values: Mapping[str, np.ndarray]
    final: bool = False


class FrameSchema:
    """The fields a session's frames carry, fixed for the session: its id is the first four bytes of the sha256 of its
    record's text, which every binary frame carries, so a frame of another schema is never read by this one."""

    def __init__(self, fields: Sequence[FieldSpec]) -> None:
        self.fields = tuple(fields)
        self.payload_bytes = sum(spec.size for spec in self.fields)
        self.frame_bytes = HEADER.size + self.payload_bytes
        text = json.dumps([spec.record() for spec in self.fields], sort_keys=True, separators=(",", ":"))
        self.id = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")

    @classmethod
    def of(cls, fields: Sequence[tuple[str, str, np.ndarray, str | None]]) -> FrameSchema:
        """The schema of ``(node, field, value, unit)`` in order, each value's shape and dtype its own."""
        specs: list[FieldSpec] = []
        offset = 0
        for node, name, value, unit in fields:
            held = np.asarray(value)
            dtype = held.dtype.newbyteorder("<").str if held.dtype.byteorder not in ("|",) else held.dtype.str
            spec = FieldSpec(f"{node}.{name}", node, name, tuple(int(n) for n in held.shape), dtype, unit, offset)
            specs.append(spec)
            offset += spec.size
        return cls(specs)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> FrameSchema:
        fields = [FieldSpec(str(f["path"]), str(f["node"]), str(f["field"]), tuple(int(n) for n in f["shape"]),
                            str(f["dtype"]), cast(str | None, f.get("unit")), int(f["offset"]))
                  for f in cast(list[Mapping[str, Any]], record["fields"])]
        schema = cls(fields)
        if int(record["id"]) != schema.id:
            raise ValueError(f"the schema's id is {record['id']}, and its fields hash to {schema.id}")
        return schema

    def record(self) -> dict[str, Any]:
        return {"id": self.id, "frame_bytes": self.frame_bytes, "header_bytes": HEADER.size,
                "byte_order": "little", "fields": [spec.record() for spec in self.fields]}

    def values(self, state: Mapping[str, Mapping[str, Any]]) -> dict[str, np.ndarray]:
        return {spec.path: np.asarray(state[spec.node][spec.field]) for spec in self.fields}

    def encode(self, number: int, step: int, time: float, values: Mapping[str, np.ndarray], *,
               final: bool = False) -> bytes:
        """A binary frame: the header, then each field's bytes in the schema's order and dtype."""
        parts = [HEADER.pack(FRAME_MAGIC, FRAME_VERSION, FINAL if final else 0, self.id, number & 0xFFFFFFFF, step,
                             time)]
        for spec in self.fields:
            held = np.asarray(values[spec.path], dtype=np.dtype(spec.dtype))
            if held.shape != spec.shape:
                raise ValueError(f"{spec.path} has shape {held.shape}, and the schema holds {spec.shape}")
            parts.append(held.tobytes())
        return b"".join(parts)

    def decode(self, data: bytes) -> Frame:
        """A binary frame of this schema, read back: refused (``ValueError``) when it is not one, of another version
        or schema, or of another length."""
        if len(data) != self.frame_bytes:
            raise ValueError(f"a frame of {len(data)} bytes; this schema's are {self.frame_bytes}")
        magic, version, flags, schema, number, step, time = HEADER.unpack_from(data)
        if magic != FRAME_MAGIC or version != FRAME_VERSION:
            raise ValueError(f"not a {PROTOCOL} frame: it starts {magic!r}, version {version}")
        if schema != self.id:
            raise ValueError(f"a frame of schema {schema}; this one is {self.id}")
        values: dict[str, np.ndarray] = {}
        for spec in self.fields:
            start = HEADER.size + spec.offset
            values[spec.path] = np.frombuffer(data, dtype=np.dtype(spec.dtype), count=int(np.prod(spec.shape)),
                                              offset=start).reshape(spec.shape)
        return Frame(int(number), int(step), float(time), values, bool(flags & FINAL))

    def json_frame(self, number: int, step: int, time: float, values: Mapping[str, np.ndarray], *,
                   final: bool = False) -> dict[str, Any]:
        """The same frame as JSON: the header's keys, and each field's value as nested lists."""
        return {"protocol": PROTOCOL, "frame": number, "step": step, "time": time, "schema": self.id, "final": final,
                "fields": {spec.path: np.asarray(values[spec.path]).tolist() for spec in self.fields}}


class FrameOrder:
    """What a client checks of the frames it receives, by their numbers: each the one after the last, so that a frame
    dropped (a gap) or delivered out of order (a number at or below one already seen) is seen, and said."""

    def __init__(self) -> None:
        self.expected = 0
        self.problems: list[str] = []

    def see(self, frame: Frame) -> str | None:
        """The problem with ``frame``'s place, or ``None`` when it is the next."""
        problem = None
        if frame.number > self.expected:
            problem = f"frames {self.expected} to {frame.number - 1} were dropped before frame {frame.number}"
        elif frame.number < self.expected:
            problem = f"frame {frame.number} arrived after frame {self.expected - 1}: out of order"
        if problem is not None:
            self.problems.append(problem)
        self.expected = max(self.expected, frame.number + 1)
        return problem


def frame_from_json(message: Mapping[str, Any], schema: FrameSchema) -> Frame:
    """A JSON frame read back into arrays of the schema's dtypes."""
    if message.get("protocol") != PROTOCOL or int(message["schema"]) != schema.id:
        raise ValueError(f"not a frame of this {PROTOCOL} schema")
    fields = cast(Mapping[str, Any], message["fields"])
    values = {spec.path: np.asarray(fields[spec.path], dtype=np.dtype(spec.dtype)).reshape(spec.shape)
              for spec in schema.fields}
    return Frame(int(message["frame"]), int(message["step"]), float(message["time"]), values,
                 bool(message.get("final")))


def error_record(refusal: Refusal) -> dict[str, Any]:
    """A refusal in the coded shape a reply carries."""
    return {"code": refusal.code, "class": refusal.category.value, "message": refusal.message, "why": refusal.why,
            "action": refusal.action, "where": refusal.where, "page": refusal.page}


def reply(command: str, **fields: Any) -> dict[str, Any]:
    return {"protocol": PROTOCOL, "ok": True, "command": command, **fields}


def refused(command: str | None, refusals: Sequence[Refusal]) -> dict[str, Any]:
    return {"protocol": PROTOCOL, "ok": False, "command": command, "errors": [error_record(r) for r in refusals]}


def is_frame(data: bytes) -> bool:
    """Whether a message is a binary frame, by its first bytes (a JSON message starts with ``{``)."""
    return data[:4] == FRAME_MAGIC


__all__ = ["COMMANDS", "FINAL", "FRAME_MAGIC", "FRAME_VERSION", "HEADER", "PROTOCOL", "FieldSpec", "Frame",
           "FrameOrder", "FrameSchema", "error_record", "frame_from_json", "is_frame", "refused", "reply"]
