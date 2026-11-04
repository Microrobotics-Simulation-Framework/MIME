"""The GLB container MIME writes in place of gltflib (``mime.viz._gltf._writer``), read back by an independent reader,
pygltflib (the ``dev`` group only): its header's length is the file's, each chunk is padded to four bytes, and every
accessor's count, bounds and values are the arrays given. The control: a file whose stated length is one byte short,
or whose JSON chunk is not padded, is refused by both readers' checks."""

import struct

import numpy as np
import pygltflib
import pytest

from mime.viz._gltf._writer import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, Buffers, read_glb


def scene() -> tuple[bytes, dict[str, np.ndarray]]:
    rng = np.random.default_rng(7)
    arrays = {"points": rng.normal(size=(5, 3)).astype(np.float32),
              "indices": np.array([0, 1, 2, 2, 3, 4], dtype=np.uint16),
              "times": np.array([0.0, 0.5, 1.25], dtype=np.float32),
              "rotations": rng.normal(size=(3, 4)).astype(np.float32)}
    buffers = Buffers()
    position = buffers.accessor(arrays["points"], target=ARRAY_BUFFER)
    indices = buffers.accessor(arrays["indices"], target=ELEMENT_ARRAY_BUFFER)
    times = buffers.accessor(arrays["times"])
    rotations = buffers.accessor(arrays["rotations"])
    document = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
                "nodes": [{"name": "body", "mesh": 0}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": position}, "indices": indices}]}],
                "animations": [{"samplers": [{"input": times, "output": rotations}],
                                "channels": [{"sampler": 0, "target": {"node": 0, "path": "rotation"}}]}]}
    return buffers.glb(document), arrays


def test_an_independent_reader_reads_every_array_back():
    data, arrays = scene()
    held = pygltflib.GLTF2.load_from_bytes(data)
    blob = held.binary_blob()
    assert blob is not None and len(held.accessors) == 4 and len(held.bufferViews) == 4
    for accessor, (name, array) in zip(held.accessors, arrays.items(), strict=True):
        view = held.bufferViews[accessor.bufferView]
        assert view.byteOffset % 4 == 0, name
        read = np.frombuffer(blob, dtype=array.dtype, count=array.size, offset=view.byteOffset).reshape(array.shape)
        assert np.array_equal(read, array) and accessor.count == len(array), name
        rows = array.reshape(len(array), -1)
        assert accessor.min == pytest.approx(rows.min(axis=0).tolist()) and accessor.max == pytest.approx(
            rows.max(axis=0).tolist()), name


def test_the_header_states_the_files_length_and_each_chunk_is_padded():
    data, _ = scene()
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert (magic, version, length) == (0x46546C67, 2, len(data))
    json_length, kind = struct.unpack_from("<II", data, 12)
    assert kind == 0x4E4F534A and json_length % 4 == 0 and data[20 + json_length - 1:20 + json_length] in (b" ", b"}")
    binary_length, kind = struct.unpack_from("<II", data, 20 + json_length)
    assert kind == 0x004E4942 and binary_length % 4 == 0 and 20 + json_length + 8 + binary_length == len(data)
    document, binary = read_glb(data)
    last = document["bufferViews"][-1]
    assert document["buffers"] == [{"byteLength": last["byteOffset"] + last["byteLength"]}]
    assert len(binary) == binary_length


def test_control_a_wrong_length_is_refused_by_both_readers():
    data, _ = scene()
    short = struct.pack("<III", 0x46546C67, 2, len(data) - 1) + data[12:]
    with pytest.raises(ValueError, match="stated length"):
        read_glb(short)
    unpadded = data[:12] + struct.pack("<II", struct.unpack_from("<I", data, 12)[0] - 1, 0x4E4F534A) + data[20:]
    with pytest.raises(ValueError):  # its JSON cut short by a byte: no document
        pygltflib.GLTF2.load_from_bytes(unpadded)


def test_an_array_of_a_type_gltf_has_no_component_for_is_refused():
    with pytest.raises(ValueError, match="float64"):
        Buffers().accessor(np.zeros((2, 3)))
