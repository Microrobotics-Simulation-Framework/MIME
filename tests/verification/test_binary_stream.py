"""§8 Step 3 — the runner's opt-in binary stream format.

The runner can publish ResultFrames as compact BinaryStateEncoder frames
instead of JSON (selected by MIME_STREAM_FORMAT / the set_stream_format REP
command; a consumer discovers the format + decode schema via stream_info).
This pins the encode path: a ResultFrame flattened by
``_result_frame_to_binary_state`` round-trips through ``BinaryStateEncoder``
/ ``decode_frame`` with pose and scalar values intact.
"""

from maddening.api.binary_encoder import BinaryStateEncoder, decode_frame
from mime.runner.server import _result_frame_to_binary_state


def test_binary_stream_round_trips_result_frame():
    """A ResultFrame survives the binary encode/decode round trip."""
    frame = {
        "simTime": 2.5,
        "positions": {"body": {"x": 1.0, "y": 2.0, "z": 3.0}},
        "orientations": {"body": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}},
        "scalars": {"drag_torque_z": 0.25},
        "meshes": {},
    }
    state = _result_frame_to_binary_state(frame)
    encoder = BinaryStateEncoder(state)
    blob = encoder.encode(frame["simTime"], state)

    schema = encoder.schema()
    sim_time, values = decode_frame(blob, schema)
    assert sim_time == frame["simTime"]

    seg = {
        (f["node"], f["field"]):
        values[f["offset"]:f["offset"] + f["count"]]
        for f in schema["fields"]
    }
    assert list(seg[("poses", "body.pos")]) == [1.0, 2.0, 3.0]
    assert list(seg[("poses", "body.quat")]) == [0.0, 1.0, 0.0, 0.0]
    assert seg[("scalars", "drag_torque_z")][0] == 0.25
