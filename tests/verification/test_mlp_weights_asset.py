"""The de Jongh surrogate's weights are tracked, and pinned by hash.

A clean checkout must run the de Jongh graph: before MIME 0.3.0 the weights
sat under the ignored ``data/`` and every test that needed them skipped.
"""

import hashlib

from mime.experiments.dejongh import MLP_WEIGHTS_V2_SHA256, default_mlp_weights_path


def test_tracked_weights_match_their_pinned_hash():
    path = default_mlp_weights_path()
    assert path.is_file(), f"{path} is missing: it is tracked, so the checkout is incomplete"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MLP_WEIGHTS_V2_SHA256
