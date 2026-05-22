"""§8 Step 1 — MADDENING's profiler, exposed via the experiment runner.

The runner's `profile` REP command runs `profile_graph` on the live graph and
replies with `profile_report_to_perfetto(report)`. This pins that pair against
a MIME experiment graph: a populated `ProfileReport` and a JSON-serialisable
Perfetto trace (the runner replies with `json.dumps` of it).
"""

import json

from maddening.core.simulation.profiler import (
    profile_graph,
    profile_report_to_perfetto,
)


def test_profile_graph_on_mime_experiment():
    """profile_graph profiles a MIME experiment graph and yields a report
    with per-node costs; profile_report_to_perfetto renders it JSON-safe."""
    from mime.experiments.dejongh import build_graph

    gm = build_graph()
    gm.compile()

    report = profile_graph(gm, n_steps=5, n_warmup=2)
    assert report.n_nodes > 0
    assert report.n_steps == 5
    assert report.mean_step_ms > 0.0
    assert report.node_times_ms, "per-node costs should be populated"

    perfetto = profile_report_to_perfetto(report)
    assert "traceEvents" in perfetto
    # The runner replies with json.dumps(perfetto) — it must round-trip.
    assert json.loads(json.dumps(perfetto)) == perfetto
