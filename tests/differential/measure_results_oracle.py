"""The results phase's gate, measured over many runs: the oracle (``mean_speed`` against the speed series' mean over
its window) and its control (the window shifted by one step either way), over the property test's own draws, run
after run. For each run of 100 draws it prints whether the oracle held at every draw and the control failed at every
draw, and the worst margins: the oracle's largest difference over its bound, and the control's smallest difference
over the same bound (above one means the shifted window failed).

    JAX_PLATFORMS=cpu python tests/differential/measure_results_oracle.py [runs]
"""

from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "results"))
sys.path.insert(0, str(HERE.parent))

from rigs import falling_bead, gain_per_step  # noqa: E402
from test_reduction_oracle import bound  # noqa: E402

from mime.units import s  # noqa: E402


def one_run() -> dict[str, float]:
    worst = {"draws": 0, "oracle_ratio": 0.0, "control_ratio": float("inf"), "oracle_failures": 0,
             "control_passes": 0}

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow], database=None)
    @given(dt=st.floats(1e-4, 2e-3), steps=st.integers(20, 200), density=st.floats(1100.0, 8000.0),
           volume=st.floats(1e-12, 1e-8), fall=st.floats(-1.0, 1.0), data=st.data())
    def draw(dt, steps, density, volume, fall, data):
        start = data.draw(st.integers(1, steps - 3))
        after = (start + data.draw(st.floats(0.05, 0.95))) * dt
        experiment, swim = falling_bead(dt, after, density, volume, fall * gain_per_step(dt, density) * steps)
        result = experiment.build().run(s(steps * dt), record=[swim.speed])
        node = result[swim].mean_speed.value
        window = result[swim].speed.between(s(after))
        tolerance = bound(window.raw)
        ratio = abs(node - window.mean().value) / tolerance
        worst["draws"] += 1
        worst["oracle_ratio"] = max(worst["oracle_ratio"], ratio)
        worst["oracle_failures"] += ratio > 1.0
        for shifted in (after - dt, after + dt):
            moved = abs(node - result[swim].speed.between(s(shifted)).mean().value) / tolerance
            worst["control_ratio"] = min(worst["control_ratio"], moved)
            worst["control_passes"] += moved <= 1.0

    draw()
    return worst


def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    held = failed = 0
    for run in range(1, runs + 1):
        found = one_run()
        held += found["oracle_failures"] == 0
        failed += found["control_passes"] == 0
        print(f"run {run:2d}: {found['draws']} draws; oracle failures {found['oracle_failures']}, worst "
              f"{found['oracle_ratio']:.3g} of its bound; control passes {found['control_passes']}, closest "
              f"{found['control_ratio']:.3g} times the bound", flush=True)
    print(f"\nthe oracle held at every draw in {held} of {runs} runs; the control failed at every draw in {failed} of "
          f"{runs} runs")


if __name__ == "__main__":
    main()
