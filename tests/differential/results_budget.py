"""The results phase's budget (7k): what a run costs per step without ``record=`` against the run path before it, what
recording costs per recorded series per step, and what ``RunResult.read`` costs on a folder of about a gigabyte.

    JAX_PLATFORMS=cpu python tests/differential/results_budget.py [repeats]

Each condition is run in alternation with the others, ``repeats`` times, after a warm-up that compiles every program,
and the medians are printed with the quartiles. The run path before this phase is ``ChunkedRun.run`` with no history,
which a run that records nothing still calls unchanged; the difference measured is what the result adds around it.
"""

from __future__ import annotations

import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from script_a import script_a  # noqa: E402

from mime.registry import OutputKind  # noqa: E402
from mime.results import Layout, ObservationLayout, OutputLayout, RunResult, Samples  # noqa: E402
from mime.results._result import RUN_MANIFEST  # noqa: E402
from mime.testing.toys import ToyBody, ToyRegionProbe, ToySpring, toy_experiment  # noqa: E402
from mime.things import Body, Medium, State, Structure  # noqa: E402
from mime.units import LENGTH, m_per_s, mm, s  # noqa: E402


def quartiles(samples: list[float]) -> tuple[float, float, float]:
    q = statistics.quantiles(samples, n=4)
    return q[0], statistics.median(samples), q[2]


def timed(conditions: dict[str, object], repeats: int) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {name: [] for name in conditions}
    for name, run in conditions.items():
        run()  # type: ignore[operator]
    for _ in range(repeats):
        for name, run in conditions.items():
            started = time.perf_counter()
            run()  # type: ignore[operator]
            found[name].append(time.perf_counter() - started)
    return found


def report(title: str, found: dict[str, list[float]], steps: int, base: str) -> None:
    print(f"\n{title} ({steps} steps a run)")
    reference = statistics.median(found[base])
    for name, samples in found.items():
        low, mid, high = quartiles(samples)
        print(f"  {name:48s} {mid / steps * 1e6:9.2f} us/step  (quartiles {low / steps * 1e6:.2f} to "
              f"{high / steps * 1e6:.2f})  ratio {mid / reference:.3f}")


def script_a_costs(repeats: int) -> None:
    built = script_a().build()
    swim = "swim_speed:umr.speed"
    steps = 2000
    duration = s(steps * 5e-4)
    found = timed({
        "before 7k: ChunkedRun.run, no history": lambda: built._runs.run(steps, history=False),  # noqa: SLF001
        "run(), nothing recorded": lambda: built.run(duration),
        "run(record=[speed])": lambda: built.run(duration, record=[swim]),
        "run(record=[speed], every=10 steps)": lambda: built.run(duration, record=[swim], every=s(5e-3)),
        "run(history=True)": lambda: built.run(duration, history=True),
    }, repeats)
    report("script A (the surrogate drag, float32, CPU)", found, steps, "before 7k: ChunkedRun.run, no history")


def probed():
    """A toy body on a spring probing three balls' regions: nine per-step outputs, three of them vectors."""
    from mime.scene import Sphere

    experiment = toy_experiment("probed")
    experiment.add_body(Body("probe", dynamics=ToyBody(), initial=State(position=mm((-0.4, 0.6, 0.3)),
                                                                        velocity=m_per_s((0.02, -0.01, 0.005)))))
    experiment.attach(ToySpring(), body="probe")
    for index in range(3):
        name = f"ball{index}"
        experiment.add_structure(Structure(name, geometry=Sphere(radius=mm(1.0 + index))))
        experiment.add_medium(Medium(f"{name}_solid", fills=f"{name}.solid"))
        experiment.observe(ToyRegionProbe(body="probe", medium=f"{name}_solid"))
    return experiment.build()


def series_costs(repeats: int) -> None:
    built = probed()
    real = [output.path for output in built.layout.per_step()]
    steps = 5000
    duration = s(steps * 0.02)
    found = timed({
        "before 7k: ChunkedRun.run, no history": lambda: built._runs.run(steps, history=False),  # noqa: SLF001
        "run(), nothing recorded": lambda: built.run(duration),
        "record 1 series": lambda: built.run(duration, record=real[:1]),
        "record 3 series": lambda: built.run(duration, record=real[:3]),
        f"record {len(real)} series": lambda: built.run(duration, record=real),
    }, repeats)
    report(f"three region probes on a toy body ({len(real)} per-step outputs)", found, steps,
           "before 7k: ChunkedRun.run, no history")


def gigabyte_folder(root: Path) -> Path:
    """A run folder of about a gigabyte, written as ``run_to`` writes one: a history of 125 000 steps of a 1000-float
    field, a recorded series and the final state, with its manifest."""
    steps, width = 125_000, 1000
    output = OutputLayout("big:b", "level", "big:b", "level", OutputKind.PER_STEP, LENGTH, (), "a level")
    at_end = OutputLayout("big:b", "total", "big:b", "total", OutputKind.AT_END, LENGTH, (), "a total")
    layout = Layout("big", 1e-3, (ObservationLayout("big:b", "big 0.1.0", (output, at_end)),), ())
    counts = np.arange(1, steps + 1, dtype=np.int64)
    history = {"big:b": {"field": np.ones((steps, width), dtype=np.float64), "level": counts.astype(np.float32)}}
    final = {"big:b": {"field": np.ones(width), "level": np.float32(steps), "total": np.float32(1.0)}}
    result = RunResult(steps, final, history, layout=layout, recorded={output.path: Samples(counts, counts.astype(
        np.float32))}, every=1)
    folder = root / "run"
    folder.mkdir()
    np.savez(folder / "outputs.npz", allow_pickle=False, **result.arrays())
    (folder / RUN_MANIFEST).write_text(json.dumps(result.manifest(), sort_keys=True), encoding="utf-8")
    return folder


def read_costs(repeats: int) -> None:
    root = Path(tempfile.mkdtemp(prefix="mime-results-budget-"))
    try:
        folder = gigabyte_folder(root)
        size = sum(path.stat().st_size for path in folder.iterdir())
        print(f"\nRunResult.read on a folder of {size / 2**30:.2f} GiB (page cache warm: the folder was just written)")
        for what, step in (("read(folder)", lambda: RunResult.read(folder)),
                           ("read(folder) and its output at the end", lambda: RunResult.read(folder)["big:b"].total),
                           ("read(folder) and its series' mean",
                            lambda: RunResult.read(folder)["big:b"].level.mean()),
                           ("read(folder) and one history field (the gigabyte)",
                            lambda: RunResult.read(folder).history["big:b"]["field"])):  # type: ignore[index]
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                step()
                samples.append(time.perf_counter() - started)
            low, mid, high = quartiles(samples)
            print(f"  {what:52s} {mid * 1e3:9.1f} ms  (quartiles {low * 1e3:.1f} to {high * 1e3:.1f})")
    finally:
        shutil.rmtree(root)


def main() -> None:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    script_a_costs(repeats)
    series_costs(repeats)
    read_costs(max(3, repeats // 2))


if __name__ == "__main__":
    main()
