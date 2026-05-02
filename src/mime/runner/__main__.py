"""Entry point.

    python3 -m mime.runner <experiment.yaml>      # YAML path
    python3 -m mime.runner <experiment_dir>       # directory containing experiment.yaml

The directory form matches MICROROBOTICA's ``ExperimentRunner::start()``
contract; the YAML form is kept for ad-hoc invocations.
"""

import sys
from pathlib import Path

from mime.runner.server import run_experiment

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m mime.runner <experiment.yaml | experiment_dir>",
              file=sys.stderr)
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        target = target / "experiment.yaml"
    if not target.exists():
        print(f"experiment.yaml not found at: {target}", file=sys.stderr)
        sys.exit(2)

    run_experiment(str(target))
