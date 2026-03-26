"""Entry point: python3 -m mime.runner /path/to/experiment.yaml"""

import sys
from mime.runner.server import run_experiment

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m mime.runner <experiment.yaml>", file=sys.stderr)
        sys.exit(1)
    run_experiment(sys.argv[1])
