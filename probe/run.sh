#!/usr/bin/env bash
# The contact-loops pull request's cylinder plants (a loop of 7 steps, a carry from two steps back, a stop at 1e-5 of the
# hull's size) and a clean arm, rated under the pull request's rule and the base's, on 3,000 cylinder draws an arm.
set -euo pipefail
mkdir -p probe/out
export OPENBLAS_NUM_THREADS=1
python probe/rates.py
