#!/bin/bash
# Overnight pipeline: de Jongh wall tables → centered speed sweep → off-center sweep
#
# Monitor progress:
#   tail -f /tmp/dejongh_overnight.log
#   ls -lh data/dejongh_benchmark/wall_tables/   # tables appear as completed
#   cat data/dejongh_benchmark/swimming_speeds_centered.json | python -m json.tool | tail
#
# Expected total: see test_dejongh_pipeline.py output for estimates

set -e
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /home/nick/MSF/MIME
LOG=/tmp/dejongh_overnight.log

{
echo "=== DE JONGH BENCHMARK OVERNIGHT RUN ==="
echo "Started at: $(date)"
echo "Log file: $LOG"
echo ""

echo "=== STEP 1/3: Wall table precomputation (4 vessel diameters) ==="
echo "Started at: $(date)"
.venv/bin/python scripts/dejongh_benchmark.py tables
echo "Tables done at: $(date)"
echo ""
ls -lh data/dejongh_benchmark/wall_tables/
echo ""

echo "=== STEP 2/3: Centered swimming speed sweep (7 designs × 4 vessels) ==="
echo "Started at: $(date)"
.venv/bin/python scripts/dejongh_benchmark.py sweep
echo "Sweep done at: $(date)"
echo ""

echo "=== STEP 3/4: Off-center sweep (FL-3, FL-9 × offsets × vessels) ==="
echo "Started at: $(date)"
.venv/bin/python scripts/dejongh_benchmark.py offcenter
echo "Off-center done at: $(date)"
echo ""

echo "=== STEP 4/4: LHS bonus sampling (surrogate training data) ==="
echo "Started at: $(date)"
.venv/bin/python scripts/dejongh_benchmark.py lhs
echo "LHS done at: $(date)"
echo ""

echo "=== COVERAGE REPORT ==="
.venv/bin/python scripts/dejongh_benchmark.py coverage
echo ""

echo "=== ALL STEPS COMPLETE ==="
echo "Finished at: $(date)"

} 2>&1 | tee "$LOG"
