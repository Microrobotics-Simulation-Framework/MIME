"""FB9's explanation (F2b PR 4a, the maintainer's item 24): the centreline at the start's cut against Poiseuille's,
beside where the inflow went (the lumen's share, the start's solid share), at both stiff interiors, tilted (bent) and
along z (no bend). One JSON line a reading. Run from the tree's root."""
import argparse
import json
import sys
import time

sys.path.insert(0, "tests/flow")
from region_cases import bend_reading  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--extensions", type=float, nargs="+", required=True)
parser.add_argument("--interiors", type=float, nargs="+", required=True)
parser.add_argument("--tilt", type=float, required=True)
parser.add_argument("--steady", action="store_true")
parser.add_argument("--cells", type=int, default=8)
parser.add_argument("--output", required=True)
args = parser.parse_args()
with open(args.output, "w") as out:
    for interior in args.interiors:
        for extension in args.extensions:
            start = time.time()
            row = {**bend_reading(extension, interior=interior, tilt=args.tilt, steady=args.steady,
                                       cells_across=args.cells), "seconds": time.time() - start}
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps(row), flush=True)
