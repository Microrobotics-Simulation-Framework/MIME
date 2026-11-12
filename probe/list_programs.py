"""The switch scan's configurations, one name a line, the arm's share (every fourth, by PROBE_ARM's index)."""
import os, sys
sys.path[:0] = ["tests", "tests/saved"]
from switch_configurations import configurations
names = sorted(configurations())
arm = int(os.environ.get("PROBE_ARM", "0"))
print(" ".join(n for k, n in enumerate(names) if k % 4 == arm))
