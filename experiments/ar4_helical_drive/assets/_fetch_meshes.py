#!/usr/bin/env python3
"""Fetch the upstream Annin AR4 (mk5) STL meshes.

The committed ``ar4_meshes.usdc`` already contains baked meshes for the
renderer, so an end-user does not need to run this script. Run it only
if you want to re-bake from source (e.g. upstream meshes were updated)
or you need the raw STLs for collision / inspection.

Sources:
  https://github.com/Annin-Robotics/ar4_ros_driver
  └── annin_ar4_description/meshes/ar4_mk5/

Usage:
    .venv/bin/python experiments/ar4_helical_drive/assets/_fetch_meshes.py
    .venv/bin/python experiments/ar4_helical_drive/scene/_bake_meshes.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parent
MESH_DIR = ASSET_DIR / "meshes" / "ar4_mk5"
BASE_URL = (
    "https://raw.githubusercontent.com/Annin-Robotics/ar4_ros_driver/main/"
    "annin_ar4_description/meshes/ar4_mk5"
)

# Visual STLs the bake script consumes. Collision STLs (Link_*_Col.STL)
# omitted: not all of them exist upstream and they aren't needed for
# rendering or for MIME's kinematics.
VISUAL_STLS = [
    "Link_Base_Aluminum.STL",
    "Link_Base_Enclosure.STL",
    "Link_Base_Motor.STL",
    "Link_1_Aluminum.STL", "Link_1_Motor.STL",
    "Link_2_Aluminum.STL", "Link_2_Motor.STL",
    "Link_2_Cover.STL", "Link_2_Logo.STL",
    "Link_3_Aluminum.STL", "Link_3_Motor.STL",
    "Link_4_Aluminum.STL", "Link_4_Motor.STL",
    "Link_4_Cover.STL", "Link_4_Logo.STL",
    "Link_5_Aluminum.STL", "Link_5_Motor.STL",
    "Link_6_Aluminum.STL",
]


def main() -> int:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    for name in VISUAL_STLS:
        out = MESH_DIR / name
        if out.exists() and out.stat().st_size > 1024:
            print(f"  ✓ {name} ({out.stat().st_size // 1024} KB) — already present")
            continue
        url = f"{BASE_URL}/{name}"
        print(f"  fetching {name} …")
        try:
            urllib.request.urlretrieve(url, out)
            print(f"    {out.stat().st_size // 1024} KB")
        except Exception as e:
            print(f"    FAIL: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
