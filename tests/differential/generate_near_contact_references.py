"""Writes the near-contact gates' two oracle files (``tests/oracles/test_near_contact.py``) from rows of the two methods
of fundamental solutions here, each sharing no code with MIME: ``references/sphere_plane_graded_mfs.json``, a sphere
above a plane (``graded_mfs.py``, levels B and C), and ``references/sphere_tube_mfs.json``, a sphere in a straight
tube at three sphere-to-tube radius ratios (``tube_mfs.py``, levels B and C, the vessel's half-length 2.5 radii).

The rows were computed on GitHub's runners by the MIME-ANO-154 probes (one gap a job, float64; runs listed in each
file's ``provenance``), with these scripts as they are here; each row is the script's own output. Not collected:

    python tests/differential/generate_near_contact_references.py <plane rows dir> <tube rows dir>
    python tests/differential/generate_near_contact_references.py --above <tube rows dir>

each directory holding the scripts' JSON outputs anywhere below it. The second writes
``references/sphere_tube_mfs_above.json``: the same tube MFS at ratios above the gated range (0.40 and 0.46), the gaps
the tube gate judges, from the rows the PR's review computed (its run in the file's ``provenance``)."""

import glob
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: The gaps (sphere radii) the gates judge above a plane and in a tube, and the tube's ratios.
PLANE_GAPS = (0.3, 0.2, 0.15, 0.1, 0.075, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.002, 0.001)
TUBE_GAPS = (0.3, 0.2, 0.15, 0.1, 0.075, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.002, 0.001)
RATIOS = (0.22, 0.28, 0.34)
#: The ratios above the gated range, and the gaps the tube gate judges there.
ABOVE_RATIOS = (0.4, 0.46)
ABOVE_GAPS = (0.1, 0.075, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.0075)
KEPT = ("resistance", "residual", "sources", "collocation", "centre")


def rows(directory: str, pattern: str) -> dict:
    found: dict = {}
    for path in glob.glob(str(Path(directory) / "**" / pattern), recursive=True):
        found.update(json.loads(Path(path).read_text()))
    return found


def above(tube_dir: str) -> None:
    held = rows(tube_dir, "tmfs-*.json")
    tube = {f"{ratio!r} {level} {gap!r}": {k: held[f"{ratio!r} {level} 2.5 {gap!r}"][k] for k in KEPT}
            for ratio in ABOVE_RATIOS for level in "BC" for gap in ABOVE_GAPS}
    (HERE / "references" / "sphere_tube_mfs_above.json").write_text(json.dumps({
        "what": "as sphere_tube_mfs.json, at sphere-to-tube ratios above the near-contact law's gated range: a unit "
                "sphere in a straight tube of radius 1 / ratio along z, its surface the gap from the wall along -x, unit "
                "viscosity, the 6 x 6 about its centre, by tube_mfs.py at levels B and C, half-length 2.5 tube radii",
        "ratios": list(ABOVE_RATIOS), "gaps": list(ABOVE_GAPS),
        "provenance": "W2a PR 1b-i's review, probe run 42972714723 (GitHub runners, float64), one ratio, level and "
                      "three or four gaps a job",
        "python": platform.python_version(), "rows": tube}, indent=1) + "\n")


def main() -> None:
    if sys.argv[1] == "--above":
        above(sys.argv[2])
        return
    plane_dir, tube_dir = sys.argv[1:3]
    held = rows(plane_dir, "sph-*.json")
    plane = {f"{level} {gap!r}": {k: held[f"1.0 0.0 {level} {gap!r}"][k] for k in KEPT}
             for level in "BC" for gap in PLANE_GAPS}
    (HERE / "references" / "sphere_plane_graded_mfs.json").write_text(json.dumps({
        "what": "a unit sphere above the plane z = 0, unit viscosity: the 6 x 6 about its centre (force and torque on "
                "the fluid per unit motion), by graded_mfs.py at levels B and C",
        "gaps": list(PLANE_GAPS), "levels": "graded_mfs.LEVELS's B and C",
        "provenance": "MIME-ANO-154 probe, runs 42864051476 and 42871833374 (GitHub runners, float64)",
        "python": platform.python_version(), "rows": plane}, indent=1) + "\n")
    held = rows(tube_dir, "tube-*.json")
    tube = {f"{ratio!r} {level} {gap!r}": {k: held[f"{ratio!r} {level} 2.5 {gap!r}"][k] for k in KEPT}
            for ratio in RATIOS for level in "BC" for gap in TUBE_GAPS}
    (HERE / "references" / "sphere_tube_mfs.json").write_text(json.dumps({
        "what": "a unit sphere in a straight tube of radius 1 / ratio along z, its surface the gap from the wall along "
                "-x, unit viscosity: the 6 x 6 about its centre (force and torque on the fluid per unit motion), by "
                "tube_mfs.py at levels B and C, half-length 2.5 tube radii",
        "ratios": list(RATIOS), "gaps": list(TUBE_GAPS),
        "provenance": "MIME-ANO-154 follow-up probe, run 42942082331 (GitHub runners, float64)",
        "python": platform.python_version(), "rows": tube}, indent=1) + "\n")


if __name__ == "__main__":
    main()
