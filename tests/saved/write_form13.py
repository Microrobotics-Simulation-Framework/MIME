"""Write the frozen folder of saved form 13 that form 14's step is shown on (``tests/saved/test_form14.py``).

Form 14 moves the lock's asset records alone, in two places: the lock's ``assets`` and the records inside a manifest of
its ``visuals``. So the folder holds both, as form 13's own code writes them: the toy with a scene layer declared by its
record (a record that may not be redistributed, with a location: the lock's ``assets`` holds it whole and its
``referenced_files`` names the layer) and a hub whose description's mesh a manifest beside it lists (the lock's
``visuals``).

It runs only under a MIME whose saved form is 13, the code that wrote form 13, never this branch's: a worktree at the
last commit of form 13, its ``src`` first on the path, in float64 with four usable CPUs and one BLAS thread::

    git worktree add <dir> 316937a8
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=<dir>/src taskset -c 0-3 \\
        python tests/saved/write_form13.py tests/saved/form13

It writes ``<out>/scanned`` (refusing an existing one) and prints the sha256 of each file, the table the test pins."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

#: The custom licence the scan is under: its redistribution not permitted, so the layer travels as a reference.
SCAN_LICENCE = ("LicenseRef-Imaging-Lab-Restricted", "The imaging lab's data licence 1.0",
                "https://example.org/imaging-lab/LICENSE.txt", "not_permitted")
SCAN_LOCATION = "https://example.org/scans/6a3ebb1/vessel_scan.usda"
#: The mesh the hub's description draws, listed by the manifest beside it (its bytes are never read: the stage names it).
HUB_MANIFEST = {
    "format": "mime.asset_manifest/2", "name": "hub_meshes", "title": "the hub's mesh", "credit": "A Maker",
    "notice": "The hub's mesh is the maker's design file.", "archives": [],
    "assets": [{"name": "hub.stl", "sha256": "ef" * 32, "size": 684, "licence": "CC-BY-4.0",
                "locations": ["https://example.org/maker/3c1f2a7/hub.stl"], "credit": "A Maker", "role": "presentation",
                "kind": "authored"}],
    "files": {"meshes/hub.stl": {"asset": "hub.stl"}}}
HUB_VISUAL = """    </inertial>
    <visual>
      <geometry>
        <mesh filename="meshes/hub.stl"/>
      </geometry>
    </visual>
  </link>"""


def scan_bytes() -> bytes:
    """A USD layer whose default prim is a mesh of one triangle, padded to 600 bytes."""
    head = ('#usda 1.0\n(\n    defaultPrim = "vessel_scan"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
            'def Xform "vessel_scan"\n{\n    def Mesh "Link"\n    {\n        int[] faceVertexCounts = [3]\n'
            '        int[] faceVertexIndices = [0, 1, 2]\n'
            '        point3f[] points = [(0, 0, 0), (0.1, 0, 0), (0, 0.1, 0)]\n'
            '        custom string pad = "')
    tail = '"\n    }\n}\n'
    return (head + "x" * (600 - len(head) - len(tail)) + tail).encode()


def main(out: Path) -> None:
    from mime.saved import _versions

    if _versions.CURRENT != 13:
        raise SystemExit(f"this MIME writes saved form {_versions.CURRENT}: run this under the last commit of form 13")
    target = out / "scanned"
    if target.exists():
        raise SystemExit(f"{target} exists: remove it first")
    work = Path(tempfile.mkdtemp(prefix="form13-"))
    os.environ.update({"MIME_ASSETS_DIR": str(work / "cache"), "MIME_OFFLINE": "1",
                       "MIME_ASSET_LOCATIONS": str(work / "locations.toml")})
    (work / "locations.toml").write_text("", encoding="utf-8")

    from mime import assets
    from mime.assets import AssetRef
    from mime.drives import Constant
    from mime.dynamics import Arm
    from mime.scene import asset
    from mime.terms import CustomLicence
    from mime.testing._graph import declare_in_order
    from mime.testing.toys._experiments import ADD_PROBE, ATTACH_SPRING, toy_experiment
    from mime.things import Apparatus
    from mime.units import N_m, rad, rad_per_s

    data = scan_bytes()
    licence = CustomLicence(*SCAN_LICENCE)
    record = AssetRef("vessel_scan.usda", hashlib.sha256(data).hexdigest(), len(data), licence.id, (SCAN_LOCATION,),
                      credit="A Hospital's imaging lab", role="results", kind="measured", licences=(licence,),
                      citation="A. Lab, a vessel scan, 2026")
    (work / "cache").mkdir()
    (work / "cache" / record.sha256).write_bytes(data)
    assets.fetch(record, accept_licences={licence.id})
    described = work / "hub"
    described.mkdir()
    urdf = (Path(__file__).parents[1] / "descriptions" / "spin_joint.urdf").read_text(encoding="utf-8")
    (described / "hub.urdf").write_text(urdf.replace("    </inertial>\n  </link>", HUB_VISUAL), encoding="utf-8")
    (described / "visuals.json").write_text(json.dumps(HUB_MANIFEST, indent=2), encoding="utf-8")

    experiment = declare_in_order(lambda: toy_experiment("scanned"), (ADD_PROBE, ATTACH_SPRING), None)
    experiment._settings.scene.reference(asset(record), at="/World/vessel_scan")  # noqa: SLF001
    experiment.add_apparatus(Apparatus("hub", chain={
        "arm": Arm.Dynamic(description=asset(described / "hub.urdf"), initial_positions=rad((0.0,)),
                           initial_velocities=rad_per_s((0.0,)))},
        commands={"arm.joint_torques": Constant(N_m((0.0,)))}))
    experiment.licence("CC-BY-4.0")
    experiment.build().save(target)
    lock = json.loads((target / "mime.lock").read_text(encoding="utf-8"))
    assert lock["schema"] == "mime.lock/13" and list(lock["visuals"]) == ["hub_meshes"], lock["schema"]
    assert lock["referenced_files"] == {"scene/layers/vessel_scan.usda": "vessel_scan.usda"}
    for path in sorted(target.rglob("*")):
        if path.is_file():
            print(f'"{path.relative_to(target).as_posix()}": "{hashlib.sha256(path.read_bytes()).hexdigest()}",')


if __name__ == "__main__":
    main(Path(sys.argv[1]))
