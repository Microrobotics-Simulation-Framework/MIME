"""The surrogate's weights are in the repository and in the package, and are the weights MIME pins. In the 0.2 line a
root ``.gitignore`` rule (``/data/``) kept them out of git, so they reached neither GitHub nor PyPI, and a user had to
ask for them; this test fails if that happens again. The wheel and the sdist are checked in ``test_packaging.py``.

- The packaged file exists where an installed ``mime.data`` finds it, reads through the loader, and its sha256 is the
  pinned one, written out here rather than read from MIME.
- Git tracks it, every committed reference array the tests read, and the environment's pins and the Modal script's
  template, and ignores none of them. The control: a path git does ignore (``.venv/``) is reported as ignored, so the
  check can fail.
"""

import hashlib
import importlib.resources
import shutil
import subprocess
from pathlib import Path

import pytest

from mime.data import PACKAGED, asset, asset_bytes
from mime.surrogates import read_cholesky_mlp

ROOT = Path(__file__).resolve().parent.parent
PINNED = "ac1fca5c109fa3e2f27520cbd0ee1c0ce5e85ac31a0b1e93e7b8085c68f0a345"
WEIGHTS = "src/mime/data/mlp_cholesky_weights_v2.npz"
SCRIPT_A = "tests/differential/references/script_a/"
TRACKED = (WEIGHTS, "src/mime/data/sphere_plane_resistance.json", "src/mime/data/derivations/sphere_plane_resistance.json",
           "tests/differential/references/sphere_plane_graded_mfs.json",
           "tests/differential/references/sphere_tube_mfs.json", "tests/differential/references/port_drag.npz", "tests/differential/references/port_nodes.npz",
           "tests/differential/references/port_table.npz", "tests/differential/references/port_table_provenance.json",
           "tests/differential/references/port_table_axis_far.npz",
           "tests/differential/references/port_table_axis_far_provenance.json",
           "tests/oracles/references/dejongh_mrs_offsets.npz",
           "tests/oracles/references/helix_fl9_free_space.json",
           "tests/oracles/references/mpem_mag_manip.npz", "tests/oracles/references/mpem_mag_manip_provenance.json",
           *(f"src/mime/data/emns_calibrations/{name}.yaml" for name in ("navion_2", "octomag", "cardiomag")),
           *(SCRIPT_A + name for name in ("f64_steady.json", "f64_provenance.json", "f32_steady.json",
                                          "f32_provenance.json", "f64_freq11hz_steady.json",
                                          "f64_freq11hz_provenance.json", "f64_trajectory_every_10th_step.npz",
                                          "f64_trajectory_every_10th_step_provenance.json")),
           "tests/differential/script_a_expected_differences.json",
           *(f"tests/saved/{form}/timed_toy/" + name for form in ("form6", "form7")
             for name in ("experiment.mime.yaml", "mime.lock", "reference.npz", "scene/scene.usda")),
           *("src/mime/data/env/" + name
             for name in ("requirements-cpu.txt", "requirements-cuda12.txt", "requirements-optional.txt", "bases.json",
                          "uv.json")),
           "src/mime/env/modal_app.txt")


def test_the_packaged_weights_exist_load_and_are_the_pinned_ones():
    path = importlib.resources.files("mime.data") / PACKAGED["dejongh_mlp_v2"].file
    data = path.read_bytes()
    assert len(data) == 151887 and hashlib.sha256(data).hexdigest() == PINNED
    assert asset("dejongh_mlp_v2").sha256 == PINNED and asset_bytes(asset("dejongh_mlp_v2")) == data
    network = read_cholesky_mlp(data)
    assert len(network.layers) == 4 and network.squared_features and network.reference_radius_mm == 1.56


#: The three eMNS calibrations' pins: the sha256 of their upstreams' bytes, written out here rather than read from MIME.
CALIBRATIONS = {"navion_2": "ea6eeb803b48d73551648e4df08116373a03fed4e871eb2b5c13706f57796b56",
                "octomag": "d0d92541ea3669bb1c4d4daf4acd36b9d3100f728e6270e2aebac53906cc0ad0",
                "cardiomag": "31bd9b234586497275c8abff6ff292ece7ed029be8b6077b027264870facd51c"}


@pytest.mark.parametrize("name", sorted(CALIBRATIONS))
def test_each_calibration_is_packaged_as_its_upstream_bytes_and_pinned(name):
    data = (importlib.resources.files("mime.data") / PACKAGED[name].file).read_bytes()
    assert hashlib.sha256(data).hexdigest() == CALIBRATIONS[name] == asset(name).sha256
    assert asset_bytes(asset(name)) == data


def test_control_a_calibration_with_one_byte_changed_is_refused(monkeypatch):
    from mime import data

    held = data._contents(PACKAGED["navion_2"].file)[0]
    monkeypatch.setattr(data, "_contents", lambda file: (held + b" ", hashlib.sha256(held + b" ").hexdigest()))
    with pytest.raises(data.AssetError, match="other contents"):
        asset_bytes(asset("navion_2"))


def git(*args: str) -> subprocess.CompletedProcess[str]:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False)


@pytest.mark.parametrize("path", TRACKED)
def test_git_tracks_the_file_and_ignores_it_nowhere(path):
    assert git("ls-files", "--error-unmatch", path).returncode == 0, f"{path} is not committed"
    ignored = git("check-ignore", "-v", "--no-index", path)
    assert ignored.returncode == 1 and not ignored.stdout, f"{path} is ignored: {ignored.stdout}"


def test_the_ignore_check_can_fail():
    ignored = git("check-ignore", "-v", "--no-index", ".venv/lib/x.py")
    assert ignored.returncode == 0 and ".venv" in ignored.stdout
