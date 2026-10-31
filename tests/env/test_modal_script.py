"""``mime modal``: the script it writes for a toy experiment and for script A, read by importing it against a stand-in
for ``modal`` (``modal_stub.py``), as ``modal run`` imports it, with no Modal account, call or network.

The script parses and is the only importer of ``modal``, importing nothing of MIME on the laptop; it names exact GPU
types only, and the generator refuses a loose one; it copies MIME's requirements file byte for byte; its image's
layers are the base by digest, uv copied from its image by digest, the variant's pins, MIME's wheel by hash, then the
lab's, in that order; and the experiment is mounted when a container starts, never copied into a layer."""

import ast
import hashlib
import io
import json
import platform
import re
import urllib.error

import jax
import pytest

import mime.env._modal_script as generator
from mime import __version__
from mime.build import BuildRefused
from mime.cli import main as mime_main
from mime.env import EXACT_GPUS, LOOSE_GPUS, bases, modal_script, requirements_bytes, series_of
from mime.env._pins import uv_digest, uv_image
from mime.env._modal_script import IMAGE_BUILDER_VERSION, UV_VERSION, pypi_wheel_sha256
from mime.saved import LOCK_FILE
from mime.testing.toys import full_toy
from modal_stub import load
from script_a import script_a

PINNED = "--require-hashes --no-deps --no-build"
DIGEST = "ab" * 32
SERIES = series_of(platform.python_version())
UV_IMAGE = f"ghcr.io/astral-sh/uv@{uv_digest()}"
UV_COPY = {"step": "dockerfile_commands", "commands": [f"COPY --from={UV_IMAGE} /uv /.uv/uv"]}


def pins(path):
    """The two steps that place a requirements file in the image: it is added, then installed with the copied uv."""
    return add(path, f"/.pins/{path.name}", True)


def add(local, remote, copy):
    return {"step": "add_local_file", "local": str(local), "remote": remote, "copy": copy}


def install(name, options=PINNED):
    return {"step": "run_commands", "commands": [
        f"/.uv/uv pip install --python $(command -v python) --compile-bytecode {options} --requirements /.pins/{name}"]}


def kinds(image):
    return [step["step"] for step in image]


@pytest.fixture(autouse=True, scope="module")
def x64():
    held = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", held[0])
    jax.config.update("jax_default_matmul_precision", held[1])


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """Experiments saved in float64 at the highest matmul precision, which a GPU run requires of its lock."""
    root = tmp_path_factory.mktemp("experiments")
    full_toy().save(root / "full_toy")
    script_a().save(root / "script_a")
    return root


@pytest.fixture
def released(monkeypatch):
    """MIME as a release installs it: no checkout, its wheel's hash from PyPI (here, a fixed one)."""
    monkeypatch.setattr(generator, "installed_checkout", lambda: None)


def write(folder, out, **options):
    options.setdefault("wheel_sha256", lambda version: DIGEST)
    return modal_script(folder, out=out, **options)


@pytest.mark.parametrize("name", ["full_toy", "script_a"])
@pytest.mark.parametrize("gpu", [None, "A100-40GB"])
def test_the_script_parses_and_only_it_imports_modal(saved, tmp_path, released, name, gpu):
    written = write(saved / name, tmp_path / "out", gpu=gpu)
    tree = ast.parse(written.script.read_text(encoding="utf-8"))
    imports = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    assert "modal" in imports and not any(name.split(".")[0] == "mime" for name in imports)
    held = load(written.script, tmp_path)
    assert "modal" in held["imported"] and "mime" not in held["imported"]
    assert held["app"] == f"mime-{name}" and held["entrypoints"] == ["main", "verify", "fetch"]


def test_the_layers_are_in_order_and_the_experiment_is_mounted_not_copied(saved, tmp_path, released):
    out = tmp_path / "out"
    written = write(saved / "script_a", out, gpu="A100-40GB")
    image = load(written.script, tmp_path)["image"]
    print(json.dumps(image, indent=1))
    assert image == [
        {"step": "from_registry", "tag": bases()[SERIES]["image"]},
        UV_COPY,
        pins(out / "requirements-cuda12.txt"), install("requirements-cuda12.txt"),
        pins(out / "mime-requirements.txt"), install("mime-requirements.txt"),
        {"step": "add_local_dir", "local": str(saved / "script_a"), "remote": "/experiment", "copy": False},
    ]
    assert (out / "mime-requirements.txt").read_text() == f"mime-engine=={__version__} \\\n    --hash=sha256:{DIGEST}\n"


def test_before_a_release_the_script_builds_the_checkout_and_adds_its_wheel_by_hash(saved, tmp_path):
    out = tmp_path / "out"
    written = write(saved / "full_toy", out, checkout=tmp_path / "checkout")
    held = load(written.script, tmp_path)
    wheel = out / "dev-wheel" / "mime_engine-0.0.0-py3-none-any.whl"
    digest = hashlib.sha256(b"a wheel the stub built").hexdigest()
    assert kinds(held["image"]) == ["from_registry", "dockerfile_commands", "add_local_file", "run_commands",
                                    "add_local_file", "add_local_file", "run_commands", "add_local_dir"]
    assert held["image"][4] == add(wheel, f"/wheels/{wheel.name}", True)
    assert held["image"][5] == pins(out / "dev-wheel" / "mime-requirements.txt")
    assert held["image"][6] == install("mime-requirements.txt")
    assert (out / "dev-wheel" / "mime-requirements.txt").read_text() == (
        f"mime-engine @ file:///wheels/{wheel.name} \\\n    --hash=sha256:{digest}\n")
    assert held["runner"]["mime_wheel_sha256"] == digest and not (out / "mime-requirements.txt").exists()


def test_the_labs_distributions_are_the_fourth_layer(saved, tmp_path, released):
    folder = tmp_path / "lab_toy"
    folder.mkdir()
    for path in (saved / "full_toy").rglob("*"):
        if path.is_file():
            (folder / path.relative_to(saved / "full_toy")).parent.mkdir(parents=True, exist_ok=True)
            (folder / path.relative_to(saved / "full_toy")).write_bytes(path.read_bytes())
    lock = json.loads((folder / LOCK_FILE).read_text())
    lock["entries"][0].update(distribution="mime-lab-fixture", distribution_version="0.1.0")
    (folder / LOCK_FILE).write_text(json.dumps(lock))
    out = tmp_path / "versions"
    held = load(write(folder, out).script, tmp_path)
    assert (out / "lab-requirements.txt").read_text() == "mime-lab-fixture==0.1.0\n"
    assert kinds(held["image"]) == ["from_registry", "dockerfile_commands", "add_local_file", "run_commands",
                                    "add_local_file", "run_commands", "add_local_file", "run_commands",
                                    "add_local_file", "add_local_dir"]
    assert held["image"][6] == pins(out / "lab-requirements.txt")
    assert held["image"][7] == install("lab-requirements.txt", "--no-deps")
    assert held["runner"]["lab_pins"] == "versions only"
    assert held["image"][8] == add(out / "lab-requirements.txt", "/mime-run/lab-requirements.txt", False)
    hashed = tmp_path / "lab.txt"
    hashed.write_bytes(b"mime-lab-fixture==0.1.0 \\\n    --hash=sha256:" + b"c" * 64 + b"\n")
    held = load(write(folder, tmp_path / "hashed", lab_requirements=hashed).script, tmp_path)
    assert (tmp_path / "hashed" / "lab-requirements.txt").read_bytes() == hashed.read_bytes()
    assert held["image"][7] == install("lab-requirements.txt") and held["runner"]["lab_pins"] == "hashed"


@pytest.mark.parametrize("gpu", [None, "A100-40GB"])
def test_the_requirements_file_is_mimes_byte_for_byte(saved, tmp_path, released, gpu):
    written = write(saved / "script_a", tmp_path / "out", gpu=gpu)
    variant = "cuda12" if gpu else "cpu"
    assert (tmp_path / "out" / f"requirements-{variant}.txt").read_bytes() == requirements_bytes(variant)
    assert written.variant == variant


@pytest.mark.parametrize("gpu", list(EXACT_GPUS))
def test_the_script_names_the_exact_gpu_type_it_was_given(saved, tmp_path, released, gpu):
    held = load(write(saved / "full_toy", tmp_path / "out", gpu=gpu).script, tmp_path)
    run = held["functions"]["run_experiment"]
    assert run["gpu"] == gpu == held["functions"]["verify_experiment"]["gpu"]
    assert run["env"]["JAX_DEFAULT_MATMUL_PRECISION"] == "highest" and "JAX_PLATFORMS" not in run["env"]


@pytest.mark.parametrize("gpu", [*LOOSE_GPUS, "A100-40GB:2", "V100"])
def test_a_loose_gpu_type_is_refused_and_nothing_is_written(saved, tmp_path, released, gpu):
    with pytest.raises(BuildRefused) as refused:
        write(saved / "full_toy", tmp_path / "out", gpu=gpu)
    assert refused.value.codes() == ["env.gpu-type"] and not (tmp_path / "out").exists()


def test_a_cpu_run_names_no_gpu_and_keeps_jax_on_the_cpu(saved, tmp_path, released):
    held = load(write(saved / "full_toy", tmp_path / "out").script, tmp_path)
    run = held["functions"]["run_experiment"]
    assert run["gpu"] is None and run["env"]["JAX_PLATFORMS"] == "cpu"
    assert run["env"]["JAX_ENABLE_X64"] == "1" and run["env"]["MIME_OFFLINE"] == "1"
    assert run["volumes"] == {"/assets": ["mime-assets", True], "/xla": ["mime-xla-cache", False],
                              "/runs": ["mime-runs", False]}
    assert held["functions"]["fetch_assets"]["volumes"] == {"/assets": ["mime-assets", False]}


@pytest.mark.parametrize("gpu", [None, "A100-80GB"])
def test_uv_is_copied_from_its_image_by_digest_alone_and_every_install_uses_that_copy(saved, tmp_path, released, gpu):
    """Modal's builder refuses a reference with a tag and a digest (observed on Modal, 2026-10-31), so `uv_version=` is
    never given: the script copies uv by digest itself, and the run records the digest the copy names."""
    out = tmp_path / "out"
    held = load(write(saved / "full_toy", out, gpu=gpu).script, tmp_path)
    assert not any("uv_version" in step for step in held["image"])
    assert "uv_pip_install" not in kinds(held["image"])
    copies = [command for step in held["image"] if step["step"] == "dockerfile_commands"
              for command in step["commands"]]
    (copy,) = copies
    assert re.fullmatch(r"COPY --from=ghcr\.io/astral-sh/uv@sha256:[0-9a-f]{64} /uv /\.uv/uv", copy)
    pulled = copy.split("--from=")[1].split()[0].split("@")[1]
    assert copy == UV_COPY["commands"][0] and pulled == uv_digest()
    installs = [command for step in held["image"] if step["step"] == "run_commands" for command in step["commands"]]
    assert installs and all(command.startswith("/.uv/uv pip install ") for command in installs)
    assert held["runner"]["uv"] == f"{UV_VERSION}@{pulled}"
    assert uv_image()["image"] == f"ghcr.io/astral-sh/uv:{UV_VERSION}@{pulled}"


def test_the_runner_records_the_digest_the_build_copies_not_one_beside_it(saved, tmp_path, released, monkeypatch):
    """The record is read from the script's own copy command: a different digest in the pins changes both, and the
    record cannot name a digest the build did not use."""
    other = "sha256:" + "5" * 64
    monkeypatch.setattr(generator, "uv_digest", lambda: other)
    held = load(write(saved / "full_toy", tmp_path / "out").script, tmp_path)
    assert held["image"][1]["commands"] == [f"COPY --from=ghcr.io/astral-sh/uv@{other} /uv /.uv/uv"]
    assert held["runner"]["uv"] == f"{UV_VERSION}@{other}"


def test_a_function_runs_in_one_container_unless_asked_and_is_never_retried(saved, tmp_path, released):
    held = load(write(saved / "full_toy", tmp_path / "one").script, tmp_path)
    assert {name: (settings["max_containers"], settings["retries"], settings["timeout"])
            for name, settings in held["functions"].items()} == {
        "run_experiment": (1, 0, 7200), "verify_experiment": (1, 0, 7200), "fetch_assets": (1, 0, 7200)}
    more = load(write(saved / "full_toy", tmp_path / "more", max_containers=4, timeout=900).script, tmp_path)
    assert {settings["max_containers"] for settings in more["functions"].values()} == {4}
    assert {settings["timeout"] for settings in more["functions"].values()} == {900}


def test_the_printed_fetch_goes_into_the_current_directory_which_exists(saved, tmp_path, released):
    """Client 1.6.3 fails on a destination folder that does not exist ([Errno 17], observed on Modal, 2026-10-31)."""
    script = write(saved / "full_toy", tmp_path / "out").script.read_text(encoding="utf-8")
    (line,) = [line for line in script.splitlines() if "modal volume get" in line]
    assert "modal volume get mime-runs {NAME}/{run_id} . " in line


def test_the_builder_version_is_pinned_and_recorded(saved, tmp_path, released):
    out = tmp_path / "out"
    held = load(write(saved / "full_toy", out, gpu="H100!").script, tmp_path)
    assert held["builder"] == IMAGE_BUILDER_VERSION == held["runner"]["image_builder"]
    assert held["runner"] == {"runner": "modal", "image_builder": IMAGE_BUILDER_VERSION,
                              "base": bases()[SERIES]["image"], "variant": "cuda12", "gpu_request": "H100!",
                              "requirements_sha256": hashlib.sha256(requirements_bytes("cuda12")).hexdigest(),
                              "uv": f"{UV_VERSION}@{uv_digest()}"}
    assert held["info"] == {"modal_client": "0.0-stub", "image": "unreadable"}


def test_the_run_checks_mime_against_the_version_of_the_wheel_it_pinned(saved, tmp_path, released):
    held = load(write(saved / "full_toy", tmp_path / "out", gpu="A100-40GB").script, tmp_path)
    arguments = held["check_arguments"]
    assert arguments[:4] == ["env", "check", "/experiment", "--variant"] and "--base" in arguments
    assert arguments[arguments.index("--mime-version") + 1] == __version__
    assert arguments[arguments.index("--gpu") + 1] == "A100-40GB"
    assert arguments[arguments.index("--record") + 1] == "/out/env-check.json"


def test_a_gpu_run_of_an_experiment_saved_below_the_highest_matmul_precision_is_refused_before_anything_is_written(
        saved, tmp_path, released):
    jax.config.update("jax_default_matmul_precision", None)
    try:
        full_toy().save(tmp_path / "default_precision")
    finally:
        jax.config.update("jax_default_matmul_precision", "highest")
    with pytest.raises(BuildRefused) as refused:
        write(tmp_path / "default_precision", tmp_path / "out", gpu="A100-40GB")
    assert refused.value.codes() == ["env.precision"] and not (tmp_path / "out").exists()
    assert write(tmp_path / "default_precision", tmp_path / "cpu").variant == "cpu"


@pytest.mark.parametrize("text", ["foo>=1\n", "just words\n"])
def test_a_lab_file_that_is_not_pins_is_refused_by_the_generator_before_anything_is_written(saved, tmp_path, released,
                                                                                          text):
    (tmp_path / "lab.txt").write_text(text)
    with pytest.raises(BuildRefused) as refused:
        write(saved / "full_toy", tmp_path / "out", lab_requirements=tmp_path / "lab.txt")
    assert refused.value.codes() == ["env.lab"] and not (tmp_path / "out").exists()


def test_the_image_id_and_the_clients_version_are_information_the_runs_environment_does_not_hold(saved, tmp_path,
                                                                                             released, monkeypatch):
    monkeypatch.setenv("STUB_IMAGE_ID", "im-workspace-one")
    one = load(write(saved / "full_toy", tmp_path / "out", gpu="H100!").script, tmp_path)
    monkeypatch.setenv("STUB_IMAGE_ID", "im-workspace-two")
    two = load(write(saved / "full_toy", tmp_path / "again", gpu="H100!").script, tmp_path)
    assert one["info"] == {"modal_client": "0.0-stub", "image": "im-workspace-one"}
    assert two["info"]["image"] == "im-workspace-two" and one["runner"] == two["runner"]
    assert "image" not in one["runner"] and "modal_client" not in one["runner"]


def test_the_deterministic_mode_is_off_unless_asked_for_and_only_on_a_gpu(saved, tmp_path, released):
    off = load(write(saved / "full_toy", tmp_path / "off", gpu="A100-80GB").script, tmp_path)
    on = load(write(saved / "full_toy", tmp_path / "on", gpu="A100-80GB", deterministic=True).script, tmp_path)
    assert off["functions"]["run_experiment"]["env"]["XLA_FLAGS"] == ""
    assert on["functions"]["run_experiment"]["env"]["XLA_FLAGS"] == "--xla_gpu_deterministic_ops=true"
    with pytest.raises(BuildRefused):
        write(saved / "full_toy", tmp_path / "cpu", deterministic=True)


def test_the_python_series_is_the_locks_unless_another_is_named(saved, tmp_path, released):
    assert write(saved / "full_toy", tmp_path / "a").python == SERIES
    held = load(write(saved / "full_toy", tmp_path / "b", python="3.12").script, tmp_path)
    assert held["image"][0]["tag"] == bases()["3.12"]["image"]
    with pytest.raises(BuildRefused) as refused:
        write(saved / "full_toy", tmp_path / "c", python="3.11")
    assert refused.value.codes() == ["env.python"]


def test_mimes_wheel_hash_is_read_from_pypi_and_refused_without_one(monkeypatch):
    def answering(data):
        return lambda url, timeout: io.BytesIO(json.dumps(data).encode())

    wheel = {"packagetype": "bdist_wheel", "filename": "mime_engine-1.0.0-py3-none-any.whl",
             "digests": {"sha256": DIGEST}}
    sdist = {"packagetype": "sdist", "filename": "mime_engine-1.0.0.tar.gz", "digests": {"sha256": "cd" * 32}}
    monkeypatch.setattr(generator.urllib.request, "urlopen", answering({"urls": [sdist, wheel]}))
    assert pypi_wheel_sha256("1.0.0") == DIGEST
    monkeypatch.setattr(generator.urllib.request, "urlopen", answering({"urls": [sdist]}))
    with pytest.raises(BuildRefused) as refused:
        pypi_wheel_sha256("1.0.0")
    assert refused.value.codes() == ["modal.wheel"]

    def unreachable(url, timeout):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(generator.urllib.request, "urlopen", unreachable)
    with pytest.raises(BuildRefused) as refused:
        pypi_wheel_sha256("1.0.0")
    assert refused.value.codes() == ["modal.wheel"]


def test_mime_modal_writes_beside_the_experiment_and_exits_two_when_refused(saved, tmp_path, capsys):
    folder = tmp_path / "toy"
    folder.mkdir()
    for path in (saved / "full_toy").rglob("*"):
        if path.is_file():
            (folder / path.relative_to(saved / "full_toy")).parent.mkdir(parents=True, exist_ok=True)
            (folder / path.relative_to(saved / "full_toy")).write_bytes(path.read_bytes())
    assert mime_main(["modal", str(folder), "--gpu", "A100-40GB", "--checkout", str(tmp_path)]) == 0
    assert (tmp_path / "toy-modal" / "modal_app.py").is_file() and "::main --duration" in capsys.readouterr().out
    assert mime_main(["modal", str(folder), "--gpu", "H100"]) == 2
    assert "--gpu H100!" in capsys.readouterr().err
