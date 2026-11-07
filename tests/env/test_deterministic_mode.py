"""XLA's deterministic GPU mode, opt in: ``--deterministic`` on ``mime run`` and ``mime verify``, and
``mime.env.deterministic_mode()`` in Python. XLA reads ``XLA_FLAGS`` once, when JAX's backend starts, so the mode is set
before that or not at all: once the backend has started, the Python route is refused by its code
(``env.deterministic-late``), naming the routes that work, unless an earlier call set the flags itself, which a second
call answers as it did. On the CPU it changes nothing, and says so in a line; the lock then records the flags as they
were.

No GPU runs here: what the mode does on one is measured elsewhere (``docs/running_on_modal.md``). The finding these rest on,
that the flags are read when the backend starts and not when JAX is imported, is measured here on the CPU with a flag
whose effect shows on it (``--xla_dump_to``)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import pytest

from mime.build import BuildRefused
from mime.env import CPU_DETERMINISTIC, deterministic_mode
from mime.saved import LOCK_FILE

#: ``when``: set the flag before ``import jax``, after ``import mime.cli`` (which starts no backend), or after
#: ``jax.devices()``; then compile one program and count the files XLA dumped.
DUMPS = """
import os, sys
when, out = sys.argv[1], sys.argv[2]
flag = f"--xla_dump_to={out}"
if when == "before":
    os.environ["XLA_FLAGS"] = flag
import jax, jax.numpy as jnp
import mime.cli
from jax._src import xla_bridge
started = xla_bridge.backends_are_initialized()
if when == "after_import":
    os.environ["XLA_FLAGS"] = flag
jax.devices()
if when == "after_backend":
    os.environ["XLA_FLAGS"] = flag
jax.jit(lambda x: 2.0 * x + 1.0)(jnp.arange(3.0)).block_until_ready()
print(started, len(os.listdir(out)) if os.path.isdir(out) else 0)
"""


def fresh(script: str, *arguments: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    held = {key: value for key, value in os.environ.items() if key != "XLA_FLAGS"}
    return subprocess.run([sys.executable, "-c", script, *arguments], capture_output=True, text=True, check=False,
                          env={**held, "JAX_PLATFORMS": "cpu", **(env or {})})


@pytest.mark.parametrize(("when", "dumped"), [("before", True), ("after_import", True), ("after_backend", False)])
def test_xla_reads_its_flags_once_when_the_backend_starts_and_importing_mime_starts_none(when: str, dumped: bool,
                                                                                         tmp_path: Path):
    done = fresh(DUMPS, when, str(tmp_path / "dump"))
    assert done.returncode == 0, done.stderr
    started, files = done.stdout.split()
    assert started == "False", "importing mime.cli started JAX's backend"
    assert (int(files) > 0) is dumped, done.stdout


def test_after_the_backend_started_the_python_route_is_refused_naming_the_routes_that_work(
        monkeypatch: pytest.MonkeyPatch):
    from mime.env import _host

    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.setattr(_host, "_ANSWERED", {})
    jax.devices()
    with pytest.raises(BuildRefused) as refused:
        deterministic_mode()
    print(f"\n{refused.value}")
    assert refused.value.codes() == ["env.deterministic-late"]
    (found,) = refused.value.refusals
    assert "XLA_FLAGS=--xla_gpu_deterministic_ops=true" in found.action and "--deterministic" in found.action
    assert "XLA_FLAGS" not in os.environ, "nothing set"


def test_with_the_mode_already_in_the_flags_it_is_not_refused_and_on_the_cpu_it_is_not_on(
        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
    assert jax.default_backend() == "cpu"
    assert deterministic_mode() is False


ROUTE = """
import os, json
import mime.env, mime.saved
on = mime.env.deterministic_mode()
print(json.dumps([on, os.environ.get("XLA_FLAGS"), mime.saved.environment()["xla_flags"]]))
"""


@pytest.mark.parametrize("flags", [None, "--xla_force_host_platform_device_count=1"], ids=["unset", "set"])
def test_in_a_fresh_cpu_process_it_changes_nothing_and_the_flags_are_as_they_were(flags: str | None):
    done = fresh(ROUTE, env={"XLA_FLAGS": flags} if flags is not None else None)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.splitlines()[-1]) == [False, flags, flags or ""]


TWICE = """
import os, json
import jax
import mime.env, mime.saved
first = mime.env.deterministic_mode()
started = jax.devices()
second = mime.env.deterministic_mode()
print(json.dumps([first, second, os.environ.get("XLA_FLAGS"), mime.saved.environment()["xla_flags"]]))
"""


def test_in_a_fresh_cpu_process_a_second_call_answers_as_the_first_and_is_not_refused():
    """The first call starts the backend and takes the flag off again; the second, with the backend started and the flag
    not set, is not a late call: the first set the flags itself. The control is the late first call above, refused."""
    done = fresh(TWICE)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.splitlines()[-1]) == [False, False, None, ""]


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from mime.testing.toys import full_toy

    folder = tmp_path_factory.mktemp("deterministic") / "toy"
    full_toy().save(folder)
    return folder


def test_mime_verify_deterministic_on_the_cpu_says_it_changes_nothing_and_verifies_as_without_it(toy: Path):
    command = "import sys; from mime.cli import main; sys.exit(main(sys.argv[1:]))"
    done = fresh(command, "verify", "--deterministic", str(toy), env={"JAX_ENABLE_X64": "1"})
    plain = fresh(command, "verify", str(toy), env={"JAX_ENABLE_X64": "1"})
    assert (done.returncode, plain.returncode) == (0, 0), done.stdout + done.stderr
    assert CPU_DETERMINISTIC in done.stderr and CPU_DETERMINISTIC not in plain.stderr
    assert done.stdout.splitlines()[-1] == plain.stdout.splitlines()[-1], "the same verdict, as without it"


def test_mime_run_deterministic_on_the_cpu_records_the_flags_as_they_were(toy: Path, tmp_path: Path):
    command = "import sys; from mime.cli import main; sys.exit(main(sys.argv[1:]))"
    done = fresh(command, "run", str(toy), "--duration", "40ms", "--out", str(tmp_path / "run"), "--deterministic",
                 "--no-xla-cache", env={"JAX_ENABLE_X64": "1"})
    assert done.returncode == 0, done.stdout + done.stderr
    assert CPU_DETERMINISTIC in done.stderr
    environment = json.loads((tmp_path / "run" / "experiment" / LOCK_FILE).read_text(encoding="utf-8"))["environment"]
    assert (environment["xla_flags"], environment["deterministic"]) == ("", False)
