"""A copy of the checkout's parts ``mime docs check`` reads, for a control to doctor and run the real check on, and
the real command run in a fresh process over a checkout."""

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Planted tests a lane does not run, beside one it does, for the evidence controls: (node name, source).
PLANTED_LANES = {
    "test_runs": "def test_runs():\n    assert True\n",
    "test_slow": "@pytest.mark.slow\ndef test_slow():\n    assert True\n",
    "test_xfail": "@pytest.mark.xfail(reason=\"planted\")\ndef test_xfail():\n    assert False\n",
    "test_marked_skip": "@pytest.mark.skip(reason=\"planted\")\ndef test_marked_skip():\n    assert True\n",
    "test_gpu": "@pytest.mark.gpu\ndef test_gpu():\n    assert True\n",
}


def pytest_section() -> str:
    """MIME's own ``[tool.pytest.ini_options]``, as its ``pyproject.toml`` writes it."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    start = text.index("[tool.pytest.ini_options]")
    end = text.find("\n[", start + 1)
    section = text[start:end if end != -1 else None]
    tomllib.loads(section)
    return section + "\n"


def planted_checkout(into: Path, files: dict[str, str]) -> Path:
    """A checkout of its own under ``into``: MIME's pytest configuration and the files ``files`` (path to text)."""
    (into / "pyproject.toml").write_text(pytest_section(), encoding="utf-8")
    for name, text in files.items():
        (into / name).parent.mkdir(parents=True, exist_ok=True)
        (into / name).write_text(text, encoding="utf-8")
    return into


def copy_checkout(into: Path) -> Path:
    """The package's sources, READMEs and refusal table, the tests' sources, ``docs/``, the two compliance scripts and
    ``pyproject.toml`` (whose pins name releases), under ``into``; returns ``into``."""
    sources = [*(ROOT / "src/mime").rglob("*.py"), *(ROOT / "tests").rglob("*.py"), *(ROOT / "src").rglob("README*.md")]
    for source in sources:
        target = into / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(ROOT / "src/mime/build/refusal_table.txt", into / "src/mime/build/refusal_table.txt")
    shutil.copyfile(ROOT / "pyproject.toml", into / "pyproject.toml")
    shutil.copytree(ROOT / "docs", into / "docs", ignore=shutil.ignore_patterns("_build"))
    (into / "scripts").mkdir()
    for script in ("check_anomalies.py",):
        shutil.copyfile(ROOT / "scripts" / script, into / "scripts" / script)
    return into


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"{old!r} is not in {path} once"
    path.write_text(text.replace(old, new), encoding="utf-8")


def docs_run(*command: str, plant: str | None = None, prelude: str = "") -> subprocess.CompletedProcess[str]:
    """``mime docs <command>`` in a fresh process, which first runs ``prelude`` and registers the planted entry
    ``plant`` (``planted_entries.py``), with the working directory the checkout's."""
    planted = "" if plant is None else f"import planted_entries\nplanted_entries.plant({plant!r})\n"
    script = (f"import sys\nsys.path.insert(0, {str(Path(__file__).parent)!r})\n{prelude}{planted}"
              f"from mime.cli import main\nsys.exit(main(['docs', *{list(command)!r}]))")
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False, cwd=ROOT)


def docs_check(root: Path = ROOT, plant: str | None = None, baseline: bool = False,
               prelude: str = "", base: str | None = None, hash_seeds: bool = False,
               evidence: bool = False) -> subprocess.CompletedProcess[str]:
    """``mime docs check --root root`` (with ``--base base`` when given) in a fresh process, which first runs
    ``prelude`` and registers the planted entry ``plant`` (``planted_entries.py``); with ``baseline``, it prints the
    references line of the entries it holds before the plant. The pages are not built under the several hash seeds
    unless ``hash_seeds``, and the test suite is not collected to resolve evidence unless ``evidence`` (the controls of
    the other stages stay fast, and a copy of the checkout holds the tests' sources without their data; the build's
    and the resolver's own tests run those stages)."""
    command = ["docs", "check", "--root", str(root), *(["--base", base] if base is not None else []),
               *([] if hash_seeds else ["--skip-hash-seed-builds"]), *([] if evidence else ["--skip-evidence"])]
    before = "planted_entries.references_before_the_plant()\n" if baseline else ""
    planted = "" if plant is None else (f"import planted_entries\n{before}planted_entries.plant({plant!r})\n")
    script = (f"import sys\nsys.path.insert(0, {str(Path(__file__).parent)!r})\n{prelude}{planted}"
              f"from mime.cli import main\nsys.exit(main({command!r}))")
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False, cwd=ROOT)


def failures(done: subprocess.CompletedProcess[str]) -> list[str]:
    """The failures a run of ``mime docs check`` printed, asserting that it failed."""
    assert done.returncode == 1, done.stdout + done.stderr
    return [line.removeprefix("FAIL: ") for line in done.stdout.splitlines() if line.startswith("FAIL: ")]
