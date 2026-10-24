"""A copy of the checkout's parts ``mime docs check`` reads, for a control to doctor and run the real check on, and
the real command run in a fresh process over a checkout."""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def copy_checkout(into: Path) -> Path:
    """The package's sources, READMEs and refusal table, the tests' sources, ``docs/`` and the two compliance scripts,
    under ``into``; returns ``into``."""
    sources = [*(ROOT / "src/mime").rglob("*.py"), *(ROOT / "tests").rglob("*.py"), *(ROOT / "src").rglob("README*.md")]
    for source in sources:
        target = into / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(ROOT / "src/mime/build/refusal_table.txt", into / "src/mime/build/refusal_table.txt")
    shutil.copytree(ROOT / "docs", into / "docs")
    (into / "scripts").mkdir()
    for script in ("check_citations.py", "check_anomalies.py"):
        shutil.copyfile(ROOT / "scripts" / script, into / "scripts" / script)
    return into


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"{old!r} is not in {path} once"
    path.write_text(text.replace(old, new), encoding="utf-8")


def docs_check(root: Path = ROOT, plant: str | None = None, baseline: bool = False,
               prelude: str = "") -> subprocess.CompletedProcess[str]:
    """``mime docs check --root root`` in a fresh process, which first runs ``prelude`` and registers the planted entry
    ``plant`` (``planted_entries.py``); with ``baseline``, it prints the references line of the entries it holds
    before the plant."""
    command = ["docs", "check", "--root", str(root)]
    before = "planted_entries.references_before_the_plant()\n" if baseline else ""
    planted = "" if plant is None else (f"import planted_entries\n{before}planted_entries.plant({plant!r})\n")
    script = (f"import sys\nsys.path.insert(0, {str(Path(__file__).parent)!r})\n{prelude}{planted}"
              f"from mime.cli import main\nsys.exit(main({command!r}))")
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False, cwd=ROOT)


def failures(done: subprocess.CompletedProcess[str]) -> list[str]:
    """The failures a run of ``mime docs check`` printed, asserting that it failed."""
    assert done.returncode == 1, done.stdout + done.stderr
    return [line.removeprefix("FAIL: ") for line in done.stdout.splitlines() if line.startswith("FAIL: ")]
