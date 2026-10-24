"""A copy of the checkout's parts ``mime docs check`` reads, for a control to doctor and run the real check on."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def copy_checkout(into: Path) -> Path:
    """The package's sources and refusal table, the tests' sources, ``docs/`` and the two compliance scripts, under
    ``into``; returns ``into``."""
    for part in ("src/mime", "tests"):
        for source in (ROOT / part).rglob("*.py"):
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
