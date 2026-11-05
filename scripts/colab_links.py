"""Every link that opens one of the notebooks of ``docs/notebooks/`` in Colab names one branch, ``BRANCH``: to move them
all, change ``BRANCH`` and run ``python scripts/colab_links.py``, which rewrites each link in the README, the
documentation's pages and the notebooks. ``--check`` changes nothing and exits 1 when a link names another branch or a
notebook this checkout does not have; ``tests/examples/test_notebooks.py`` runs the same check."""

import re
import sys
from pathlib import Path

#: The branch every Colab link opens the notebooks from.
BRANCH = "v0.3-colab-notebooks"
ROOT = Path(__file__).resolve().parents[1]
PREFIX = "https://colab.research.google.com/github/Microrobotics-Simulation-Framework/MIME/blob/"
#: A link to a notebook in Colab, its branch (which may hold slashes) and the notebook's name.
LINK = re.compile(re.escape(PREFIX) + r"(?P<branch>[^\s)\"'<>]+?)/docs/notebooks/(?P<name>[A-Za-z0-9_]+\.ipynb)")
#: The badge's image, which Colab serves.
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"


def link(name: str, branch: str = BRANCH) -> str:
    """The link that opens the notebook ``name`` in Colab."""
    return f"{PREFIX}{branch}/docs/notebooks/{name}"


def files(root: Path = ROOT) -> list[Path]:
    """The files that may hold a Colab link: the README, the documentation's hand-written pages and the notebooks."""
    docs = root / "docs"
    held = [path for path in [*docs.rglob("*.md"), *docs.rglob("*.ipynb")]
            if not any(part.startswith("_") for part in path.relative_to(docs).parts)]
    return [root / "README.md", *sorted(held)]


def problems(root: Path = ROOT, branch: str = BRANCH) -> list[str]:
    """Each Colab link that names another branch than ``branch``, or a notebook ``docs/notebooks/`` does not hold."""
    found: list[str] = []
    for path in files(root):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for held in LINK.finditer(line):
                where = f"{path.relative_to(root)}:{number}"
                if held["branch"] != branch:
                    found.append(f"{where}: a Colab link names the branch {held['branch']!r}, not {branch!r}")
                if not (root / "docs" / "notebooks" / held["name"]).is_file():
                    found.append(f"{where}: a Colab link names docs/notebooks/{held['name']}, which is not there")
    return found


def rewrite(root: Path = ROOT, branch: str = BRANCH) -> list[Path]:
    """Point every Colab link at ``branch``; the files changed."""
    changed: list[Path] = []
    for path in files(root):
        text = path.read_text(encoding="utf-8")
        new = LINK.sub(lambda held: link(held["name"], branch), text)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed.append(path)
    return changed


def main(arguments: list[str]) -> int:
    if arguments == ["--check"]:
        found = problems()
        print("\n".join(found) or f"every Colab link opens the notebooks from {BRANCH}")
        return 1 if found else 0
    if arguments:
        print("usage: python scripts/colab_links.py [--check]", file=sys.stderr)
        return 2
    for path in rewrite():
        print(f"{path.relative_to(ROOT)}: its Colab links now open {BRANCH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
