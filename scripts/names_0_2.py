#!/usr/bin/env python3
"""Write the 0.2 line's public names as the tag ``pre-0.3`` holds them: ``docs/data/names_0.2.txt``, the frozen list
the migration table's names are held to (``mime.docs._migration``).

The names are what a 0.2 script built from: every node class under ``src/mime/nodes`` (a class whose base is
``MimeNode``, ``SimulationNode`` or another such class) and every effect model of ``mime.effects`` (the two families,
``HydrodynamicModel`` and ``MagneticModel``, and each ``BaseEffectModel`` subclass), each with its module. It reads the
tag with git and runs once; the list is frozen, and ``mime docs check`` holds it to its digest. Standard library only.
"""

import ast
import subprocess
import sys
from pathlib import Path

TAG = "pre-0.3"
OUT = Path("docs/data/names_0.2.txt")
FAMILIES = ("HydrodynamicModel", "MagneticModel")


def files(prefix):
    listed = subprocess.run(["git", "ls-tree", "-r", "--name-only", TAG, prefix], capture_output=True, text=True,
                            check=True).stdout.split()
    return sorted(f for f in listed if f.endswith(".py"))


def classes(path):
    source = subprocess.run(["git", "show", f"{TAG}:{path}"], capture_output=True, text=True, check=True).stdout
    module = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            yield node.name, module, [ast.unparse(base).split(".")[-1] for base in node.bases]


def main():
    found = {}
    nodes = {"MimeNode", "SimulationNode"}
    pending = [c for f in files("src/mime/nodes") for c in classes(f)]
    while True:
        added = [(name, module) for name, module, bases in pending if name not in nodes and set(bases) & nodes]
        if not added:
            break
        for name, module in added:
            nodes.add(name)
            found[name] = module
    for name, module, bases in (c for f in files("src/mime/effects") for c in classes(f)):
        if name in FAMILIES or "BaseEffectModel" in bases:
            found[name] = module
    lines = ["# The 0.2 line's public names as the tag pre-0.3 holds them, with the module of each: written by",
             "# scripts/names_0_2.py, frozen. The migration table names each of them once (mime.docs._migration).",
             *(f"{name}  {module}" for name, module in sorted(found.items()))]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{OUT}: {len(found)} names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
