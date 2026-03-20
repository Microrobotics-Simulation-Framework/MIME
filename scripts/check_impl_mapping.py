#!/usr/bin/env python3
"""CI script — validate Implementation Mapping tables in algorithm guides.

Parses each algorithm guide's Implementation Mapping table, extracts
function qualified names from the 'Implementation' column, and verifies
via importlib + getattr() that each resolves to an existing callable.
"""

import importlib
import os
import re
import sys

ALGORITHM_GUIDE_DIR = "docs/algorithm_guide/nodes"


def extract_impl_references(md_path):
    """Extract function qualified names from Implementation Mapping tables."""
    refs = []
    in_table = False
    with open(md_path) as f:
        for i, line in enumerate(f, 1):
            if "Implementation Mapping" in line:
                in_table = True
                continue
            if in_table and line.startswith("##"):
                in_table = False
                continue
            if in_table and "|" in line:
                cells = [c.strip() for c in line.split("|")]
                if len(cells) >= 3:
                    impl = cells[2]
                    # Extract qualified names like module.Class.method()
                    for m in re.finditer(r"`([\w.]+(?:\(\))?)`", impl):
                        name = m.group(1).rstrip("()")
                        if "." in name and not name.startswith("jnp.") and not name.startswith("jax."):
                            refs.append((name, md_path, i))
    return refs


def verify_callable(qualified_name):
    """Try to import and resolve a qualified name to a callable."""
    parts = qualified_name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:i])
        attr_path = parts[i:]
        try:
            mod = importlib.import_module(module_path)
            obj = mod
            for attr in attr_path:
                obj = getattr(obj, attr)
            return True
        except (ImportError, AttributeError):
            continue
    return False


def main():
    if not os.path.isdir(ALGORITHM_GUIDE_DIR):
        print(f"OK: no algorithm guide directory at {ALGORITHM_GUIDE_DIR}")
        sys.exit(0)

    all_refs = []
    for fn in os.listdir(ALGORITHM_GUIDE_DIR):
        if fn.startswith("_") or not fn.endswith(".md"):
            continue
        fpath = os.path.join(ALGORITHM_GUIDE_DIR, fn)
        all_refs.extend(extract_impl_references(fpath))

    if not all_refs:
        print("OK: no implementation mapping references found")
        sys.exit(0)

    errors = []
    for name, fpath, lineno in all_refs:
        if not verify_callable(name):
            errors.append(f"{fpath}:{lineno}: `{name}` does not resolve to a callable")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {len(all_refs)} implementation mapping reference(s) verified")


if __name__ == "__main__":
    main()
