#!/usr/bin/env python3
"""CI script — validate that all [@Key] citations in docs/ resolve to
entries in docs/bibliography.bib.

Adapted from MADDENING's check_citations.py for MIME's directory structure.
"""

import os
import re
import sys

BIB_PATH = os.environ.get("BIB_PATH", "docs/bibliography.bib")
DOCS_DIR = "docs"


def extract_bib_keys(bib_path):
    """Extract all @type{Key, entries from the .bib file."""
    keys = set()
    with open(bib_path) as f:
        for line in f:
            m = re.match(r"@\w+\{(\w+),", line)
            if m:
                keys.add(m.group(1))
    return keys


def extract_citations(docs_dir):
    """Extract all [@Key] citations from .md files, excluding templates."""
    citations = {}  # key -> list of (file, line_number)
    for root, dirs, files in os.walk(docs_dir):
        for fn in files:
            if fn.startswith("_") or not fn.endswith(".md"):
                continue
            fpath = os.path.join(root, fn)
            with open(fpath) as f:
                for i, line in enumerate(f, 1):
                    for m in re.finditer(r"\[@(\w+)", line):
                        key = m.group(1)
                        citations.setdefault(key, []).append((fpath, i))
    return citations


def main():
    if not os.path.exists(BIB_PATH):
        print(f"OK: no bibliography file at {BIB_PATH} — nothing to check")
        sys.exit(0)

    bib_keys = extract_bib_keys(BIB_PATH)
    citations = extract_citations(DOCS_DIR)

    errors = []
    for key, locations in citations.items():
        if key not in bib_keys:
            for fpath, lineno in locations:
                errors.append(f"{fpath}:{lineno}: [@{key}] not found in {BIB_PATH}")

    # Warn about unused bib entries (non-blocking)
    used_keys = set(citations.keys())
    unused = bib_keys - used_keys
    for key in sorted(unused):
        print(f"WARNING: {BIB_PATH} entry '{key}' is not cited in any docs/ file")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {len(citations)} citation(s) validated against {len(bib_keys)} bib entries")


if __name__ == "__main__":
    main()
