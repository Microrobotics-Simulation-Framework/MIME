# Changelog fragments

A change that belongs in `CHANGELOG.md` writes a fragment here instead of editing the changelog, so two changes never
touch the same line. The file is `changes.d/<slug>.<kind>.md`, and holds the entry exactly as it should read in the
changelog: a `- ` bullet and its continuation lines.

| Kind | Section of the Unreleased block |
|---|---|
| `added` | Added |
| `changed` | Changed |
| `fixed` | Fixed |
| `anomaly` | Known Anomalies |

The slug is lower-case letters, digits, `-` and `_`, and names the change. After merges, a small pull request runs
`python scripts/assemble_changelog.py`, which folds the waiting fragments into the Unreleased block in the order of kind and
then slug, and deletes them; `--dry-run` prints the result and `--check` exits 1 while a fragment waits. The maintainer may
still edit `CHANGELOG.md` directly: the fold touches nothing else in the file.
