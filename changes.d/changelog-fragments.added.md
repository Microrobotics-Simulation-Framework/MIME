- **Changelog fragments** (`changes.d/`, `scripts/assemble_changelog.py`). A change writes `changes.d/<slug>.<kind>.md`
  (`added`, `changed`, `fixed` or `anomaly`) instead of editing `CHANGELOG.md`, so that two changes no longer conflict on its
  Unreleased block (18 of the last 45 pull requests did). A small pull request after a merge, or a burst of merges, folds the
  waiting fragments into the block's sections, by kind and then slug, and deletes them; the fold touches nothing else in the
  file, so a hand edit survives, and a second fold changes nothing. The page and the text checks read the fragments too.
