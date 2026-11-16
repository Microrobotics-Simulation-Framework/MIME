- **Changelog fragments** (`changes.d/`, `scripts/assemble_changelog.py`). A change writes `changes.d/<slug>.<kind>.md`
  (`added`, `changed`, `deprecated`, `removed`, `fixed`, `security` or `anomaly`) instead of editing `CHANGELOG.md`, so that
  two changes no longer conflict on its Unreleased block (18 of the last 45 pull requests did). A small pull request after a
  merge, or a burst of merges, folds the waiting fragments into the block's sections, by kind and then slug, and deletes them.
  A fragment with a heading line or unbalanced code fences, and a changelog the fold cannot read (not one Unreleased heading,
  unbalanced fences, mixed line endings), are refused before anything is written. The fold touches nothing else in the file
  and keeps its line endings, so a hand edit survives, and a second fold changes nothing. The page and the text checks read
  the fragments too.
