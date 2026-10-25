# Commit and Push Skill — MIME

This skill holds every commit and push to the checks CI runs, the documentation's among them.

## Trigger

When the user asks to commit and push, or invokes `/commit-and-push`.

---

## Pre-Commit Checklist

### 1. Tests Pass

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest -m "not slow" -n auto --dist worksteal
```

**Gate**: all pass. This is CI's default lane: a `slow` or `gpu` test does not run in it (CLAUDE.md, "What CI does not
run"), and a skipped test looks like a pass.

### 2. The Documentation's Checks and the Typing Gate Pass

```bash
uv run --no-sync mime docs check    # needs the `verify` extra; runs check_anomalies.py and check_citations.py too
uv run --no-sync python scripts/typing_check.py
```

**Gate**: both exit 0. `mime docs check` runs the sources' checks (entry fields, `NodeMeta` and symbol tables, the
refusal table, the lists in `docs/ratchets/`), the docs walk and its ledger (`docs/unpaged.txt`), the number rule, then
`scripts/check_anomalies.py` and `scripts/check_citations.py` unchanged. Each failure names the file or the line to
add or delete. Its word-numeral warnings never fail it; read them.

### 3. Confidentiality

CI's `confidential` job scans the tracked files and their paths, each commit's diff and message in the push or pull
request, and the branch's name (`CONTRIBUTING.md`, "Confidentiality"). Never write a private project's name or a private
path in a file, a file's name, a commit message or a branch name, even in a commit a later one undoes. Without the key,
`python3 scripts/check_confidential.py --messages origin/v0.3..HEAD --diffs origin/v0.3..HEAD` still prints any private
path it finds, and fails.

### 4. Commit Message Convention

| Prefix | When to use |
|--------|-------------|
| `feat:` | New feature or capability |
| `fix:` | Bug fix |
| `refactor:` | Code restructuring (no behaviour change) |
| `docs:` | Documentation-only changes |
| `test:` | Test additions or changes |
| `perf:` | Performance improvement |
| `verify:` | Verification/validation evidence |
| `break:` | Breaking API change |
| `deprecate:` | Deprecation notice |
| `security:` | Security-relevant change |

### 5. CHANGELOG.md Updated

If changes affect user-visible functionality, update `CHANGELOG.md` under `## [Unreleased]`:

- **Added** / **Changed** / **Deprecated** / **Removed** / **Fixed**
- **Known Anomalies** — changes to `known_anomalies.yaml`

### 6. New Node, Entry or Refusal Code Checks (if applicable)

If the commit adds or changes a node, an entry or a refusal code (the **new-node** skill has each step):

- [ ] A new node: `meta = NodeMeta(...)` with the next free `algorithm_id` (`MIME-NODE-104` onward; every 0.2 id is
      retired), the equations in `governing_equations`, `hazard_hints` empty; `symbols`, a row per equation symbol,
      every row pinned in `docs/ratchets/symbol_rows.txt`
- [ ] A new entry: `builds`, `gradient` (unless exact), `references` (bibliography keys), `family` (an interaction),
      and its hand-written file `docs/entries/<kind>/<name>.md` naming the version it describes; a new target kind's
      `docs/mechanisms/<kind>.md`; a packaged asset's card `docs/assets/<name>.md` (or, for a table landing at 7f
      to 7i, a ledger line with its reason logged, paid at 7i). No other new ledger line
- [ ] An entry's version bumped: its file describes the new version, with the old one's history in its text
- [ ] A new refusal code: a row of `src/mime/build/refusal_table.txt`, written as a literal
- [ ] No measured number in a hand-written file or a package README: `see MIME-ANO-nnn`
- [ ] Tests where the tree has them: `tests/unit/`, an oracle in `tests/oracles/` with its control, a differential in
      `tests/differential/` for a ported node; and a `NodeCase` in `tests/saved/node_cases.py`

### 7. New Anomaly Checks (if applicable)

- [ ] Entry in `docs/validation/known_anomalies.yaml` with `MIME-ANO-XXX` ID
- [ ] `safety_relevance_rationale` is filled in
- [ ] CHANGELOG updated under `### Known Anomalies`
- [ ] `python scripts/check_anomalies.py` passes

### 8. Bibliography Checks (if applicable)

- [ ] All `[@Key]` citations resolve to `docs/bibliography.bib`, and every entry's `references` key is one of its
      entries, each field checked against its DOI record or publisher
- [ ] `uv run --no-sync mime docs check` passes

## Execution

```bash
git add <specific files>
git commit -m "prefix: concise description

<the attribution lines the session gives>"
git push -u origin <branch>    # a branch, then a pull request; never to main, never forced
```

## Quick Reference

| Change Type | CHANGELOG | Anomaly YAML | Docs sources | Bibliography |
|-------------|-----------|-------------|--------------|--------------|
| New node | Added | If limitations | `NodeMeta`, symbol table, pinned rows | If citing papers |
| New entry | Added | If limitations | entry fields, `docs/entries/<kind>/<name>.md` | `references` keys |
| Bug fix | Fixed | If known anomaly | Update if equations change | No |
| New limitation | Known Anomalies | Yes (new entry) | Name the anomaly's id | No |
| New benchmark | Added | No | No | If citing reference |
