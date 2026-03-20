# Commit and Push Skill — MIME

This skill enforces MIME's documentation architecture requirements on every commit and push.

## Trigger

When the user asks to commit and push, or invokes `/commit-and-push`.

---

## Pre-Commit Checklist

### 1. Tests Pass

```bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v --tb=short
```

**Gate**: All tests must pass.

### 2. Compliance Scripts Pass

```bash
python scripts/check_anomalies.py
python scripts/check_citations.py
python scripts/check_impl_mapping.py
```

**Gate**: All must exit 0.

### 3. Commit Message Convention

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

### 4. CHANGELOG.md Updated

If changes affect user-visible functionality, update `CHANGELOG.md` under `## [Unreleased]`:

- **Added** / **Changed** / **Deprecated** / **Removed** / **Fixed**
- **Verification** — changes to B0–B5 status, new benchmarks
- **Security** — required by MDCG 2019-16
- **Known Anomalies** — changes to `known_anomalies.yaml`

### 5. New Node Checks (if applicable)

If the commit adds or modifies a `MimeNode` subclass:

- [ ] `meta` ClassVar has `NodeMeta` with: `algorithm_id` (MIME-NODE-*), `stability`, `description`, `assumptions`, `limitations`, `hazard_hints`
- [ ] `mime_meta` ClassVar has `MimeNodeMeta` with role, anatomical regimes, role-specific metadata
- [ ] `@stability(StabilityLevel.EXPERIMENTAL)` decorator applied
- [ ] `update()` is JAX-traceable
- [ ] Algorithm guide in `docs/algorithm_guide/nodes/` follows `_template.md`
- [ ] Implementation Mapping table traces all equation terms to code
- [ ] Unit tests and at least one `@verification_benchmark` (MIME-VER-*)
- [ ] `validate_mime_consistency()` returns no errors

### 6. New Anomaly Checks (if applicable)

- [ ] Entry in `docs/validation/known_anomalies.yaml` with `MIME-ANO-XXX` ID
- [ ] `safety_relevance_rationale` is filled in
- [ ] CHANGELOG updated under `### Known Anomalies`
- [ ] `python scripts/check_anomalies.py` passes

### 7. Bibliography Checks (if applicable)

- [ ] All `[@Key]` citations resolve to `docs/bibliography.bib`
- [ ] `python scripts/check_citations.py` passes

## Execution

```bash
git add <specific files>
git commit -m "prefix: concise description

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
git push
```

## Quick Reference

| Change Type | CHANGELOG | Anomaly YAML | Algorithm Guide | Bibliography |
|-------------|-----------|-------------|-----------------|--------------|
| New node | Added | If limitations | Yes (new doc) | If citing papers |
| Bug fix | Fixed | If known anomaly | Update if equations change | No |
| New limitation | Known Anomalies | Yes (new entry) | Update Known Limitations | No |
| New benchmark | Verification | No | Update Verification Evidence | If citing reference |
