# Contributing to MIME

## Development Setup

```bash
git clone git@github.com:Microrobotics-Simulation-Framework/MIME.git
cd MIME

# The locked environment, with the `verify` extra and the `dev` dependency group (pytest, Hypothesis, pyright,
# hatchling), which uv installs by default
uv sync --locked --extra verify
uv pip install --no-deps ./lab_fixture
```

## ID Prefix Convention

MIME uses the `MIME-` prefix for all compliance identifiers:

| Category | Prefix | Example |
|----------|--------|---------|
| Node algorithm IDs | `MIME-NODE-` | `MIME-NODE-001` |
| Anomaly IDs | `MIME-ANO-` | `MIME-ANO-001` |
| Verification benchmark IDs | `MIME-VER-` | `MIME-VER-001` |

Never use `MADD-` prefixed IDs for MIME artifacts. Cross-reference upstream MADDENING anomalies by their `MADD-ANO-*` ID in the `safety_relevance_rationale` field.

## Commit Message Convention

| Prefix | Meaning |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `refactor:` | Code restructuring (no behaviour change) |
| `docs:` | Documentation only |
| `test:` | Test additions or changes |
| `perf:` | Performance improvement |
| `verify:` | Verification/validation evidence |
| `break:` | Breaking change |
| `deprecate:` | Deprecation notice |
| `security:` | Security-relevant change |

## Running Tests

```bash
JAX_PLATFORMS=cpu uv run python -m pytest
uv run python scripts/typing_check.py    # pyright, strict over src/mime, at zero errors
```

## Compliance Scripts

```bash
python scripts/check_anomalies.py
python scripts/check_citations.py
python scripts/check_impl_mapping.py
```

## MADDENING Version Update Policy

When MADDENING publishes a new version:

1. Review MADDENING's CHANGELOG.md — specifically Known Anomalies, Security, Verification sections
2. Review new/changed entries in MADDENING's known_anomalies.yaml
3. Run MIME's full test suite against the candidate version
4. Update the version pin in pyproject.toml
5. Update docs/validation/soup_package.md Section 8
6. Document the update in CHANGELOG.md
