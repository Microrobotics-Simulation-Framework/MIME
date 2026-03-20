# Contributing to MIME

## Development Setup

```bash
# Clone the repository
git clone git@github.com:Microrobotics-Simulation-Framework/MIME.git
cd MIME

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install MADDENING from local source (not yet on PyPI)
pip install -e /path/to/MADDENING

# Install MIME in editable mode with dev dependencies
pip install -e ".[dev]"
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
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v --tb=short
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
