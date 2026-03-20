# MIME

MIcrorobotics Multiphysics Engine.

## What is MIME?

MIME is a domain-specific physics engine for microrobot simulation, built on the [MADDENING](https://github.com/Microrobotics-Simulation-Framework/MADDENING) framework. It provides microrobotics-specific node classes, a structured asset schema, control abstractions, uncertainty models, and a benchmark suite (B0–B5).

MIME sits at Layer 2 of an open-source stack: MADDENING (physics framework) → MIME (microrobotics engine) → MICROBOTICA (simulator + registry).

## Intended Use and Disclaimers

> **MIME is research software.** It is not a medical device as defined by EU MDR (EU 2017/745) or US FDA regulations. It has no medical purpose. It is not intended for clinical use, clinical decision-making, or patient diagnosis. It has not been CE-marked, cleared, or approved by any regulatory body.
>
> When used as a component within regulated medical software, MIME is classified as SOUP (Software of Unknown Provenance) under IEC 62304. The device manufacturer is responsible for assessing MIME's suitability and performing all required verification and validation. See [Regulatory Documentation](docs/regulatory/) for details.

## Quick Start

```bash
# Install MADDENING from local source (not yet on PyPI)
pip install -e /path/to/MADDENING

# Install MIME
pip install -e ".[dev]"

# Run tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture Plan](ARCHITECTURE_PLAN.md) | System architecture and design decisions |
| [Node Taxonomy](MIME_NODE_TAXONOMY.md) | Scientific rationale for physics node categories |
| [Documentation Architecture](DOCUMENTATION_ARCHITECTURE.md) | Documentation standards and compliance |
| [Algorithm Guide](docs/algorithm_guide/) | Mathematical docs per node |
| [Validation](docs/validation/) | V&V evidence, SOUP package, anomaly registry |
| [Regulatory](docs/regulatory/) | Intended use, downstream integration |
| [Contributing](CONTRIBUTING.md) | Development setup, conventions |
| [CHANGELOG](CHANGELOG.md) | Version history |

## Citation

If you use MIME in academic work, please cite:

```bibtex
@software{mime,
  title = {MIME: MIcrorobotics Multiphysics Engine},
  version = {0.1.0},
  license = {LGPL-3.0-or-later},
  url = {https://github.com/Microrobotics-Simulation-Framework/MIME}
}
```

## License

LGPL-3.0-or-later. See [LICENSE](LICENSE).
