# `tests/`

**MIME's own tests.** Users write theirs with the harnesses in [`mime.testing`](../src/mime/testing/).

## What is here now

- `test_package.py`: the package layout matches the design. Every subpackage imports and states its role, the version
  is the distribution's, and `mime.testing` names its extra when Hypothesis is missing.
- `test_ci_lane.py`: each CI lane installed what it claims. It reads the distributions' own metadata. For example,
  the OpenMaterica-`main` lane must have OpenMaterica from git, not PyPI. A lane that doesn't name itself fails, so no
  lane can skip the check. The test also runs its own control: claims meant for other lanes must fail in this one.
- `test_canary.py`: CI's planted failure. A manual run with `plant_failure` set must turn every lane red, which
  proves each lane really runs its tests.

## The planned layout

| Folder | What it will hold |
|---|---|
| `unit/` | the vocabulary, units, the part resolver, frames |
| `build/` | the canonical order and its controls; graph snapshots as golden text; shuffled declarations that must give bitwise-identical runs; a mutation test per post-build check |
| `families/` | one conformance suite per family: hydrodynamic, magnetic, acoustic radiation, gravity, contact |
| `differential/` | comparisons with frozen outputs of the 0.2 code, with tolerances declared before the first run |
| `oracles/` | analytic solutions; small physics oracles per interaction; software oracles, one lane each |
| `acceptance/` | comparisons with measurements, each stating what it bounds and what it estimates |
| `gradients/` | gradient checks that rebuild the graph, with planted failures as controls |
| `saved/` | a round trip for every registered entry and kind of thing; the load checks |
| `scene/` | the stage writer, frames, orientation lines |

## Markers and lanes

- `slow` tests are too expensive for CI's default lane. Run them with `-m "slow or not slow"`.
- `gpu` tests need a CUDA device and **skip without one**. A skip looks like a pass in a summary line, so count skips.
- CI's lanes are: the default (Python 3.12 to 3.14, released dependencies), OpenMaterica `main`, MADDENING `main`, and
  a scheduled run against the newest usd-core. The last three report without blocking.

Run the tests locally with `JAX_PLATFORMS=cpu python -m pytest`.
