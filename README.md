# MIME

MIcrorobotics Multiphysics Engine, on [MADDENING](https://microrobotica.org/maddening/).

**MIME 0.3 is being rebuilt on the `v0.3` branch.** So far it holds the core that turns a declared experiment into a
MADDENING graph, tested on toy entries with no physics; the physics arrives phase by phase.

The 0.2 line (the last release on MADDENING 0.3.x, with its nodes, experiments, docs and verification
scripts) is preserved at the tag [`pre-0.3`](https://github.com/Microrobotics-Simulation-Framework/MIME/tree/pre-0.3).

## Where things are

Every folder has a README saying what it is for.

- [`src/mime/`](src/mime/): the package. Its [README](src/mime/README.md) is the manifest: what MIME 0.3 is, how a
  declared experiment becomes a MADDENING graph, the dependency direction, each subpackage's role, and the status of
  the rewrite.
- [`tests/`](tests/): MIME's own tests, and the planned test layout.
- [`lab_fixture/`](lab_fixture/): a lab's package in miniature, which registers an entry from outside MIME.
- [`docs/`](docs/): the bibliography and the known-anomalies registry. Other docs return with the code they describe.
- [`scripts/`](scripts/): the compliance checks CI runs.

MIME is research software. It is not a medical device and is not intended for clinical use.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Releases of the 0.2 line (the tag `pre-0.3` and earlier)
keep the licence they were distributed under, which [NOTICE](NOTICE) names.
