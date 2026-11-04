# `examples/`

**Scripts as a user writes them**, each runnable on its own and each held by a test, so that none drifts from what
MIME does.

- `descriptions/ar4/`: the AR4 robot arm's description (`ar4.urdf`, MIT, with its `LICENSE`) and the manifest of its
  meshes beside it (`visuals.json`: records at Annin Robotics' repository at a pinned commit, never the meshes'
  bytes, which its licence lets no copy of be hosted). An experiment reading the URDF finds its meshes' records there;
  `mime visuals fetch examples/descriptions/ar4/` bakes them on this machine. Shipped in the sdist.

- `script_a.py`: script A's reproduction arm. A de Jongh FL-9 screw on the floor of a 1/4" tube of water, turned by
  a field rotating at 10 Hz and 1.2 mT, with every number the 0.2 line's run of A used, declared in the experiment
  with where it comes from. Run it with `python examples/script_a.py` (float32) or with `JAX_ENABLE_X64=1` (float64):
  it prints the build report, then the swim speed and the rotation rate over the second half of a 5 s run.
  `tests/differential/test_script_a.py` checks that it declares the same experiment as the test fixture
  (`tests/script_a.py`'s `reproduction_arm`), which is held to the frozen 0.2 reference.
- `sphere_settling.py`: a steel sphere settling in still water in free space, 10 um in radius, with its reasons and a
  comparison with Stokes's law. `python examples/sphere_settling.py` runs it in a few seconds (4 s as a whole process, imports and compile included, one run
  on a laptop at light load); `--radius-um 50` makes
  the sphere one whose Reynolds number is past the model's range, and the result flags it with the size of the inertia
  the model leaves out. `tests/examples/test_sphere_settling.py` runs it.
- `screw_in_tube.py`: a de Jongh FL-9 screw swimming along a tube of glycerol, turned by a field rotating at 1 Hz, with
  no material records: every value is declared, and the inertia is the diagonal a research build prints when it refuses the
  parts' own tensor (an exploratory build uses it and stamps it pending). `python examples/screw_in_tube.py` runs it in a few seconds more (7 s, measured the same way); `--spacing-mm` sets how fine the
  samples are, which `docs/choosing_spacing.md` prices and judges. It reads the tube's table, which a build never
  downloads: fetch it once first, `mime assets fetch unit_tube_R0.997`. `tests/examples/test_screw_in_tube.py` runs it.
- `tutorial_a.py` and `tutorial_a0.py`: the runs the user guide's tutorials print the output of
  ([tutorial A](../docs/guide/tutorial_a.md), [tutorial A0](../docs/guide/tutorial_a0.md)). The first builds script A's two
  arms from `script_a.py` and runs each for a short time; the second is A0, the same screw driven by a magnet on an AR4
  arm in a closed loop, with the arm's description from `tests/descriptions/`. `tests/docs/test_tutorials.py` runs both and
  holds the guide's output blocks to what they print; the guide includes their regions, and those of `script_a.py`.
