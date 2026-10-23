# `examples/`

**Scripts as a user writes them**, each runnable on its own and each held by a test, so that none drifts from what
MIME does.

- `script_a.py`: script A's reproduction arm. A de Jongh FL-9 screw on the floor of a 1/4" tube of water, turned by
  a field rotating at 10 Hz and 1.2 mT, with every number the 0.2 line's run of A used, declared in the experiment
  with where it comes from. Run it with `python examples/script_a.py` (float32) or with `JAX_ENABLE_X64=1` (float64):
  it prints the build report, then the swim speed and the rotation rate over the second half of a 5 s run.
  `tests/differential/test_script_a.py` checks that it declares the same experiment as the test fixture
  (`tests/script_a.py`'s `reproduction_arm`), which is held to the frozen 0.2 reference.
