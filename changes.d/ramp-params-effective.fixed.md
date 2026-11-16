- **`drive.ramp`'s node verification no longer fails about one run in 400.** MADDENING's `params_effective` found a zero
  gradient on `hold` in a run whose 200 drawn examples all fell where the hold has no effect: it acts only where the
  step's middle falls inside the ramp, and a step counter drawn over all of int32 with a timestep from 0.1 to 10 ms put
  about one example in twenty there. The entry's node case now folds three in four drawn step counters into the first
  21 steps (about one example in eight inside the ramp), and keeps the rest of the draw, int32 extremes included. The
  check is unchanged, and a planted read of `hold` from the constructed value still fails every run.
