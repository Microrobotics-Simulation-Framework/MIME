# Spike result: does the exact allocator's tie window make a loop chatter between faces?

Branch `spike/tie-chatter` (no PR, no `src/` change). Predictions: `scripts/spikes/tie_chatter/predictions.json`, committed
before the harness existed, and `predictions_last64.json` before the second launch. Runs: GitHub runners, CPU, probe
branch `probe/tie-chatter` (since deleted), runs 43809934591 (six arms) and 43811465603 (`last64`); each arm's peak
0.54 GiB against a bound of 0.55 (no UNDERESTIMATE). Raw per-arm JSON: `scripts/spikes/tie_chatter/results/`.

## What was measured, and what it bounds

The allocator called once a tick (`lax.scan`, the dtype's own arithmetic) on a **given** target path; there is no plant and
no feedback, so this is the allocator's behaviour along a slowly moving target, not a closed loop with a plant. It bounds
what a controller whose target moves slowly sees from the allocator alone. Targets are rounded once to the dtype, as a caller
would hand them.

- **Paths**, `u(k) = centre + amp sin(2 pi k / T + phi)`, target `w0 e^u`: OctoMag at the centre of its region, kind
  `both`, 5 A, field along y at `e^u` times the 5 A reach (draw 1 is `u = 1`); `u` in [-0.4, 2.4]. CardioMag at the centre,
  the second draw's wanted vector times `e^u`, `u` in [-0.3, 0.3]. Fast = 3 periods (OctoMag T = 9000, CardioMag 2000),
  slow = 1 period (90000, 20000); the target's relative move per tick is at most 9.4e-4 (fast) and 9.4e-5 (slow), under
  the **tolerance 1e-3** of `|target|`. Two phases: start at the bottom (the optimum face's side) and at the top (inside
  the tied band). A third, **not F0's, problem** `octo_field` (OctoMag, the field's 3 rows only: non-unique, where the
  window's least-effort tie-break matters), `u` in [-0.5, 0.5].
- **Chatter** = a tick whose largest command jump exceeds the threshold while the target moved less than the tolerance.
  **1 A** is a fifth of the 5 A limit and a tenth of the 10 A box: legitimate motion on these paths is 0.005 to 0.01 A a
  tick at the fast speed (5 A x 1e-3 x gain), float32's rounding of a 5 A command is 6e-7 A, and the faces here are 10 A
  apart. A second threshold, 0.03 A, is read only where noted (CardioMag's faces are 0.05 to 0.06 A apart).
- **Controls.** Float64 (all arms). `w64x`, float64 with the window widened to float32's absolute width (64 x 2^29): the
  noise-free control of the edge crossings. A planted law alternating faces every tick, through the same counter, 20
  random paths and jump sizes: caught on every tick in 20 of 20 (and in every one of the six arms' jobs); a 0.9 A
  alternation is not counted at 1 A (20/20) and is counted at 0.03 A (20/20); a target moving 5 % a tick is not counted
  (20/20). The arm's variant was recorded as traced in every job (`plain 16/64/128`, `sticky 64`, `plain 2^35`).

## Counts (switches at 1 A per run; bottom-start / top-start; fast, then slow)

| arm | OctoMag float32 fast | slow | OctoMag float64 | CardioMag, both dtypes | octo_field float32 (float64) |
|---|---|---|---|---|---|
| TIE_ULPS 0 | 0 / 0 | 0 / 0 | 0 | 0 | **11489 / 11489** fast, 38177 slow (11576, 38484) |
| 16 | 6 / 6 | **170** / 170 | 0 | 0 | 0 |
| 64 (shipped) | 18 / 18 | 70 / 70 | 0 | 0 | 0 |
| 128 | 12 / 12 | 16 / 16 | 0 | 0 | 0 |
| sticky 64 (prototype) | 0 / 1 | 0 / 1 | 0 | 0 | 0 |
| last64 (`reference=` the previous commands) | 0 / 1 | 0 / 1 | 0 | 0 | 0 |
| w64x (float64, window as wide as float32's) | - | - | 6 / 6 fast, 2 / 2 slow | 0 | 0 |

Where the switches are (OctoMag float32, `u`): TIE64 at 0.58 to 0.59; TIE16 at 1.93 to 1.97; TIE128 at 0.19 and 2.145. Each
is a 10 A jump (command 0 between -5 and +5 A, all eight coils at a bound) at a target move under 0.1 %.

**Reading.**
- The chatter is real in float32 on draw 1 at every non-zero window: the window's edge is crossed twice a period, and
  **rounding flickers on top of the crossings**. `w64x` shows the crossings alone: exactly 2 per period (6 fast, 2 slow) and
  no flicker, so the excess (TIE64: 12 fast, 68 slow) is float32 rounding at the window's edge, and it grows as the ticks
  get finer (slow TIE16: 170 a run). Float64 does not chatter in any F0 arm (control held).
- Moving the constant moves the edge; it does not remove it: TIE16's edge is at `u` 1.97, TIE128's at 0.19 and 2.145 (a
  third face enters the tie as the window grows with `|target|`). Any non-zero window has an edge where some pair's gap
  equals it.
- **TIE_ULPS 0 removes it on both draws and is not an option:** on `octo_field` it flips faces on about 96 % of ticks
  (jumps over 1 A) in float32 and in float64, because exactly tied faces then differ only by rounding. The window is what
  holds the least-effort rule there.
- **CardioMag draw: F0's 1 A definition cannot fire.** No arm, no dtype, reaches 1 A (its faces differ by 0.05 to 0.06 A).
  At 0.03 A the fast path is contaminated by legitimate motion (float64 itself counts 306 a phase), so read the slow
  path: float64 0; TIE0 0; TIE16 4, TIE64 6, TIE128 6, w64x 6, **sticky64 11**, last64 9 (a 0.05 A hold at the bound, then a
  forced release). Sticky does not help, and slightly worsens, this small-jump effect; it is below F0's chatter threshold.
- Float32 TIE0 on OctoMag shows 48 (fast) and 104 (slow) switches of 0.03 to 1 A near `u` 0.14 to 0.18, where command 0
  leaves its bound (adjacent faces, rounding); none reach 1 A.

## Predictions

Held: float64 quiet in every F0 arm; `w64x` exactly 6 and 2 with no flicker; TIE0 0 on the draws; TIE64 and TIE128 within
bands, TIE16 fast at the band's floor (6); TIE64's edge (predicted `u` 0.57, found 0.58 to 0.59), TIE16's (1.94, found
1.97); sticky 0 from the optimum face and 1 from the tied band; no 1 A switch on CardioMag; TIE0 chatters on `octo_field`;
the controls caught 20/20.
**Missed (recorded as missed):**
- TIE16 slow: 170 against a band of 2 to 150 (flicker grows faster with finer ticks than I estimated).
- TIE128's edges: predicted one at `u` about -0.1; found 0.19 (where the face structure changes) and 2.145 (a third face).
- `octo_field` TIE0: band [1, 200]; found about 11500 a fast run.
- CardioMag at 0.03 A: float64 and TIE0/TIE16 predicted 0 on the fast path; float64 counts 306 (I took the legitimate move per
  tick from a coarser scan: the true gain is higher where commands leave bounds); TIE16 slow predicted 0, found 4; sticky
  predicted 0 to 1, found 11 slow.
- The `last64` arm was added after the first read (its prediction, 0 / 1 switches, held).

## The smallest change that removes chatter

On F0's two draws, with F0's 1 A definition: **hysteresis**, and the smallest form of it already exists: passing the
previous commands as the law's `reference=` (what the Allocator passes for `toward='last'`) at the unchanged window 64
counts 0 / 1 on the same paths as the sticky prototype (the one switch is the forced release when a run starts inside the
tied band). It needs no state seam and no `src/` change, for a caller that already carries `last`. Its limits: it replaces
the least-effort tie-break by least-distance-to-last (the same on `octo_field`, 0 switches, but a different minimiser where
faces tie), the first tick's reference is zero, and a controller that does not carry the commands it wrote last gets nothing.
A narrower constant (16) or a wider one (128) does not remove it; 0 removes it on the draws and destroys the non-unique case.
The sticky prototype (within the window, keep the previous face) is the explicit form: same counts, independent of the
reference.

## What the sticky variant would cost (from the code; not measured beyond the counts above)

- **State seam.** `ExactAllocation` gains the selected set's index (an int32, already computed as `selected` inside
  `_solved`); `exact_allocation` and `Allocator.solve` take `previous` (-1 for none); the controller carries one int32 per
  allocator beside its `last` commands (initial -1, reset with the controller, per batch element under `vmap`). A saved form
  gains one leaf. Inside `_selection` it is three lines before the last argmin (the prototype is that textual insert; the
  real one would use `switch.predicate` for the held-face test).
- **Suboptimality.** The held face is inside the tied set, so its residual is at most `residual[best] + width[best] +
  width[held]`, `width = 64 eps (|target| + |made|)`: at draw 1, 0.0095 in residual norm of 610.9 (1.6e-5 relative, about 3e-5
  of the objective), which is the same worst case the shipped law already admits; sticky does not widen it, it removes the
  alternation. History-dependent: a run that starts in the band holds the tied face (0.0062 above the optimum) until the
  band is left. Inert in float64 (a window 5e8 times too narrow to reach any gap here): 0 switches measured.
- **Gradients.** Selection is under `stop_gradient` (the derivative is the selected set's solve by one QR, or the free
  columns' pseudo-inverse, in this file; there is no `custom_root` in `src/mime/control`), so autodiff is unchanged in form.
  At a tie the derivative is that of the face held, not the lower-effort one: at draw 1 both faces have every command at a
  bound, so both derivatives are zero and nothing moves; in general it is a history-dependent choice among valid
  one-sided derivatives. The previous face is an int leaf (no tangent or cotangent), so a differentiated rollout through
  the controller gains no gradient path and no new term, only the held selection. The switch declaration
  (`allocation_switch`) would need one more indicator, the held face's margin to the window's edge, resolution `width`;
  the declared orders stay (the jump moves from the residual crossing to the window's exit). The held-face test is not
  differentiated and not checked by the derivative tests as they stand: a plant for it would need adding.
- **Not measured:** a closed loop with a plant and feedback (a held face's suboptimality could interact with the loop); a
  hold at a bound on CardioMag-like faces (the 0.03 A count rose, 11 against 6); a third non-unique draw with the sticky
  arm in float32 beyond `octo_field` (0 switches); the effect on the allocator's property tests (the oracle's tie
  window accepts a held face by construction, but its detection rates were not re-measured).
