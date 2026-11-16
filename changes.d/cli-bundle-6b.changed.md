- **CLI bundle 6b: one frame for every error, and the tour's smaller findings.**
  A mistake in the command line (a missing or unknown argument, a mistyped command, a token typed where it does not
  belong) is a coded refusal, `cli.usage`, with what to do (the command's own `--help` and its example, and the nearest
  name, never a flag the line already has; a removed flag says what replaces it), drawn as every refusal is; argparse's usage block and the credentials command's own frame are gone, and the exit
  code is still 2. `mime study`, `mime assets` and every other group named alone list their commands. `--traceback` shows
  the refusal first, then the traceback, then the refusal's one line again. `mime run --override` is checked against the
  saved document before anything loads (a path that names nothing, a value the thing does not have, a value in the wrong
  units, a bare number included), and a declared value in the wrong units is refused once, not again as undeclared.
  `mime push <folder> --init` needs no `--to`; `mime push --defaults` is gone (`mime defaults` is the one spelling, and the refusal says so).
  `mime assets check` always ends with a total. `mime visuals list` says how to get a set when there is none, and knows
  the AR4 arm's mesh manifest, shipped as references only (names, hashes, sizes, upstream addresses and terms; no mesh
  bytes). The GPU precision refusal of `mime modal --gpu` reads "an A100-40GB" and says how to save for a GPU.
  `mime scaffold --yes` chooses no licence and no namespace: where none is given by `--set` or the defaults it writes a
  marked placeholder (`LicenseRef-CHOOSE-A-LICENCE`, `choose_a_namespace`) that every check refuses until it is replaced:
  a bundle's `mime.toml` and a plugin's example, benchmark manifest and record carry it too.
  A repeated `mime study run` says how many members it skipped and why any are run again. `mime verify` ends with an
  "An acceptance" paragraph for a saved experiment and a run folder as it does for a study and a block. Every command's
  `--help` lists its exit codes. `mime check` and `mime pack` write a path below the current directory relative to it.
  `mime registry` has a header row and a `from` column (`mime` or the plugin's namespace). A plain `mime env check`
  names an installed plugin that nothing pins as a plugin in its notes; a strict check (`--strict`, or a folder,
  `--base` or `--mime-version`) refuses it, as any distribution nothing pins.
