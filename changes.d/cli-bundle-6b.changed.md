- **CLI bundle 6b: one frame for every error, and fewer surprises.** A mistake in the command line (a missing or unknown
  argument, a mistyped command, a token typed where it does not belong) is a coded refusal, `cli.usage`, with what to do
  (the command's own `--help` and its example, and the nearest name), drawn as every refusal is; argparse's usage block
  and the credentials command's own frame are gone, and the exit code is still 2. `mime study`, `mime assets` and every
  other group named alone list their commands. `--traceback` shows the refusal first, then the traceback, then the
  refusal's one line again. `mime run --override` is checked against the saved document before anything loads: a path
  that names nothing, a value the thing does not have, and a value in the wrong units (a bare number included) are
  refused at once, and a declared value in the wrong units is refused once, not again as undeclared.
  `mime push <folder> --init` needs no `--to`; `mime push --defaults` is gone (`mime defaults` is the one spelling).
