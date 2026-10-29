# `mime.credentials`

**The access token `mime push` will use, kept where a commit cannot publish it and shown nowhere.** `push` is not built
yet; this is its credentials, prepared. Nothing here goes online: a check of the token against the service's API is
`push`'s. MIME has no accounts, so nothing is "logged into": it stores another service's credential.

## Commands

`mime credentials set | status | remove <service>`, for `zenodo` and `zenodo-sandbox`.

- **set** reads the token with `getpass` (nothing is echoed). It is never an argument of the command: a shell keeps
  its arguments in its history and `ps` shows them. A token typed as an argument, or in the place of the service, is
  refused by a message that quotes no word of the command line (argparse's own error would print it). A value that is
  not one word of at least eight characters is refused, naming neither the value nor any part of it.
- **status** says whether there is a token, where it comes from, and the last four characters of a token of thirty-two
  characters or more. A shorter token shows nothing of itself: four characters of a short token are a large part of it. Nothing else.
- **remove** deletes the token from the file, and the file when it was the last. A token in the environment is the
  environment's to unset, and `remove` says when one is still there.

## Where it lives

- **The environment variable overrides the file:** `ZENODO_TOKEN` and `ZENODO_SANDBOX_TOKEN`. A script or a cluster job
  sets the variable and stores nothing.
- **The file** is `credentials` in the user's config directory (`platformdirs`: `~/.config/mime/` on Linux), a JSON
  object of service to token. The directory is owner-only and so is the file. It is written through a temporary name
  that was created owner-only, then renamed into place, so another user can never read it, not even for a moment.
- **Never inside a git working tree.** A `set` whose config directory is in a repository (a `git add .` would publish
  it) is refused as `credentials.in-git-tree`, and nothing is written. The tree is found three ways: the nearest
  ancestor holding a `.git`; the `GIT_WORK_TREE` the environment names (a dotfiles repository used as `git
  --git-dir=... --work-tree=$HOME` has no `.git` below its work tree); and what `git rev-parse --is-inside-work-tree`
  says from the config directory, with the user's own git configuration and environment (a `GIT_DIR`, a
  `core.worktree`). A repository reached only through an alias that passes `--git-dir` and `--work-tree` on each
  call leaves no trace in the environment or on the disk for any of the three to find. A file other users can read is
  refused as `credentials.file-mode` when a token is wanted, and `status` says so: the rule `ssh` keeps for a private key.
- On Windows the modes mean nothing and are not set; the file is in the user's profile.
- **The operating system's keyring is optional and not built here.** It would be one more place to read a token from,
  after the environment and before the file.

## Kept out of everywhere else

A token is a `Token` from the moment it is read: `credentials.prompt` wraps what `getpass` returns in the same
expression, and the file is read straight into tokens, so no frame holds the text under a name of its own. Its `repr`,
`str` and `format` say the service and nothing of the value, and it cannot be pickled, so a token that reaches a log
line, an exception message, a document or **a traceback printed with its locals** (`pytest --showlocals`, `rich`,
`traceback.TracebackException(capture_locals=True)`) shows no secret. The value comes out through `reveal()` alone,
which the line that writes the file calls. No message of this package quotes a value, including the one for a file it
cannot read. A test captures tracebacks with their locals at every refusal and at a failure past the prompt, and
searches them.

`mime pack` searches every file it would carry for every token held (the environment's and the file's; a value shorter
than the minimum is not a token and is left out), through `Token.occurs_in`, which never returns the text. It finds the
token as written, percent-encoded, as JSON writes it (with and without `\u` escapes), in base64 (either alphabet, at
any offset inside a larger blob), in hex (either case) and as two-byte Unicode text, and inside a gzip stream or a compressed zip
member. It cannot find every encoding: a token run through a cipher, reversed, split across lines or upper-cased
passes. That is why `pack` carries only the files a saved folder names (see [the saved form](../saved/README.md#the-bundle)):
the scan is the second line, and the whitelist is the first.

## Status

Landed with `mime pack` and `mime pull`, ahead of `mime push`.

Phases and releases are explained in the [package manifest](../README.md#status).
