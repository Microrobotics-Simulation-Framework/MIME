# `mime.auth`

**The access token `mime push` will use, kept where a commit cannot publish it and shown nowhere.** `push` is not built
yet; this is its credentials, prepared. Nothing here goes online: a check of the token against the service's API is
`push`'s.

## Commands

`mime auth login | logout | status <service>`, for `zenodo` and `zenodo-sandbox`.

- **login** reads the token with `getpass` (nothing is echoed). It is never an argument of the command: a shell keeps
  its arguments in its history and `ps` shows them. A value that is not one word of at least eight characters is
  refused, naming neither the value nor any part of it.
- **status** says whether there is a token, where it comes from, and its last four characters. Nothing else.
- **logout** deletes the token from the file, and the file when it was the last. A token in the environment is the
  environment's to unset, and `logout` says when one is still there.

## Where it lives

- **The environment variable overrides the file:** `ZENODO_TOKEN` and `ZENODO_SANDBOX_TOKEN`. A script or a cluster job
  sets the variable and stores nothing.
- **The file** is `credentials` in the user's config directory (`platformdirs`: `~/.config/mime/` on Linux), a JSON
  object of service to token. The directory is owner-only and so is the file. It is written through a temporary name
  that was created owner-only, then renamed into place, so another user can never read it, not even for a moment.
- **Never inside a git working tree.** A login whose config directory is in a repository (a `git add .` would publish
  it) is refused as `auth.in-git-tree`, and nothing is written. A file other users can read is refused as
  `auth.file-mode` when a token is wanted, and `status` says so: the rule `ssh` keeps for a private key.
- On Windows the modes mean nothing and are not set; the file is in the user's profile.
- **The operating system's keyring is optional and not built here.** It would be one more place to read a token from,
  after the environment and before the file.

## Kept out of everywhere else

A token is a `Token`: its `repr`, `str` and `format` say the service and nothing of the value, and it cannot be
pickled, so a token that reaches a log line, an exception message, a traceback or a document shows no secret. The value
comes out through `reveal()` alone, which the code that sends it to the service calls. No message of this package
quotes a value, including the one for a file it cannot read.

`mime pack` searches every file it would carry for the exact value of every token held (the environment's and the
file's; a value shorter than the minimum is not a token and is left out) and refuses the bundle, naming the file and
never the token.

## Status

Landed with `mime pack` and `mime pull`, ahead of `mime push`.

Phases and releases are explained in the [package manifest](../README.md#status).
