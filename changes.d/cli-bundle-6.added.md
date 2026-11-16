- **CLI bundle 6, found by walking the CLI end to end.** `mime push <folder> --to <service> --dry-run` shows the plan push
  shows before it sends anything (what would be uploaded, and where) and sends nothing: it reads no token and opens no
  connection. `mime show <run folder>` prints the run's outputs summary (`--json` the manifest it was read from), where it was
  refused with no hint. `mime env check --strict` holds every pin exactly. `mime modal --force` replaces the files of a script
  written earlier.
