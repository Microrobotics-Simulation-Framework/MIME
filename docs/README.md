# `docs/`

**Documentation that outlives the code it describes.** The 0.2 line's docs (algorithm guides, user guides,
regulatory notes, release notes) are preserved at the tag `pre-0.3`.

## What is here now

- [The README's quickstart](../README.md#quickstart): install, a first experiment that runs in seconds, reading its
  result and what is pending; it is run by a test, as written.
- [`choosing_spacing.md`](choosing_spacing.md): choosing the MRS rung's `spacing` and the base step `dt`,
  with the cost and accuracy measured on the two examples, and what each number is worth.
- [`bibliography.bib`](bibliography.bib): the references that docs cite, with Pandoc-style keys. `mime docs check`
  parses every citation of the built pages by Pandoc's grammar and checks that each resolves.
- [`concepts.md`](concepts.md): the one hand-written note on the page types (entry, rung, node, family, mechanism)
  and how they link.
- [`results.md`](results.md): reading a run's results, each observation's declared outputs by name with their
  units, series and windows by time, flags and the exports; printed among the pages beside the concepts note.
- [`profiles.md`](profiles.md): the profiles' path for a new user, exploring with nothing stated, seeing what is
  pending (`built.report.pending()`, `mime pending <folder>`) and sealing, with where a value's source goes, the roles
  and defaults, and each refusal on the way.
- [`platforms.md`](platforms.md): what each platform supports (Linux, macOS on Apple Silicon, native Windows and WSL2),
  what the smoke job tests on the last two and what nothing tests, where a large job or a GPU belongs, and that
  verification off Linux is within the lock's tolerances, by decision.
- [`running_on_modal.md`](running_on_modal.md): running a saved experiment on Modal, with the environment built once
  per release and pinned by hash, and what that does and does not let a run claim.
- [`validation/`](validation/): the known-anomalies registry.
- [`data/`](data/): records no code owns, such as the 0.2 line's frozen algorithm ids.
- [`ratchets/`](ratchets/): the lists `mime docs check` reads, one item per line with its reason, each only shrinking.
- [`unpaged.txt`](unpaged.txt): the ledger, each hand-written source that merged before the docs walk without it.
- [`number_idioms.txt`](number_idioms.txt): the digit forms the number rule exempts, each with its reason.

## What comes back, and when

The pages are generated from the code's own declarations (an entry's fields, a node's `NodeMeta` and symbol table,
the refusal table), and a small hand-written source beside each: `entries/<kind>/<name>.md` for an entry,
`mechanisms/<target kind>.md`, `assets/<name>.md`, and optional `nodes/` and `families/` notes. Generated pages are
never committed. `mime docs check` requires a hand-written source for everything of MIME's that needs one: anything
registered since the walk landed arrives with its source, and what merged before it is on the ledger, which only
shrinks as each item is paid. No hand-written file or package README holds a measured number: it names the claim
record that holds it (`see MIME-ANO-114`).

## Building the pages

```sh
uv run --no-sync mime docs build                    # to docs/_build/ (untracked, listed in .gitignore)
uv run --no-sync mime docs build --out <directory>  # index.md and pages/ under <directory>
```

`mime docs build` needs the `verify` extra. It writes `index.md` and `pages/`: an entry page for every registered entry
(MIME's, the toys' and every installed `mime.entries` package's, so a lab's entry has one with no work), a node page
for every node an entry builds, a family page for every registered family, a mechanism page for every target kind,
the vocabulary, and the concepts note. Each hand-written source in the directories above is printed inside its page.
A source whose entry, version or id no longer resolves fails the build, naming the file, and no page is written. The
pages hold no absolute path, and the same checkout gives the same text under the eight hash seeds `mime docs check` builds with, each in a fresh
process.

CI builds the pages, scans the built directory for a private name or path (`scripts/check_confidential.py
--directory`), and only then uploads it as a run artifact named `docs-pages-python-<version>`; download it from the
run page to read them. A page you change is read the same way, by building locally.
