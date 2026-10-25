# `docs/`

**Documentation that outlives the code it describes.** The 0.2 line's docs (algorithm guides, user guides,
regulatory notes, release notes) are preserved at the tag `pre-0.3`.

## What is here now

- [`bibliography.bib`](bibliography.bib): the references that docs cite, with Pandoc-style keys.
  `scripts/check_citations.py` checks that every citation resolves.
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
