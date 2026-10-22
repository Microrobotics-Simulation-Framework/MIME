# `docs/`

**Documentation that outlives the code it describes.** The 0.2 line's docs (algorithm guides, user guides,
regulatory notes, release notes) are preserved at the tag `pre-0.3`.

## What is here now

- [`bibliography.bib`](bibliography.bib): the references that docs cite, with Pandoc-style keys.
  `scripts/check_citations.py` checks that every citation resolves.
- [`validation/`](validation/): the known-anomalies registry.

## What comes back, and when

Docs return with the code they describe, in the same change. A node's algorithm guide arrives with the node. Its
implementation-mapping table links each term of the equation to the code, and `scripts/check_impl_mapping.py`
checks every link.
