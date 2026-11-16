"""The documentation's machine-readable sources and their checks, run by ``mime docs check`` and ``mime docs build`` and by
``mime registry --json`` (the export the pages are generated from) over a checkout; ``mime docs check`` needs the ``verify``
extra, and nothing imported here needs it, so ``mime registry --json`` runs on a base install.

What is public is the evidence resolver's reading of a record's claims: :func:`claim` hands a test the number a record
states, as a :class:`Claim`, so that it is held once, in the record (``docs/validation/README.md``). The rest (the
registry export, the checks on a node's ``NodeMeta`` and symbol table, the refusal scan the refusal table is generated
from, the ratchets kept as lists, the pages) is the commands' own."""

from mime.docs._evidence import Claim, claim

__all__ = ["Claim", "claim"]
