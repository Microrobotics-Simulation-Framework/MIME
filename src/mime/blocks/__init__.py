"""Blocks: published, versioned parts of an experiment that another lab reuses (an apparatus, a vessel network, a
vendor's probe model), written as data in the saved document's vocabulary, never code.

A **block folder** holds ``mime.toml`` (its definition: ``[block]`` with its id, version and licence, ``[requires]``,
``[sockets]``, ``[[assets]]`` and the rest, and ``[record]``, its record's metadata, which no container carries),
``block.mime.yaml`` (the fragment, ``mime.fragment/1``), ``records/`` and ``descriptions/`` (the material records and
descriptions the fragment names by sha256), ``assets/`` (the author's copies of open or conditional asset bytes) and
``cases/`` (saved folders whose experiment uses the block). Its **content hash** names its definition and content files
(:mod:`mime.blocks._definition`); its **tier** is its licence's verdict: open or conditional is copied into the
document that uses it, any other referenced.

``exp.use(<block folder>, <socket>=...)`` (:meth:`mime.experiment.Experiment.use`) uses one; ``mime show`` and ``mime
check`` read one offline; :func:`fragment_from` writes a fragment from a saved experiment, the named things its
sockets."""

from mime.blocks._definition import (
    FRAGMENT_FILE,
    MANIFEST_FILE,
    SOCKET_KINDS,
    Definition,
    content_hash,
    definition_of,
    is_block_folder,
    read_definition,
)
from mime.blocks._fragment import Fragment, read_fragment
from mime.blocks._schema import FRAGMENT_SCHEMA, FragmentDoc
from mime.blocks._from import fragment_from
from mime.blocks._use import BlockHandle, BlockRefused, requirement_rows

__all__ = ["FRAGMENT_FILE", "FRAGMENT_SCHEMA", "MANIFEST_FILE", "SOCKET_KINDS", "BlockHandle", "BlockRefused",
           "Definition", "Fragment", "FragmentDoc", "content_hash", "definition_of", "fragment_from",
           "is_block_folder", "read_definition", "read_fragment", "requirement_rows"]
