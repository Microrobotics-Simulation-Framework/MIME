"""``mime.toml``, the record's manifest, named once: its file, its format and the family of that format's ids, the tables
only a block folder's holds, and the file it replaced. A saved folder, a study folder and a block folder each keep one,
and ``mime.push``, ``mime.blocks``, ``mime.results`` and ``mime.saved`` all read these names from here."""

MANIFEST_FILE = "mime.toml"
MANIFEST_FORMAT = "mime.toml/1"
MANIFEST_FAMILY = "toml"
#: The record's metadata before ``mime.toml``, which this MIME no longer reads.
OLD_FILE = "sharing.yaml"
#: A block's tables, besides ``format`` and ``[record]``.
BLOCK_TABLES = ("block", "requires", "sockets", "assets", "licences", "permissions", "validity", "cases")

__all__ = ["BLOCK_TABLES", "MANIFEST_FAMILY", "MANIFEST_FILE", "MANIFEST_FORMAT", "OLD_FILE"]
