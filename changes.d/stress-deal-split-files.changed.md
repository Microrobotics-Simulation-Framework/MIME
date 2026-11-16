- The stress lane deals a test file the shard table splits (`split` in `tests/shard_weights.json`) across its parts by test,
  by the table's measured seconds, as the default lane's shards do, so a heavy file (`tests/sensing/test_crease.py`,
  about 48 minutes whole) fits a job's 20 minutes. A part is given node ids, or the file with `--deselect` for the listed
  tests other parts run, which also carries the file's tests the table does not list; a listed name that is a prefix of
  another test's stays with the file. Every test still runs in every run, once; files without a split are dealt whole as
  before, and `stress.yml` is unchanged.
