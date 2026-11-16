"""``mime push``: a saved experiment as a draft record on Zenodo, and nothing published.

``mime push <folder> --to zenodo|zenodo-sandbox`` shows the plan, computed offline (every record file, every file inside
the bundle with its tier and verdict, every file that travels as a reference, each declared permission in words, the
metadata), then, confirmed, makes or updates the experiment's draft, reserves its DOI and uploads its files through
InvenioRDM's records API, each md5 checked against the service's. ``--init`` writes ``mime.toml`` (the record's
manifest) from what the folder knows; ``--status`` reads the draft as the service holds it; ``--discard`` deletes it
(only a draft can be). ``--to`` is required: the sandbox and production are two services, each fixed in code, never
one flag apart by accident. A push stopped at any request is resumed by running it again: what was uploaded and
verified is not sent again. Push writes to, or deletes, only the draft it made (it marks each with an identifier of its
own), and never writes over metadata edited on the site unasked: each such edit is kept (written into
``mime.toml``'s ``[record]``) or written over, field by field. ``--publish`` publishes the draft, the one irreversible step, after
the plan, the confirmation in words, each declared permission asked on its own and the record number typed (or
``--confirm <DOI>`` from a script); then checks the record as published and, on production, writes the staged lock
into the folder, so ``mime pack <folder>`` makes the published bundle again. A changed folder's next push is a new
version under the same concept DOI. See the README.
"""

from mime.push._check import Checked, check
from mime.push._manifest import MANIFEST_FILE, MANIFEST_FORMAT, Manifest, read_manifest

__all__ = ["MANIFEST_FILE", "MANIFEST_FORMAT", "Checked", "Manifest", "check", "read_manifest"]
