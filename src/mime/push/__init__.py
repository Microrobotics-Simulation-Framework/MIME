"""``mime push``: a saved experiment as a draft record on Zenodo, and nothing published.

``mime push <folder> --to zenodo|zenodo-sandbox`` shows the plan, computed offline (every record file, every file inside
the bundle with its tier and verdict, every file that travels as a reference, each declared permission in words, the
metadata), then, confirmed, makes or updates the experiment's draft, reserves its DOI and uploads its files through
InvenioRDM's records API, each md5 checked against the service's. ``--init`` writes ``sharing.yaml`` (the record's
credit) from what the folder knows; ``--status`` reads the draft as the service holds it; ``--discard`` deletes it
(only a draft can be). ``--to`` is required: the sandbox and production are two services, each fixed in code, never
one flag apart by accident. A push stopped at any request is resumed by running it again: what was uploaded and
verified is not sent again. Publishing, versions and communities come later. See the README.
"""

from mime.push._client import (
    LONGEST_WAIT_SECONDS,
    PAUSE_SECONDS,
    RDM_TYPE,
    RETRIES,
    Answer,
    Client,
    Dropped,
    Errors,
    Gone,
    HttpTransport,
    Transport,
)
from mime.push._draft import UPLOAD_ATTEMPTS, Drafted, discard, init, push, send, status
from mime.push._plan import KEYWORD, MOST_BYTES, MOST_FILES, AssetChoice, Plan, RecordFile, card_html, make_plan
from mime.push._service import SERVICES, TEST_PREFIX, ZENODO, ZENODO_SANDBOX, Service
from mime.push._sharing import (
    CONTRIBUTOR_TYPES,
    DEFAULT_LICENCE,
    RELATION_TYPES,
    SHARING_FILE,
    SHARING_FORMAT,
    Person,
    Related,
    Sharing,
    Uploader,
    orcid_problem,
    read_sharing,
)
from mime.push._state import DraftState, Uploaded, read_state, state_directory

__all__ = [
    "CONTRIBUTOR_TYPES",
    "DEFAULT_LICENCE",
    "KEYWORD",
    "LONGEST_WAIT_SECONDS",
    "MOST_BYTES",
    "MOST_FILES",
    "PAUSE_SECONDS",
    "RDM_TYPE",
    "RELATION_TYPES",
    "RETRIES",
    "SERVICES",
    "SHARING_FILE",
    "SHARING_FORMAT",
    "TEST_PREFIX",
    "UPLOAD_ATTEMPTS",
    "ZENODO",
    "ZENODO_SANDBOX",
    "Answer",
    "AssetChoice",
    "Client",
    "DraftState",
    "Drafted",
    "Dropped",
    "Errors",
    "Gone",
    "HttpTransport",
    "Person",
    "Plan",
    "RecordFile",
    "Related",
    "Service",
    "Sharing",
    "Transport",
    "Uploaded",
    "Uploader",
    "card_html",
    "discard",
    "init",
    "make_plan",
    "orcid_problem",
    "push",
    "read_sharing",
    "read_state",
    "send",
    "state_directory",
    "status",
]
