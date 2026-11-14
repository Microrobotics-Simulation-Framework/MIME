"""MIME: microrobot experiments declared as scenes, built into MADDENING graphs, on OpenMaterica's materials."""

__version__ = "0.3.0.dev1"

# Ctrl+C while the `mime` command imports the rest of MIME ends it quietly (mime/_import_guard.py): first, because the
# imports below are what takes the time.
from mime import _import_guard  # noqa: E402

_import_guard.guard()

from mime import units  # noqa: E402
from mime.errors import install as _install_errors  # noqa: E402
from mime.experiment import Experiment  # noqa: E402
from mime.study import Study  # noqa: E402
from mime.scene import Scene, asset  # noqa: E402
from mime.things import Apparatus, Body, CommandPort, Measurement, Medium, PermanentMagnet, State, Structure  # noqa: E402

# MIME's own entries register when MIME is imported: a load finds them with no entry point.
from mime import drives, dynamics, fields, interactions, observations, scene, walls  # noqa: E402, I001

# A MIME error no one catches is shown with the script's own frames (``MIME_TRACEBACK=full`` shows every one).
_install_errors()

__all__ = ["Apparatus", "Body", "CommandPort", "Experiment", "Measurement", "Medium", "PermanentMagnet", "Scene", "State",
           "Structure", "Study", "asset", "drives", "dynamics", "fields", "interactions", "observations", "scene", "units",
           "walls"]
