"""MIME's errors at the boundary of its public calls.

Every error a public call raises (``Experiment.build``, a run, ``save``, ``load``, ``verify``, a result's accessors,
the CLI) is a :class:`MimeError`: one or more refusals, each with a code and its class (``user``: what to change;
``environment``: an install or a device; ``internal``: a MIME defect, reported with its file; ``unrecognised``: an
error from below that MIME has no translation for, which is never called MIME's defect). Each refusal says what
happened, in the experiment's names; why; what to do, pasteable where it can be; where, at the script's line; and its
code, which names its page.

An uncaught MIME error is shown with the script's own frames and none of MIME's, JAX's or MADDENING's. The full
chain is kept on the error (``chain``) and written to a file when it is shown (``traceback_file``);
``MIME_TRACEBACK=full`` shows it in place of the short form, and the CLI's ``--traceback`` does for one command. The
classes are caught as before: ``BuildRefused`` and its subclasses, and each error that was a ``ValueError``,
``TypeError``, ``KeyError`` or ``RuntimeError`` is still one."""

from mime.build._refusals import (
    ArgumentRefused,
    ArgumentTypeRefused,
    BuildRefused,
    InternalError,
    MimeError,
    Refusal,
    RefusalClass,
    RunFailed,
    UnrecognisedError,
    page_of,
)
from mime.errors._boundary import crossed, full_chain, public
from mime.errors._display import (
    DIRECTORY_VARIABLE,
    KEPT,
    TRACEBACK_VARIABLE,
    full_mode,
    install,
    short_text,
    traceback_directory,
    write_traceback,
)
from mime.errors._frames import HIDDEN_PACKAGES, Place, caller_place, is_library, script_places, script_traceback
from mime.errors._translate import first_lines, script_place, traced_value_error, translated

__all__ = [
    "DIRECTORY_VARIABLE",
    "HIDDEN_PACKAGES",
    "KEPT",
    "TRACEBACK_VARIABLE",
    "ArgumentRefused",
    "ArgumentTypeRefused",
    "BuildRefused",
    "InternalError",
    "MimeError",
    "Place",
    "Refusal",
    "RefusalClass",
    "RunFailed",
    "UnrecognisedError",
    "caller_place",
    "crossed",
    "first_lines",
    "full_chain",
    "full_mode",
    "install",
    "is_library",
    "page_of",
    "public",
    "script_place",
    "script_places",
    "script_traceback",
    "short_text",
    "traceback_directory",
    "traced_value_error",
    "translated",
    "write_traceback",
]
