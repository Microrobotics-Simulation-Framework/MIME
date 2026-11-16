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
classes are caught as before: ``BuildRefused`` and its subclasses, each error that was a ``ValueError``,
``TypeError``, ``KeyError`` or ``RuntimeError`` is still one, and an error from below that a public call translates
is still of its own class (``except OSError``, the script's own ``except Done``), the original kept as ``original``."""

from mime.build._refusals import BuildRefused, MimeError, Refusal, RefusalClass, RunFailed
from mime.errors._display import write_traceback

__all__ = ["BuildRefused", "MimeError", "Refusal", "RefusalClass", "RunFailed", "write_traceback"]
