"""The boundary of MIME's public calls: what leaves one is a MIME error with a code.

An error raised inside a public call that is not a MIME error is translated into one (:func:`translated`), keeping
the error as its ``original``, and of the original's class too, so that a handler written for it (``except Done``,
``except OSError``, ``pytest.raises(ValueError)``) catches it as before; where the original's class cannot be a base
of one, the original leaves unchanged. Either way, the full chain (every frame from the script's first down to the
raise, and every chained error) is kept on the error as ``chain``. Then, by default, the error leaves the call with
the script's own frames alone, and none chained; with ``MIME_TRACEBACK=full`` it leaves with every frame, the original
chained as its cause. A public call made inside another is left to the outer one."""

from __future__ import annotations

import functools
import inspect
import traceback
from collections.abc import Callable
from contextvars import ContextVar
from types import FrameType

from mime.build._refusals import MimeError, standing_for
from mime.errors._display import full_mode
from mime.errors._frames import script_traceback
from mime.errors._translate import translated

_DEPTH: ContextVar[int] = ContextVar("mime_public_calls", default=0)


def full_chain(error: BaseException, frame: FrameType | None) -> str:
    """Every frame from the script's first down to where ``error`` was raised, with every error chained to it, as
    Python writes them: ``frame`` is the public call's, whose callers the error's own traceback does not yet hold."""
    exception = traceback.TracebackException.from_exception(error)
    above: list[traceback.FrameSummary] = list(traceback.extract_stack(frame.f_back)) \
        if frame is not None and frame.f_back is not None else []
    exception.stack = traceback.StackSummary.from_list([*above, *exception.stack])
    return "".join(exception.format())


def crossed(error: Exception, frame: FrameType | None = None) -> MimeError:
    """``error`` as a MIME error, with its full chain kept on it: itself, or its translation, holding it as
    ``original``."""
    coded = error if isinstance(error, MimeError) else translated(error)
    if coded is not error:
        coded.original = error
    if coded.chain is None:
        coded.chain = full_chain(error, frame)
    return coded


def public[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Make ``function`` a public call: every error that leaves it is a MIME error with a code, shown with the script's
    own frames unless ``MIME_TRACEBACK=full``."""

    @functools.wraps(function)
    def call(*args: P.args, **kwargs: P.kwargs) -> R:
        token = _DEPTH.set(_DEPTH.get() + 1)
        try:
            return function(*args, **kwargs)
        except Exception as error:
            if _DEPTH.get() > 1:
                raise
            try:
                coded = crossed(error, inspect.currentframe())
                standing = coded if coded is error else standing_for(coded, error)
            except Exception:  # noqa: BLE001 - a defect of the translation never hides the error it translates
                raise error from None
            if standing is None:  # the original's class cannot be a base of one: it leaves as it was raised
                raise
            coded = standing
            if full_mode():
                if coded is error:
                    raise
                raise coded from error
            raise coded.with_traceback(script_traceback(error.__traceback__)) from None
        finally:
            _DEPTH.reset(token)

    return call


__all__ = ["crossed", "full_chain", "public"]
