"""Learned models that nodes evaluate: a surrogate as a type (``Surrogate``: an asset record naming its evaluator, its
outputs, features, accuracy per use and provenance), the Cholesky MLP that predicts a screw body's resistance in a tube
as its first evaluator, and what each packaged surrogate was trained on. The generator of training data and its
held-out protocol are ``mime.surrogates.generator``, imported on its own."""

from mime.surrogates._screw import SURROGATES
from mime.surrogates._type import (
    Accuracy,
    Evaluator,
    Provenance,
    Surrogate,
    evaluator_named,
    register_evaluator,
    staleness,
)

__all__ = [
    "SURROGATES",
    "Accuracy",
    "Evaluator",
    "Provenance",
    "Surrogate",
    "evaluator_named",
    "register_evaluator",
    "staleness",
]
