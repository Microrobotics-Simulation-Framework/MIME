"""Learned models that nodes evaluate: a surrogate as a type (``Surrogate``: an asset record naming its evaluator, its
outputs, features, accuracy per use and provenance), the Cholesky MLP that predicts a screw body's resistance in a tube
as its first evaluator, and what each packaged surrogate was trained on. The generator of training data and its
held-out protocol are ``mime.surrogates.generator``, imported on its own."""

from mime.surrogates._cholesky_mlp import (
    CHOLESKY_MLP,
    DIAGONAL_FLOOR,
    TRIANGLE,
    CholeskyMLP,
    cholesky_resistance,
    mlp_forward,
    predict_resistance,
    read_cholesky_mlp,
)
from mime.surrogates._screw import DEJONGH_MLP_V2, SCREW_FEATURES, SURROGATES, DeclaredRange, ScrewSurrogate
from mime.surrogates._type import (
    OUTPUTS,
    REFERENCE_POINTS,
    Accuracy,
    Evaluator,
    Provenance,
    Surrogate,
    current_inputs,
    evaluator_named,
    evaluators,
    register_evaluator,
    staleness,
)

__all__ = [
    "CHOLESKY_MLP",
    "DEJONGH_MLP_V2",
    "DIAGONAL_FLOOR",
    "OUTPUTS",
    "REFERENCE_POINTS",
    "SCREW_FEATURES",
    "SURROGATES",
    "TRIANGLE",
    "Accuracy",
    "CholeskyMLP",
    "DeclaredRange",
    "Evaluator",
    "Provenance",
    "ScrewSurrogate",
    "Surrogate",
    "cholesky_resistance",
    "current_inputs",
    "evaluator_named",
    "evaluators",
    "mlp_forward",
    "predict_resistance",
    "read_cholesky_mlp",
    "register_evaluator",
    "staleness",
]
