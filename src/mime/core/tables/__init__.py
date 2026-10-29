"""The table evaluator: one differentiable reader of tables, in pure JAX, for MIME's own solver-output tables and for
OpenMaterica's material tables alike. A table is described as OpenMaterica describes one (axes with units, readable
range, declared edges, the scheme, the consumer's extrapolation), read in the rig's dtype, refused at an edge and never
clipped outside its readable range."""

from mime.core.tables._compile import AxisArrays, TableArrays
from mime.core.tables._evaluate import SEARCH_METHOD, Evaluation, TableRefused, evaluate, refuse
from mime.core.tables._table import (
    GATHERED_AT_ONCE,
    SCHEMES,
    SIDES,
    STRUCTURES,
    Axis,
    ConstantExtrapolation,
    Scheme,
    Side,
    Structure,
    Table,
    TableInvalid,
    query_range_problems,
)

__all__ = [
    "GATHERED_AT_ONCE",
    "SCHEMES",
    "SEARCH_METHOD",
    "SIDES",
    "STRUCTURES",
    "Axis",
    "AxisArrays",
    "ConstantExtrapolation",
    "Evaluation",
    "Scheme",
    "Side",
    "Structure",
    "Table",
    "TableArrays",
    "TableInvalid",
    "TableRefused",
    "evaluate",
    "query_range_problems",
    "refuse",
]
