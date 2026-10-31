"""The seam to OpenMaterica, with the inline floor: the one place that reads the values of things (a body's mass, a
medium's density) and of a body's parts, and the only importer of OpenMaterica. A value comes from a declaration in
the experiment, a record read at the experiment's conditions, or a body's parts, and says which."""

from mime.materials._read import CONDITION_UNITS, OperatingCondition, ReadOutcome, operating_conditions, read_target
from mime.materials._records import (
    RECORDS,
    Acceptance,
    Extrapolation,
    LoadedRecord,
    Nominal,
    Record,
    RecordFile,
    load_record,
    packaged_record,
    record_file,
    record_problems,
    record_store,
)
from mime.materials._resolver import PartPlan, Resolution, Resolver
from mime.materials._tables import evaluator_table
from mime.materials._values import (
    APPARATUS_PART_VALUE_KINDS,
    DECLARED_IN_THE_EXPERIMENT,
    DERIVED_FROM_PARTS,
    PART_VALUE_KINDS,
    PARTS,
    VALUE_KINDS,
    Declared,
    Directed,
    RecordField,
    ResolvedValue,
    ValueKind,
    ValuePath,
    declaration_problems,
    kind_of_path,
    part_value_kinds,
    resolve,
    resolve_declared,
    value_kind,
)


def openmaterica_version() -> str:
    """The installed OpenMaterica's version, which the lock records."""
    import importlib.metadata

    return importlib.metadata.version("openmaterica")


__all__ = [
    "CONDITION_UNITS",
    "DECLARED_IN_THE_EXPERIMENT",
    "DERIVED_FROM_PARTS",
    "PARTS",
    "APPARATUS_PART_VALUE_KINDS",
    "PART_VALUE_KINDS",
    "RECORDS",
    "VALUE_KINDS",
    "Acceptance",
    "Declared",
    "Directed",
    "Extrapolation",
    "LoadedRecord",
    "Nominal",
    "OperatingCondition",
    "PartPlan",
    "ReadOutcome",
    "Record",
    "RecordField",
    "RecordFile",
    "Resolution",
    "ResolvedValue",
    "Resolver",
    "ValueKind",
    "ValuePath",
    "declaration_problems",
    "evaluator_table",
    "kind_of_path",
    "part_value_kinds",
    "load_record",
    "openmaterica_version",
    "operating_conditions",
    "read_target",
    "packaged_record",
    "record_file",
    "record_problems",
    "record_store",
    "resolve",
    "resolve_declared",
    "value_kind",
]
