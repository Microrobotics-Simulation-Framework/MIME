"""The seam to OpenMaterica, with the inline floor: the one place that reads the values of things (a body's mass, a
medium's density) and of a body's parts, and the only importer of OpenMaterica. A value comes from a declaration in
the experiment, a record read at the experiment's conditions, or a body's parts, and says which."""

from mime.materials._records import Acceptance, Extrapolation, Nominal, Record, RecordFile, packaged_record, record_file
from mime.materials._tables import evaluator_table
from mime.materials._values import VALUE_KINDS


def openmaterica_version() -> str:
    """The installed OpenMaterica's version, which the lock records."""
    import importlib.metadata

    return importlib.metadata.version("openmaterica")


__all__ = [
    "VALUE_KINDS",
    "Acceptance",
    "Extrapolation",
    "Nominal",
    "Record",
    "RecordFile",
    "evaluator_table",
    "packaged_record",
    "record_file",
]
