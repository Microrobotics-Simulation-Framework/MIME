"""The documentation's machine-readable sources and their checks: the registry export (``mime registry --json``), the
checks on a node's ``NodeMeta`` and symbol table, the refusal scan the refusal table is generated from, and the
ratchets kept as lists. ``mime docs check`` runs them all over a checkout, and needs the ``verify`` extra; nothing
imported here needs it, so ``mime registry --json`` runs on a base install. The pages themselves are generated
later, from these sources."""

from mime.docs._export import (
    ENTRY_POINT_GROUP,
    Unencodable,
    class_vars,
    completeness_problems,
    encode,
    entry_record,
    export,
    export_json,
    load_namespaces,
    registry_export,
)
from mime.docs._lists import Line, RatchetList, floor_problems, list_problems, read_list
from mime.docs._refusal_scan import CODE_SHAPE, Census, Derived, Scan, Site, census, derive, name_rule, scan_package
from mime.docs._symbols import (
    STRUCTURE,
    TableReport,
    Token,
    instance_problems,
    meta_of,
    next_id,
    path_problem,
    read_frozen_ids,
    row_key,
    rows_of,
    static_problems,
    symbols_of,
    table_problems,
    tokens,
    unrowed,
)

__all__ = [
    "CODE_SHAPE",
    "ENTRY_POINT_GROUP",
    "STRUCTURE",
    "Census",
    "Derived",
    "Line",
    "RatchetList",
    "Scan",
    "Site",
    "TableReport",
    "Token",
    "Unencodable",
    "census",
    "class_vars",
    "completeness_problems",
    "derive",
    "encode",
    "entry_record",
    "export",
    "export_json",
    "floor_problems",
    "instance_problems",
    "list_problems",
    "load_namespaces",
    "meta_of",
    "name_rule",
    "next_id",
    "path_problem",
    "read_frozen_ids",
    "read_list",
    "registry_export",
    "row_key",
    "rows_of",
    "scan_package",
    "static_problems",
    "symbols_of",
    "table_problems",
    "tokens",
    "unrowed",
]
