"""Interface names with their kind, units, frame, shape, sign and dtype; the capability flags."""

from mime.vocabulary._vocabulary import (
    MIME_NAMESPACE,
    Capability,
    Interface,
    ReadKind,
    TargetKind,
    define,
    lookup,
    names,
    namespace_of,
    register_namespace,
    versions,
)

__all__ = [
    "MIME_NAMESPACE",
    "Capability",
    "Interface",
    "ReadKind",
    "TargetKind",
    "define",
    "lookup",
    "names",
    "namespace_of",
    "register_namespace",
    "versions",
]
