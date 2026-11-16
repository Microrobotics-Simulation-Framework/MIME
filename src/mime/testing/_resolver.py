"""A resolver of records held in memory: a source a test declares through the ``mime.assets`` entry point, as a DOI's,
a hub's or a derived scan's resolver is declared, with nothing fetched to answer a name."""

from __future__ import annotations

from mime.assets import AssetRef


class RecordResolver:
    """Answers a name with the record of that name among ``records``, or ``None``; ``asked`` lists every name it was
    asked for, in order, so a test can say when MIME asked it."""

    def __init__(self, *records: AssetRef) -> None:
        self.records = {held.name: held for held in records}
        self.asked: list[str] = []

    def record(self, name: str) -> AssetRef | None:
        self.asked.append(name)
        return self.records.get(name)


__all__ = ["RecordResolver"]
