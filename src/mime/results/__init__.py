"""A run's results: one ``RunResult`` from a run and from a run's folder, its observations' declared outputs by name,
with units, series with their time axes, windows by time, readings and the flags raised over them, and the exports
(a table, CSV, NPZ and the manifest a run folder holds)."""

from mime.results._layout import FlagLayout, Layout, ObservationLayout, OutputLayout
from mime.results._result import (
    MANIFEST_FORMAT,
    MANIFEST_VERSION,
    RUN_MANIFEST,
    ObservationOutputs,
    OutputRef,
    RunResult,
    Samples,
)
from mime.results._series import FlagRead, FlagTrack, NotRecorded, Reading, Series, Table

__all__ = ["MANIFEST_FORMAT", "MANIFEST_VERSION", "RUN_MANIFEST", "FlagLayout", "FlagRead", "FlagTrack", "Layout",
           "NotRecorded", "ObservationLayout", "ObservationOutputs", "OutputLayout", "OutputRef", "Reading",
           "RunResult", "Samples", "Series", "Table"]
