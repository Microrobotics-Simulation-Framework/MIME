"""A run's results: one ``RunResult`` from a run and from a run's folder, its observations' declared outputs by name,
with units, series with their time axes, windows by time, readings and the flags raised over them, and the exports
(a table, CSV, NPZ and the manifest a run folder holds)."""

from mime.results._layout import FlagLayout, Layout, ObservationLayout, OutputLayout
from mime.results._result import (
    MANIFEST_FORMAT,
    MANIFEST_VERSION,
    REPORT_FORMAT,
    RUN_MANIFEST,
    RUN_REPORT,
    NoOutput,
    NotARunFolder,
    ObservationOutputs,
    OutputRef,
    RunResult,
    Samples,
    Stopped,
)
from mime.results._series import FlagRead, FlagTrack, NotInResult, NotRecorded, Reading, Series, Table
from mime.results._watch import WatchRead, read_watches

__all__ = ["MANIFEST_FORMAT", "MANIFEST_VERSION", "REPORT_FORMAT", "RUN_MANIFEST", "RUN_REPORT", "FlagLayout", "FlagRead", "FlagTrack", "Layout",
           "NoOutput", "NotARunFolder", "NotInResult", "NotRecorded", "ObservationLayout", "ObservationOutputs",
           "OutputLayout", "OutputRef", "Reading", "RunResult", "Samples", "Series", "Stopped", "Table", "WatchRead", "read_watches"]
