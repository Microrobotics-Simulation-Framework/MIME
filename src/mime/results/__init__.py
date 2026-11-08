"""A run's results: one ``RunResult`` from a run and from a run's folder, its observations' declared outputs by name,
with units, series with their time axes, windows by time, readings and the flags raised over them, and the exports
(a table, CSV, NPZ and the manifest a run folder holds)."""

from mime.results._containment import ContainmentRead, read_containments
from mime.results._energy import EnergyAccount, LawEnergy, energy_account, group_verdicts
from mime.results._layout import FlagLayout, Layout, ObservationLayout, OutputLayout
from mime.results._result import (
    MANIFEST_FORMAT,
    MANIFEST_VERSION,
    REPORT_FORMAT,
    RUN_MANIFEST,
    RUN_REPORT,
    MemberOf,
    NoOutput,
    NotARunFolder,
    ObservationOutputs,
    OutputRef,
    RunResult,
    Samples,
    Stopped,
)
from mime.results._order import Order, observed_order
from mime.results._study import NOT_DECLARED, Dropped, StackedSeries, StudyGrid, StudyOutputs, StudyResult
from mime.results._series import FlagRead, FlagTrack, FlagWhole, NotInResult, NotRecorded, Reading, Series, Table
from mime.results._watch import WatchRead, read_watches

__all__ = ["MANIFEST_FORMAT", "MANIFEST_VERSION", "NOT_DECLARED", "REPORT_FORMAT", "RUN_MANIFEST", "RUN_REPORT",
           "ContainmentRead", "Dropped", "EnergyAccount", "LawEnergy", "MemberOf", "Order", "StackedSeries", "StudyGrid",
           "StudyOutputs", "StudyResult",
           "FlagLayout", "FlagRead", "FlagTrack", "FlagWhole", "Layout", "NoOutput", "NotARunFolder", "NotInResult",
           "NotRecorded", "ObservationLayout", "ObservationOutputs", "OutputLayout", "OutputRef", "Reading", "RunResult",
           "Samples", "Series", "Stopped", "Table", "WatchRead", "energy_account", "group_verdicts", "observed_order",
           "read_containments",
           "read_watches"]
