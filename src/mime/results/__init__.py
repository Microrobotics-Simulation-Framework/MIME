"""A run's results: one ``RunResult`` from a run and from a run's folder, its observations' declared outputs by name,
with units, series with their time axes, windows by time, readings and the flags raised over them, and the exports
(a table, CSV, NPZ and the manifest a run folder holds)."""

from mime.results._compare import TOLERANCE_EPS, Comparison, Miss
from mime.results._containment import ContainmentRead, read_containments
from mime.results._data import FlagData, LimitData, Measured, ReadingData, ResultData, RunData, SeriesData, StudyData, Uncertainty
from mime.results._draw import FigureOptions
from mime.results._energy import EnergyAccount, LawEnergy, energy_account, group_verdicts
from mime.results._figure import FigurePage, FigureRecipe, FigureSpec, figure
from mime.results._fit import FitIterate, FitResult, LineMargin
from mime.results._gradient import GradientResult, MemberGradient, RaisedFlag, Recheck, SolveReport
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
from mime.results._plot import PlotDescription
from mime.results._study import NOT_DECLARED, Dropped, StackedSeries, StudyGrid, StudyOutputs, StudyResult
from mime.results._series import FlagRead, FlagTrack, FlagWhole, NotInResult, NotRecorded, Reading, Series, Table
from mime.results._watch import WatchRead, read_watches

__all__ = ["FlagData", "LimitData", "Measured", "ReadingData", "ResultData", "RunData", "SeriesData", "StudyData",
           "Uncertainty", "figure", "TOLERANCE_EPS", "Comparison", "Miss", "FigureOptions", "FigurePage", "FigureRecipe", "FigureSpec", "PlotDescription", "MANIFEST_FORMAT", "MANIFEST_VERSION", "NOT_DECLARED", "REPORT_FORMAT", "RUN_MANIFEST", "RUN_REPORT",
           "ContainmentRead", "Dropped", "EnergyAccount", "FitIterate", "FitResult", "GradientResult", "MemberGradient", "RaisedFlag", "Recheck",
           "SolveReport", "LawEnergy", "MemberOf", "Order", "StackedSeries", "StudyGrid", "StudyOutputs", "StudyResult",
           "FlagLayout", "FlagRead", "FlagTrack", "FlagWhole", "Layout", "NoOutput", "NotARunFolder", "NotInResult",
           "NotRecorded", "ObservationLayout", "ObservationOutputs", "OutputLayout", "OutputRef", "Reading", "RunResult",
           "LineMargin", "Samples", "Series", "Stopped", "Table", "WatchRead", "energy_account", "group_verdicts", "observed_order",
           "read_containments",
           "read_watches"]
