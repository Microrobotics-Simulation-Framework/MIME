"""The control that registering from outside MIME works: the lab fixture package (``lab_fixture/``, built and
installed in every CI lane) registers ``labx.nudge`` from its own distribution, and it builds beside MIME's toys."""

import importlib
import os

import pytest

from mime.registry import registration_of
from mime.testing import history_differences, run_history
from mime.testing.toys import coupled_toy
from mime.units import s


def lab_fixture():
    try:
        return importlib.import_module("mime_lab_fixture")
    except ImportError:
        if os.environ.get("CI"):
            pytest.fail("the lab fixture package is not installed; every CI lane installs it from ./lab_fixture")
        pytest.skip("the lab fixture package is not installed: uv pip install --no-deps ./lab_fixture")


def test_an_entry_registered_from_outside_mime_builds_beside_mimes_own():
    fixture = lab_fixture()
    record = registration_of(fixture.Nudge)
    assert record is not None and record.namespace == "labx"
    assert record.cls.__module__ == "mime_lab_fixture"

    experiment = coupled_toy()
    assert experiment.attach(fixture.Nudge(), body="probe") == "labx.nudge:probe"
    built = experiment.build()
    assert [line.node for line in built.report.nodes][:2] == ["labx.nudge:probe", "toy.drag:probe"]

    duration = s(0.2)
    assert history_differences(run_history(coupled_toy().build(), duration), run_history(built, duration))
