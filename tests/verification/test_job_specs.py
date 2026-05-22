"""§8 Step 2 — every job spec parses and names a real cloud provider.

Pins the `jobs/*.yaml` specs (including the AWS/GCP variants added in the
v0.2 fit-up) as parseable MADDENING `JobConfig`s whose `provider:` resolves
through MADDENING's `PROVIDERS` registry.
"""

from pathlib import Path

import pytest

from maddening.cloud.launcher import JobConfig
from maddening.cloud.providers import PROVIDERS

_JOBS = sorted((Path(__file__).resolve().parents[2] / "jobs").glob("*.yaml"))


def test_jobs_dir_is_not_empty():
    assert _JOBS, "no job specs found under jobs/"


@pytest.mark.parametrize("job_path", _JOBS, ids=lambda p: p.name)
def test_job_spec_parses_and_provider_resolves(job_path):
    """The spec parses as a JobConfig and its provider is a known cloud."""
    jc = JobConfig.from_yaml(job_path)
    assert jc.provider in PROVIDERS, (
        f"{job_path.name}: provider {jc.provider!r} is not one of "
        f"{sorted(PROVIDERS)}"
    )
