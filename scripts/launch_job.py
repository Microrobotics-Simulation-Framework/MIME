#!/usr/bin/env python3
"""Launch a MIME cloud job from a JobConfig YAML.

A provider-agnostic wrapper over MADDENING's ``CloudLauncher`` — it reads the
spec's ``provider:`` field and dispatches to whichever cloud it names
(runpod / aws / gcp / lambda_labs). Credentials are read from
``~/.maddening/cloud_credentials.yaml``.

Usage::

    python scripts/launch_job.py --job jobs/production_h100.yaml
    python scripts/launch_job.py --job jobs/production_h100_aws.yaml --dry-run

``--dry-run`` resolves the provider, instance type and cost without
provisioning anything — use it to sanity-check a spec.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--job", required=True, help="path to the JobConfig YAML")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve provider / resources / cost without "
                         "provisioning")
    ap.add_argument("--credentials", default=None,
                    help="cloud credentials YAML "
                         "(default: ~/.maddening/cloud_credentials.yaml)")
    args = ap.parse_args()

    from maddening.cloud.launcher import CloudLauncher, JobConfig

    jc = JobConfig.from_yaml(args.job)
    print(f"Job: {args.job}  ->  provider={jc.provider}, "
          f"gpu={jc.gpu_type} x{jc.gpu_count}, spot={jc.use_spot}")

    launcher = CloudLauncher(credentials_path=args.credentials)
    try:
        if args.dry_run:
            plan = launcher.validate(jc)
            print("Dry run — resolved plan (nothing provisioned):")
            for key, value in plan.items():
                print(f"  {key}: {value}")
        else:
            job = launcher.launch(jc)
            print(f"Launched: cluster={job.cluster_name}")
            print("Follow provisioning via the job's stream_logs().")
    except Exception as exc:  # noqa: BLE001 — surface any launch failure cleanly
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        if "credential" in f"{type(exc).__name__}{exc}".lower():
            print("Add the provider's block to "
                  "~/.maddening/cloud_credentials.yaml.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
