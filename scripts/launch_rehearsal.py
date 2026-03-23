#!/usr/bin/env python3
"""Launch a cloud rehearsal or sweep job via MADDENING CloudLauncher.

Handles: validation, launch, log streaming, HDF5 retrieval via scp,
and explicit instance teardown.

Usage:
    python3 scripts/launch_rehearsal.py --job jobs/rehearsal_a100.yaml
    python3 scripts/launch_rehearsal.py --job jobs/rehearsal_a100.yaml --dry-run
    python3 scripts/launch_rehearsal.py --job jobs/sweep_h100.yaml --output data/sweep_192.h5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch cloud job via CloudLauncher")
    parser.add_argument(
        "--job", required=True,
        help="Path to job config YAML",
    )
    parser.add_argument(
        "--creds", default=None,
        help="Path to credentials YAML (default: ~/.maddening/cloud_credentials.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and resolve resources without provisioning",
    )
    parser.add_argument(
        "--output", default="data/rehearsal_192.h5",
        help="Local path to save the retrieved HDF5 file",
    )
    parser.add_argument(
        "--remote-output", default="/root/MIME/data/rehearsal_192.h5",
        help="Remote path of the HDF5 file on the cloud instance",
    )
    parser.add_argument(
        "--skip-retrieve", action="store_true",
        help="Skip HDF5 retrieval (e.g. if no output file is expected)",
    )
    args = parser.parse_args()

    job_path = Path(args.job)
    if not job_path.exists():
        print(f"Job config not found: {job_path}")
        sys.exit(1)

    try:
        from maddening.cloud.launcher import (
            CloudLauncher,
            CostLimitError,
            CredentialError,
            LaunchError,
        )
    except ImportError:
        print(
            "CloudLauncher requires SkyPilot. Install with:\n"
            "  pip install 'skypilot[runpod]>=0.11'"
        )
        sys.exit(1)

    try:
        launcher = CloudLauncher(credentials_path=args.creds)

        # ── Validate ──────────────────────────────────────────────
        print("Validating configuration...")
        result = launcher.validate(job_path)
        print(f"  Provider: {result['provider']}")
        print(f"  GPU: {result['gpu_type']}")
        print(f"  Instance: {result['instance_type']}")
        print(f"  Hourly cost: ${result['hourly_cost']:.2f}")
        print(f"  Budget used: ${result['budget_used']:.2f}")
        print(f"  Budget remaining: ${result['budget_remaining']:.2f}")

        if args.dry_run:
            print("\nDry run complete — no resources provisioned.")
            return

        # ── Launch ────────────────────────────────────────────────
        print(f"\nLaunching job...")
        print("(Streaming logs — this blocks until the job completes)\n")

        job = launcher.launch(job_path)

        print(f"\nJob completed.")
        print(f"  Cluster: {job.cluster_name}")
        print(f"  VM IP: {job.vm_ip}")
        print(f"  SSH port: {job.ssh_port}")

        # ── Retrieve HDF5 output ──────────────────────────────────
        if not args.skip_retrieve:
            print(f"\nRetrieving output: {args.remote_output} -> {args.output}")
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

            scp_result = subprocess.run(
                [
                    "scp",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "LogLevel=ERROR",
                    "-P", str(job.ssh_port),
                    f"root@{job.vm_ip}:{args.remote_output}",
                    str(args.output),
                ],
                capture_output=True,
                text=True,
            )

            if scp_result.returncode == 0:
                file_size = os.path.getsize(args.output)
                print(f"  Retrieved: {args.output} ({file_size} bytes)")
            else:
                print(f"  WARNING: scp failed (exit {scp_result.returncode})")
                print(f"  stderr: {scp_result.stderr.strip()}")
                print("  (Instance will still be torn down)")

        # ── Teardown ──────────────────────────────────────────────
        print(f"\nTearing down instance...")
        job.teardown()
        print("Instance terminated.")

        # ── Cost report ───────────────────────────────────────────
        try:
            cost = job.cost_so_far()
            print(f"\nEstimated cost: ${cost:.2f}")
        except Exception:
            pass

        print("\nDone.")

    except CredentialError as e:
        print(f"Credential error: {e}")
        sys.exit(1)
    except CostLimitError as e:
        print(f"Cost limit exceeded: {e}")
        print(f"  guard_type={e.guard_type}, limit={e.limit}, actual={e.actual}")
        sys.exit(1)
    except LaunchError as e:
        print(f"Launch error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
