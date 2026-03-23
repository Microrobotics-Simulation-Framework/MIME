#!/usr/bin/env python3
"""Launch a cloud rehearsal or sweep job via MADDENING CloudLauncher.

Architecture:
  - SkyPilot handles: provisioning, Docker image, workdir sync, setup commands
  - SSH handles: running the actual job (blocks until complete), scp retrieval
  - CloudLauncher handles: credentials, cost guards, teardown

This avoids the issue where SkyPilot's `stream_and_get` returns after the
run command is *submitted* (not completed), which would cause premature
teardown if we relied on it for job completion detection.

Usage:
    python3 scripts/launch_rehearsal.py --job jobs/rehearsal_a100.yaml
    python3 scripts/launch_rehearsal.py --job jobs/rehearsal_a100.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch cloud job via CloudLauncher")
    parser.add_argument("--job", required=True, help="Path to job config YAML")
    parser.add_argument("--creds", default=None, help="Path to credentials YAML")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no provisioning")
    parser.add_argument("--output", default="data/rehearsal_192.h5",
                        help="Local path to save retrieved HDF5")
    parser.add_argument("--skip-retrieve", action="store_true",
                        help="Skip HDF5 retrieval")
    args = parser.parse_args()

    job_path = Path(args.job)
    if not job_path.exists():
        print(f"Job config not found: {job_path}")
        sys.exit(1)

    try:
        from maddening.cloud.launcher import (
            CloudLauncher, JobConfig,
            CostLimitError, CredentialError, LaunchError,
        )
    except ImportError:
        print("CloudLauncher requires SkyPilot. Install with:\n"
              "  pip install 'skypilot[runpod]>=0.11'")
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

        # ── Write git hash for provenance ─────────────────────────
        # SkyPilot workdir sync doesn't include .git/, so we capture
        # the hash locally and write it to a file that gets synced.
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, timeout=5,
            ).strip()
            Path(".mime_git_hash").write_text(git_hash + "\n")
            print(f"  Git hash: {git_hash}")
        except Exception:
            print("  WARNING: Could not capture git hash")

        # ── Launch (provisions + setup + run) ─────────────────────
        print(f"\nLaunching instance (provision + setup)...")
        print("(SkyPilot streams provisioning and setup logs)\n")

        job = launcher.launch(job_path)

        print(f"\nInstance ready.")
        print(f"  Cluster: {job.cluster_name}")
        print(f"  VM IP: {job.vm_ip}")
        print(f"  SSH port: {job.ssh_port}")

        # ── Wait for SkyPilot job to finish ───────────────────────
        # SkyPilot submitted the run command as job ID 1.
        # Poll until it finishes, then retrieve output.
        print(f"\nWaiting for job to complete on instance...")
        print(f"  (Polling via SSH every 30s)")

        # Wait for the SkyPilot job to complete by checking if the
        # output file exists or by monitoring the process
        max_wait = 1800  # 30 minutes max
        poll_interval = 30
        t_start = time.time()

        while time.time() - t_start < max_wait:
            # Check if the rehearsal output file exists
            check = job.ssh_run(
                "test -f ~/sky_workdir/data/rehearsal_192.h5 && echo EXISTS || echo MISSING",
                capture=True, check=False, timeout=30,
            )
            if check.returncode == 0 and "EXISTS" in check.stdout:
                print(f"\n  Output file detected after {time.time() - t_start:.0f}s")
                break

            # Also check if the SkyPilot job process is still running
            proc_check = job.ssh_run(
                "pgrep -f 'rehearse_192' > /dev/null 2>&1 && echo RUNNING || echo DONE",
                capture=True, check=False, timeout=30,
            )
            if proc_check.returncode == 0:
                status = proc_check.stdout.strip()
                elapsed = time.time() - t_start
                print(f"  [{elapsed:.0f}s] Job status: {status}", flush=True)

                if "DONE" in status:
                    # Job finished — check if output exists
                    time.sleep(5)  # brief pause for file flush
                    check2 = job.ssh_run(
                        "test -f ~/sky_workdir/data/rehearsal_192.h5 && echo EXISTS || echo MISSING",
                        capture=True, check=False, timeout=30,
                    )
                    if check2.returncode == 0 and "EXISTS" in check2.stdout:
                        print(f"\n  Output file found after job completed")
                    else:
                        print(f"\n  WARNING: Job finished but output file not found")
                        # Grab the last 50 lines of job output for diagnosis
                        log_check = job.ssh_run(
                            "tail -50 /tmp/sky_logs/*/run.log 2>/dev/null || "
                            "tail -50 ~/sky_workdir/*.log 2>/dev/null || "
                            "echo 'No log files found'",
                            capture=True, check=False, timeout=30,
                        )
                        if log_check.returncode == 0:
                            print(f"\n  Last 50 lines of job log:")
                            print(log_check.stdout)
                    break

            time.sleep(poll_interval)
        else:
            print(f"\n  WARNING: Timed out after {max_wait}s waiting for job completion")

        # ── Get job output logs ───────────────────────────────────
        print(f"\nRetrieving job logs...")
        log_result = job.ssh_run(
            "cat /tmp/sky_logs/*/run.log 2>/dev/null || echo 'No run.log found'",
            capture=True, check=False, timeout=60,
        )
        if log_result.returncode == 0 and log_result.stdout.strip():
            print("\n" + "=" * 60)
            print("JOB OUTPUT")
            print("=" * 60)
            print(log_result.stdout)

        # ── Retrieve HDF5 output ──────────────────────────────────
        if not args.skip_retrieve:
            remote_path = "~/sky_workdir/data/rehearsal_192.h5"
            print(f"\nRetrieving output: {remote_path} -> {args.output}")
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

            scp_result = subprocess.run(
                [
                    "scp",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "LogLevel=ERROR",
                    "-P", str(job.ssh_port),
                    f"root@{job.vm_ip}:{remote_path}",
                    str(args.output),
                ],
                capture_output=True, text=True,
            )

            if scp_result.returncode == 0:
                file_size = os.path.getsize(args.output)
                print(f"  Retrieved: {args.output} ({file_size} bytes)")
            else:
                print(f"  WARNING: scp failed (exit {scp_result.returncode})")
                print(f"  stderr: {scp_result.stderr.strip()}")

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
        sys.exit(1)
    except LaunchError as e:
        print(f"Launch error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
