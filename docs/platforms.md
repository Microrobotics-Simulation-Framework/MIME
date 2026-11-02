# Platforms

What MIME supports on each platform, what a test shows of it, and what none does. Each line says what the **smoke
job** (`.github/workflows/platforms.yml`) tests, with the run that showed it, and what it does not.

| Platform | CPU | GPU | Tested by |
|---|---|---|---|
| Linux, x86_64 | supported | supported, through the `cuda12` variant | every lane of `ci.yml` (the full suite) |
| Linux, aarch64 | supported (locked wheels) | the `cuda12` variant, untested | no lane: the runners are x86_64 |
| macOS, Apple Silicon | supported, with the limits below | none (JAX has no CUDA there) | the smoke job |
| macOS, Intel | **not supported**: the locked jaxlib has no wheel for it | none | nothing |
| Windows, native | supported, with the limits below | none: JAX's CUDA wheels are Linux-only | the smoke job |
| Windows, WSL2 | as Linux | as Linux, through WSL2 | nothing of its own |

## For a large job, or a GPU

**On macOS and on native Windows, run a large job on the CPU or in the cloud** (`mime modal`,
[Running on Modal](running_on_modal.md)). **For a GPU, use the cloud, or WSL2 on Windows**: JAX's CUDA wheels exist for
Linux only, and WSL2 is Linux. The lock pins the `cuda12` variant's wheels by hash for Linux alone.

An experiment saved on one platform is another environment on every other: a load there says so, and a run is
verified within the lock's tolerances there, never bit for bit (below).

## What the smoke job tests

On `macos-latest` (Apple Silicon) and `windows-latest`, each with Python 3.12 (the default lane's) and 3.14, installed
from the lock as the default lane is (`uv sync --locked --extra verify`, then the lab fixture). **It sets no
`PYTHONUTF8` and no `PYTHONIOENCODING`**: it meets what a user's machine meets, and on Windows that is the ANSI code
page for a redirected stream (Python before 3.15 does not default to UTF-8 there). It runs the test files named in
`tests/platforms.txt`, one reason each, and then `mime docs check`:

- `import mime`, and the `mime` console script through pipes: `mime registry`, `mime env check` (a development install
  is refused for its tools and for nothing else), `mime run` and `mime verify` on a toy experiment saved in a folder
  whose name holds a space, a micro sign and a Greek capital omega;
- a free-space experiment built, run, saved, loaded and verified, and every text file a save or a run writes read as
  bytes (`tests/platforms/test_smoke.py`), `mime run` (`tests/env/test_run.py`) and the saved form's round trips
  (`tests/saved/`: the document, the load, the schema, the array guards, description files in a chain);
- `mime modal`'s script and files, written and read through a stand-in for `modal` (`tests/env/test_modal_script.py`):
  a Windows or Mac user runs `mime modal` on their own machine;
- `mime docs check --base origin/v0.3`, with the pages built under eight hash seeds.

The run that showed it: https://github.com/Microrobotics-Simulation-Framework/MIME/actions/runs/41222449103 (all four
jobs passed, with the skips below), at commit `ca6836f`. The run before it
(https://github.com/Microrobotics-Simulation-Framework/MIME/actions/runs/41221919224) is what the fixes in that commit
answer: a carriage return in every saved text file on Windows, a `ValueError` showing a script on another drive, and
`mime docs check` dying on a minus sign.

### What it found, and what is still so

- **Fixed:** a redirected stream no longer kills a command on a character the code page lacks (it is written as its
  escape); saved and written text files are the same bytes on every platform; `mime modal` shows a path on another
  drive as it is.
- **Fixed: a checkout on Windows changed the bytes of a pinned file.** Git there ships with `core.autocrlf=true`, which
  wrote `records/water.json` with CRLF line ends (22,899 bytes, not 22,396), so its pinned sha256 no longer matched and two
  of the error corpus's cases were refused before their planted mistake was reached (they were skipped on Windows). The
  root `.gitattributes` (`* text=auto eol=lf`, binaries marked) keeps every tracked file's bytes on every platform, and
  `tests/test_gitattributes.py` holds that no tracked file is left to the user's configuration.
- **By decision: off Linux, verification is within the lock's tolerances.** The CPU's flags are read from
  `/proc/cpuinfo`, and a device field that cannot be read is never equal to anything. So on macOS and Windows a load
  reports the environment as different from the one the lock was written in, a verification is "reproduced within
  tolerance" and never "bit for bit", and the guard that a factory building another graph under the same version is
  refused cannot fire. The smoke holds the diagnosis (the flags are the only difference), and five tests that need a
  readable hash are skipped there (`needs_readable_cpu_flags`, `tests/_platform_support.py`). The maintainer decided
  (2026-10-31, "within tolerances is fine off Linux for now") that for dev1 this stands: **verification off Linux is
  within the lock's tolerances, by decision, and exact replication is had by running in the cloud** (`mime modal`,
  [Running on Modal](running_on_modal.md)), where the environment is Linux and recorded. Not built, and not needed
  while that holds: a hash of the platform's feature set on macOS (`sysctl hw.optional`) and on Windows
  (`IsProcessorFeaturePresent`), which would say what makes two machines the same.
- **Skipped on Windows:** the release test of the Modal script's generation, which starts a script with a shebang line
  as `uv`; Windows starts only an `.exe`. And the three tests of Ctrl+C on `mime run` that send SIGINT to a child
  process (`needs_posix_sigint`, `tests/_platform_support.py`): Windows delivers Ctrl+C to a console's process group.
  The same behaviour is tested in process there (`tests/platforms/test_interrupt.py`, with `signal.raise_signal`); a
  Windows console's own Ctrl+C event and a terminal's keypress were not exercised.

## What it does not test

- **No GPU off Linux**, and no GPU in any lane: the `gpu`-marked tests skip without a CUDA device, and no runner has
  one. The `cuda12` variant is pinned, and `mime modal --gpu` is the way to a GPU that has been exercised.
- **The full lane runs on Linux x86_64 only.** The smoke is a selection of free-space cases: nothing that needs the
  unit tube table (the tube wall kind, 209 MB), the MRS families, the physics suites or the property tests
  runs on macOS or Windows. Float32 and float64 results on those platforms are not compared with Linux's.
- **Intel Macs, Linux on aarch64 and WSL2** have no runner; the first has no wheel, the second has locked wheels, the
  third is Linux under Windows.
- **Terminals.** The streams were tested redirected, which is where Windows differs. An interactive Windows console
  writes Unicode and was not run.
