"""The typing gate refuses a run it cannot trust, rather than reading it as clean: each way a pyright run goes wrong
is planted with a stored report or a fake pyright, and must exit 2; errors exit 1; a clean, trustworthy run exits 0.
The live run of the gate is CI's ``typecheck`` job, which also plants a type error in ``src/mime`` and requires it
to fail."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "typing_check.py"


def _load():
    spec = importlib.util.spec_from_file_location("typing_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


typing_check = _load()
RELEASE = typing_check.pinned_release()


def report(*, version=RELEASE, files=59, diagnostics=()):
    return {"version": version, "summary": {"filesAnalyzed": files, "errorCount": 0},
            "generalDiagnostics": list(diagnostics)}


def error(rule="reportArgumentType", file="/repo/src/mime/units.py", message="planted"):
    return {"severity": "error", "rule": rule, "file": file, "message": message}


def gate(tmp_path, document, *extra):
    path = tmp_path / "pyright.json"
    path.write_text(json.dumps(document))
    return typing_check.main(["--json", str(path), *extra])


def fake_pyright(tmp_path, stdout: str, code: int) -> str:
    fake = tmp_path / "fake_pyright.py"
    fake.write_text(f"import sys\nsys.stdout.write({stdout!r})\nsys.exit({code})\n")
    return f"{sys.executable} {fake}"


def test_the_pin_is_the_release_the_project_requires():
    assert RELEASE == "1.1.414"


def test_a_clean_trustworthy_run_passes(tmp_path, capsys):
    assert gate(tmp_path, report()) == 0
    assert "files analysed: 59; errors: 0" in capsys.readouterr().out


def test_an_error_fails_the_gate(tmp_path, capsys):
    assert gate(tmp_path, report(diagnostics=[error()])) == 1
    assert "units.py" in capsys.readouterr().out


@pytest.mark.parametrize(("document", "reason"), [
    (report(version="1.1.400"), "every release changes diagnostics"),
    (report(files=0), "analysed 0 files"),
    (report(files=12), "analysed 12 files"),
    (report(diagnostics=[error("reportMissingImports", message='Import "jax.numpy" could not be resolved')]),
     "core imports jax.numpy"),
    ({"version": RELEASE, "generalDiagnostics": []}, "no summary"),
], ids=["another release", "no file", "too few files", "core import unresolved", "no summary"])
def test_a_run_that_cannot_be_trusted_exits_2(tmp_path, capsys, document, reason):
    assert gate(tmp_path, document) == 2
    assert reason in capsys.readouterr().err


@pytest.mark.parametrize(("stdout", "code", "reason"), [
    (json.dumps(report()), 3, "pyright exited 3"),
    ("not json", 1, "wrote no JSON"),
], ids=["exit 3", "no JSON"])
def test_a_pyright_that_misbehaves_exits_2(tmp_path, capsys, stdout, code, reason):
    assert typing_check.main(["--pyright", fake_pyright(tmp_path, stdout, code)]) == 2
    assert reason in capsys.readouterr().err


def test_a_missing_pyright_exits_2(tmp_path, capsys):
    assert typing_check.main(["--pyright", str(tmp_path / "no-such-pyright")]) == 2
    assert "cannot run pyright" in capsys.readouterr().err
