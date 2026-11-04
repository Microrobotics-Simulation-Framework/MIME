"""``mime --version`` and ``mime -V``: the line a bug report starts with, MIME's version from its one source
(``mime.__version__``, which the distribution's metadata is built from, ``tests/test_package.py``), on standard output
with nothing else, exit 0. The flag is in ``mime --help`` and in the CLI reference generated from the parser.

The control: a command whose parser has no such flag refuses ``--version`` with argparse's usage error and exit 2,
which is what ``mime --version`` gave before the flag existed, so the first test cannot pass on a parser without it."""

from pathlib import Path

import pytest

import mime
from mime.cli import main as mime_main
from mime.docs._generated import cli_reference

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_the_flag_prints_mimes_version_alone_and_exits_zero(capsys, flag):
    with pytest.raises(SystemExit) as stopped:
        mime_main([flag])
    assert stopped.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == f"mime {mime.__version__}\n" and captured.err == ""


def test_the_flag_is_in_the_help_and_in_the_generated_reference(capsys):
    with pytest.raises(SystemExit):
        mime_main(["--help"])
    out = capsys.readouterr().out
    assert "[-V]" in out.splitlines()[0] and "-V, --version" in out
    assert "`-V`, `--version`" in cli_reference(ROOT)


def test_control_a_parser_without_the_flag_refuses_it(capsys):
    with pytest.raises(SystemExit) as stopped:
        mime_main(["registry", "--version"])
    assert stopped.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "unrecognized arguments: --version" in captured.err
