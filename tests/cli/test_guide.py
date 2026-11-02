"""How ``mime`` introduces itself, with no dependency: ``--help`` grouped by what one is doing, an example under every
command, a bare ``mime`` that says what one can do here, and "did you mean" under a mistyped command or flag, on every
Python MIME supports."""

import json
import sys

import pytest

from mime.cli import build_parser
from mime.cli import main as mime_main
from mime.cli import _guide
from mime.saved import DOCUMENT_FILE, LOCK_FILE


def said(capsys):
    captured = capsys.readouterr()
    return captured.out, captured.err


def refused_parse(capsys, *words):
    with pytest.raises(SystemExit) as stopped:
        mime_main(list(words))
    assert stopped.value.code == 2
    return said(capsys)[1]


def top_commands():
    return build_parser()


def test_every_command_is_in_exactly_one_group_and_the_help_lists_them_in_order(capsys):
    parser = top_commands()
    names = list(_guide._children(parser))
    grouped = [name for _, group in _guide.GROUPS for name in group]
    assert sorted(grouped) == sorted(names), "a command in no group, or in two, or a group naming none"
    with pytest.raises(SystemExit):
        mime_main(["--help"])
    out = said(capsys)[0]
    titles = [out.index(f"{title}:") for title, _ in _guide.GROUPS]
    assert titles == sorted(titles)
    for title, group in _guide.GROUPS:
        section = out[out.index(f"{title}:"):]
        later = [out.index(f"{other}:") for other, _ in _guide.GROUPS if out.index(f"{other}:") > out.index(f"{title}:")]
        section = section[:min(later) - out.index(f"{title}:")] if later else section[:section.index("Options:")]
        for name in group:
            assert f"\n  {name} " in section, f"{name} is not listed under {title}"


def test_every_command_has_a_worked_example_under_its_help(capsys):
    parser = top_commands()
    for path in _guide.leaf_paths(parser):
        assert path in _guide.EXAMPLES, f"mime {' '.join(path)} has no example"
        assert _guide.EXAMPLES[path].startswith("mime " + " ".join(path))
        with pytest.raises(SystemExit):
            mime_main([*path, "--help"])
        out = said(capsys)[0]
        assert f"example:\n  {_guide.EXAMPLES[path]}" in out, f"mime {' '.join(path)} --help shows no example"


def test_an_example_names_only_commands_that_exist():
    parser = top_commands()
    assert set(_guide.EXAMPLES) <= set(_guide.leaf_paths(parser))


@pytest.mark.parametrize(("words", "hint"), [
    (["verfy", "folder"], "did you mean verify?"),
    (["regsitry"], "did you mean registry?"),
    (["assets", "fetxh"], "did you mean fetch?"),
    (["docs", "cehck"], "did you mean check?"),
    (["run", "folder", "--duraton", "1s", "--out", "o"], "did you mean --duration?"),
    (["run", "folder", "--duration", "1s", "--out", "o", "--histroy"], "did you mean --history?"),
    (["push", "folder", "--to", "zenodo", "--overwite-site-edits"], "did you mean --overwrite-site-edits?"),
    (["verify", "folder", "--tracback"], "did you mean --traceback?"),
])
def test_a_mistyped_command_or_flag_gets_the_one_nearest(capsys, words, hint):
    assert hint in refused_parse(capsys, *words)


@pytest.mark.parametrize("words", [["zzzzzz"], ["run", "folder", "--out", "o", "--qqqqqq", "1"]])
def test_a_word_near_nothing_gets_no_suggestion(capsys, words):
    """The control: the suggestion is a match found, not a line every error carries."""
    assert "did you mean" not in refused_parse(capsys, *words)


def test_a_suggestion_never_repeats_the_word_that_was_typed(capsys):
    """A token typed where a flag belongs is not written out by the suggestion (argparse's own line is its business)."""
    err = refused_parse(capsys, "run", "folder", "--out", "o", "--duraton", "1s")
    hint = [line for line in err.splitlines() if "did you mean" in line]
    assert hint == ["  did you mean --duration?"]


def test_bare_mime_says_what_one_can_do_and_exits_zero(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert mime_main([]) == 0
    out, err = said(capsys)
    assert err == "" and "mime run <folder> --duration 10s --out <dir>" in out and "mime pull" in out
    assert "neither a saved experiment nor a run" in out and "mime --help" in out


def test_bare_mime_in_a_saved_experiments_folder_names_it_and_its_commands(capsys, tmp_path, monkeypatch):
    (tmp_path / DOCUMENT_FILE).write_text("schema: mime.document/4\nexperiment: lab_screw\n", encoding="utf-8")
    (tmp_path / LOCK_FILE).write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert mime_main([]) == 0
    out = said(capsys)[0]
    assert "the saved experiment lab_screw" in out
    for command in ("mime run .", "mime verify .", "mime pending .", "mime serve .", "mime pack ."):
        assert command in out


def test_bare_mime_in_a_run_folder_names_the_experiment_and_how_to_read_the_result(capsys, tmp_path, monkeypatch):
    (tmp_path / "outputs.json").write_text(json.dumps({"experiment": "lab_screw"}), encoding="utf-8")
    (tmp_path / "outputs.npz").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    assert mime_main([]) == 0
    out = said(capsys)[0]
    assert "a run of lab_screw" in out and "RunResult.read('.')" in out and "mime verify experiment" in out


def test_bare_mime_never_writes_a_foreign_name_as_anything_but_plain_text(capsys, tmp_path, monkeypatch):
    """A folder from a stranger names its experiment; an escape in the name does not reach the terminal."""
    (tmp_path / DOCUMENT_FILE).write_text("experiment: \"x\\e[31mred\"\n".replace("\\e", "\x1b"), encoding="utf-8")
    (tmp_path / LOCK_FILE).write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert mime_main([]) == 0
    out = said(capsys)[0]
    assert "\x1b" not in out and "\\x1b[31mred" in out


def test_the_plain_flag_is_accepted_anywhere_and_changes_no_parse(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert mime_main(["--plain"]) == 0
    first = said(capsys)[0]
    assert mime_main(["--plain", "--traceback"]) == 0
    assert said(capsys)[0] == first


#: Names a folder from a stranger might hold: a shell reads each of these as more than a word.
HOSTILE_NAMES = ["x;touch pwned;$(id)", "x;touch${IFS}pwned;$(id)", "a b", "it's", '"quoted"', "`id`", "$HOME", "a&b|c>d", "x\\y", "tab\there", "-rf",
                 "ünï"]


def pasted_into_a_shell(command: str, directory) -> tuple[list[str], bool]:
    """``command`` (as printed, ``mime`` first) run by ``sh`` with ``mime`` standing for a program that lists its
    arguments: the words it got, and whether the line made a file ``pwned`` as a hostile one would."""
    import subprocess

    done = subprocess.run(["sh", "-c", "mime() { for word in \"$@\"; do printf '%s\\n' \"$word\"; done; }\n" + command],
                          capture_output=True, text=True, cwd=directory, check=False)
    return done.stdout.splitlines(), (directory / "pwned").exists()


def command_line_of(out: str, start: str) -> str:
    """The command in the line that starts with ``start``: the text before the description column."""
    line = next(line for line in out.splitlines() if line.strip().startswith(start))
    return line[:line.index("      ")].strip() if "      " in line else line.strip()


@pytest.mark.skipif(sys.platform == "win32", reason="the line is for a POSIX shell; the Windows form is tested apart")
@pytest.mark.parametrize("name", HOSTILE_NAMES)
def test_a_command_offered_for_pasting_keeps_a_foreign_name_one_word_and_runs_nothing_it_holds(name, tmp_path):
    folder = tmp_path / "pulled"
    folder.mkdir()
    (folder / DOCUMENT_FILE).write_text(f"experiment: {name}\n", encoding="utf-8")
    (folder / LOCK_FILE).write_text("{}", encoding="utf-8")
    shell = tmp_path / "shell"
    shell.mkdir()
    words, ran = pasted_into_a_shell(command_line_of(_guide.bare(folder), "mime run ."), shell)
    read = name.strip("'\"")  # the folder's document is read by a line pattern, which drops the quotes around a name
    assert words == ["run", ".", "--duration", "10s", "--out", f"../{read}_run"] and not ran


@pytest.mark.skipif(sys.platform == "win32", reason="the line is for a POSIX shell")
def test_the_control_the_old_unquoted_line_does_run_what_the_name_holds(tmp_path):
    """The check above can fail: the line as it was (spaces replaced, nothing quoted) makes the file when pasted."""
    name = "x;touch${IFS}pwned;$(id)"  # no space, which the old line replaced and which would have stopped it
    shell = tmp_path / "shell"
    shell.mkdir()
    old = "mime run . --duration 10s --out ../" + f"{name}_run".replace(" ", "_")
    words, ran = pasted_into_a_shell(old, shell)
    assert ran and words != ["run", ".", "--duration", "10s", "--out", f"../{name}_run"]


@pytest.mark.parametrize(("word", "shown"), [
    ("plain/path_1.txt", "plain/path_1.txt"), ("a b", '"a b"'), ("x;y", '"x;y"'), ("50%", '"50_"'), ('say "hi"', '"say _hi_"'),
    ("C:\\work\\run", "C:\\work\\run"),
])
def test_on_windows_a_word_with_more_than_plain_characters_is_double_quoted_and_the_two_that_cmd_reads_inside_are_replaced(
        word, shown):
    from mime.cli._shell import quoted

    assert quoted(word, windows=True) == shown


@pytest.mark.parametrize("word", ["plain/path", "run-1_a.b", "a b", "x;y", "it's", ""])
def test_on_posix_a_word_is_shlex_quoted(word):
    import shlex

    from mime.cli._shell import quoted

    assert shlex.split(quoted(word, windows=False)) == [word]
