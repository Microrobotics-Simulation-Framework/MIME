"""``mime.credentials``: a token is stored where a commit cannot publish it, read from the environment first, and shown
nowhere. Every test sets ``HOME`` and the config directory to a temporary folder, and takes git's variables and
configuration out of reach, so the maintainer's real credentials and git configuration are never read or written; tokens
are generated at run time, and every output a token could reach (a message, an output, a traceback captured with its
locals) is searched for it.

The checks are functions of a home folder; the controls run each under a plant (a mode not set, a mask removed, a git
check skipped, the file preferred to the environment, a status that prints the token) twenty times, and each time it
must fail.
"""

import os
import pickle
import secrets
import stat
import subprocess
import sys
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from mime import credentials
from mime.build import BuildRefused, Refusal
from mime.cli import FAILED, main
from mime.cli import _credentials as credentials_command
from mime.credentials import _store, _token

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="file modes are POSIX")


def a_token() -> str:
    return "tok" + secrets.token_urlsafe(30)


def put(service: str, value: str) -> Path:
    return credentials.store(service, credentials.Token(service, value))


GIT_VARIABLES = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_CEILING_DIRECTORIES",
                 "GIT_DISCOVERY_ACROSS_FILESYSTEM")


@contextmanager
def home(root: Path) -> Iterator[Path]:
    """``HOME`` and the config directory in ``root``, and no token in the environment."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HOME", str(root))
        patch.setenv("XDG_CONFIG_HOME", str(root / "config"))
        patch.setenv("APPDATA", str(root / "appdata"))
        patch.setenv("USERPROFILE", str(root))
        for variable in (*credentials.SERVICES.values(), *GIT_VARIABLES):
            patch.delenv(variable, raising=False)
        patch.setenv("GIT_CONFIG_GLOBAL", str(root / "gitconfig"))
        patch.setenv("GIT_CONFIG_NOSYSTEM", "1")
        assert root in credentials.credentials_path().parents, "the config directory is not in the temporary folder"
        yield root


@pytest.fixture
def folder(tmp_path: Path) -> Iterator[Path]:
    with home(tmp_path):
        yield tmp_path


def refused_codes(action: Callable[[], object]) -> list[str]:
    try:
        action()
    except BuildRefused as refused:
        return refused.codes()
    raise AssertionError("not refused")


def shown(secret: str, text: str) -> bool:
    """Whether any part of ``secret`` long enough to be it (all but its last four characters) is in ``text``."""
    return secret in text or secret[:-4] in text or secret[:12] in text


# --- checks ----------------------------------------------------------------------------------------------------------


def check_the_file_is_private_and_so_is_its_directory(root: Path) -> None:
    for umask in (0o022, 0o000):
        old = os.umask(umask)
        try:
            credentials.credentials_path().unlink(missing_ok=True)
            path = put("zenodo", a_token())
        finally:
            os.umask(old)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, f"umask {umask:03o}"
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700, f"umask {umask:03o}"
    assert not [each for each in path.parent.iterdir() if each != path], "a temporary file was left"


def check_a_wider_directory_is_narrowed(root: Path) -> None:
    directory = credentials.credentials_path().parent
    directory.mkdir(parents=True)
    directory.chmod(0o755)
    put("zenodo", a_token())
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def check_nothing_is_written_inside_a_git_working_tree(root: Path) -> None:
    repository = root / "repo"
    (repository / ".git").mkdir(parents=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CONFIG_HOME", str(repository / "deep" / "config"))
        assert refused_codes(lambda: put("zenodo", a_token())) == ["credentials.in-git-tree"]
        assert not (repository / "deep").exists(), "the refused store wrote into the working tree"
    assert put("zenodo", a_token()).exists(), "the check refused a folder outside a working tree"


def check_the_environment_overrides_the_file(root: Path) -> None:
    stored, from_environment = a_token(), a_token()
    put("zenodo", stored)
    assert credentials.token("zenodo").reveal() == stored and credentials.source("zenodo") == "the credentials file"  # type: ignore[union-attr]
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ZENODO_TOKEN", from_environment)
        found = credentials.token("zenodo")
        assert found is not None and found.reveal() == from_environment
        assert credentials.source("zenodo") == "environment variable ZENODO_TOKEN"
        assert credentials.token("zenodo-sandbox") is None, "the sandbox read the production variable"
        patch.setenv("ZENODO_SANDBOX_TOKEN", stored[::-1])
        sandbox = credentials.token("zenodo-sandbox")
        assert sandbox is not None and sandbox.reveal() == stored[::-1]


def check_a_token_is_masked_in_every_form(root: Path) -> None:
    secret = a_token()
    token = credentials.Token("zenodo", secret)
    forms = [repr(token), str(token), f"{token}", f"{token!r}", "%s %r" % (token, token), format(token, ">40"),
             repr([token]), repr({"zenodo": token}), repr(token.service)]
    assert not [form for form in forms if shown(secret, form)], forms
    with pytest.raises(TypeError) as error:
        pickle.dumps(token)
    assert not shown(secret, str(error.value))
    assert token.last_four() == secret[-4:] and token.reveal() == secret
    assert credentials.Token("zenodo", secret[:20]).last_four() == "", "a short token showed a part of itself"


def check_no_output_or_message_holds_a_token(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    secret = a_token()
    outputs: list[str] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("getpass.getpass", lambda prompt="": secret)
        assert main(["credentials", "set", "zenodo"]) == 0
    for argv in (["credentials", "status", "zenodo"], ["credentials", "status"], ["credentials", "remove", "zenodo-sandbox"]):
        main(argv)
    outputs += [captured for captured in capsys.readouterr()]
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ZENODO_TOKEN", secret + "x")
        main(["credentials", "status", "zenodo"])
        outputs += list(capsys.readouterr())
    assert not [out for out in outputs if shown(secret, out)], outputs
    # a token with a space in it, a short one, and a file that is not JSON: messages name neither the value nor a part
    messages: list[str] = []
    for bad in (secret + " " + secret, "short", ""):
        try:
            put("zenodo", bad)
        except BuildRefused as refused:
            messages += [str(refused), *traceback.format_exception(refused)]
    path = credentials.credentials_path()
    path.write_text('{"zenodo": "' + secret, encoding="utf-8")
    for action in (lambda: credentials.token("zenodo"), lambda: put("zenodo", a_token()), lambda: credentials.remove("zenodo")):
        try:
            action()
        except BuildRefused as refused:
            messages += [str(refused), *traceback.format_exception(refused)]
    path.write_text('["' + secret + '"]', encoding="utf-8")
    try:
        credentials.token("zenodo")
    except BuildRefused as refused:
        messages += [str(refused), *traceback.format_exception(refused)]
    path.write_text('{"zenodo": "' + secret + '"}', encoding="utf-8")
    path.chmod(0o644)
    try:
        credentials.token("zenodo")
    except BuildRefused as refused:
        messages += [str(refused), *traceback.format_exception(refused)]
    messages.append(credentials.describe("zenodo"))
    assert len(messages) >= 8
    assert not [message for message in messages if shown(secret, message)], messages


def check_status_shows_its_source_and_four_characters(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    secret = a_token()
    put("zenodo", secret)
    assert main(["credentials", "status", "zenodo"]) == 0
    line = capsys.readouterr().out
    assert "the credentials file" in line and f"ending {secret[-4:]}" in line and not shown(secret, line)
    assert main(["credentials", "status", "zenodo-sandbox"]) == 0 and "no token" in capsys.readouterr().out


def check_status_shows_nothing_of_a_short_token(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for length in (credentials.MINIMUM_LENGTH, 16, credentials.SHOWN_FROM_LENGTH - 1):
        short = "k" * (length - 4) + secrets.token_hex(2)
        put("zenodo", short)
        assert main(["credentials", "status", "zenodo"]) == 0
        line = capsys.readouterr().out
        assert short[-4:] not in line.replace("zenodo", "") and "too short to show" in line, line
        assert credentials.describe("zenodo") == line.rstrip("\n")
    long = "k" * (credentials.SHOWN_FROM_LENGTH - 4) + "wxyz"
    put("zenodo", long)
    assert main(["credentials", "status", "zenodo"]) == 0
    assert "ending wxyz" in capsys.readouterr().out, "a token of 32 characters shows its last four"


def captured_with_locals(error: BaseException) -> str:
    """What a tool that prints a traceback with its locals (pytest's ``--showlocals``, ``rich``) would show."""
    return "".join(traceback.TracebackException.from_exception(error, capture_locals=True).format())


def check_no_traceback_captured_with_its_locals_shows_a_token(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    secret = a_token()
    shown_in: list[str] = []

    def collect(action: Callable[[], object], expected: type[BaseException]) -> None:
        try:
            action()
        except expected as error:
            shown_in.append(captured_with_locals(error))
            return
        raise AssertionError("the action did not raise")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("getpass.getpass", lambda prompt="": secret + "\n")

        def refuse_to_open(*arguments: object, **keywords: object) -> int:
            raise OSError("the disk is full")

        with pytest.MonkeyPatch.context() as inner:
            inner.setattr(_store.os, "open", refuse_to_open)
            collect(lambda: credentials_command("set", "zenodo"), OSError)  # past every frame that held the token
            capsys.readouterr()
            assert main(["credentials", "set", "zenodo"]) == FAILED, "the command shows the error and exits"
            assert not shown(secret, capsys.readouterr().err), "the command's error showed the token"
    # the actions hold the token only wrapped: a closure over the text would put it in the frame of the action
    wrapped, doubled = credentials.Token("zenodo", secret), credentials.Token("zenodo", secret + " " + secret)
    collect(lambda: credentials.store("zenodo", doubled), BuildRefused)
    collect(lambda: credentials.token("elsewhere"), BuildRefused)
    path = credentials.store("zenodo", wrapped)
    path.chmod(0o644)
    collect(lambda: credentials.token("zenodo"), BuildRefused)
    path.write_text('{"zenodo": "' + secret, encoding="utf-8")
    collect(lambda: credentials.token("zenodo"), BuildRefused)
    path.write_text('{"zenodo": "' + secret + '", "other": 5}', encoding="utf-8")
    collect(lambda: credentials.remove("zenodo"), BuildRefused)
    assert len(shown_in) == 6
    assert not [text for text in shown_in if shown(secret, text)], "a captured traceback showed the token"
    assert all("Token" in text or "masked" in text or "elsewhere" in text or "BuildRefused" in text for text in shown_in)


# --- the tests -------------------------------------------------------------------------------------------------------


def test_the_file_and_its_directory_are_private(folder: Path):
    check_the_file_is_private_and_so_is_its_directory(folder)


def test_a_wider_directory_is_narrowed(folder: Path):
    check_a_wider_directory_is_narrowed(folder)


def test_nothing_is_written_inside_a_git_working_tree(folder: Path):
    check_nothing_is_written_inside_a_git_working_tree(folder)


def test_the_environment_overrides_the_file(folder: Path):
    check_the_environment_overrides_the_file(folder)


def test_a_token_is_masked_in_every_form(folder: Path):
    check_a_token_is_masked_in_every_form(folder)


def test_no_output_or_message_holds_a_token(folder: Path, capsys: pytest.CaptureFixture[str]):
    check_no_output_or_message_holds_a_token(folder, capsys)


def test_status_shows_its_source_and_four_characters(folder: Path, capsys: pytest.CaptureFixture[str]):
    check_status_shows_its_source_and_four_characters(folder, capsys)


def test_status_shows_nothing_of_a_token_shorter_than_32_characters(folder: Path, capsys: pytest.CaptureFixture[str]):
    check_status_shows_nothing_of_a_short_token(folder, capsys)


def test_a_traceback_captured_with_its_locals_shows_no_token(folder: Path, capsys: pytest.CaptureFixture[str]):
    check_no_traceback_captured_with_its_locals_shows_a_token(folder, capsys)


def check_no_token_is_echoed_by_the_argument_parser(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    secret = a_token()
    asked = 0
    for argv in (["credentials", "set", "zenodo", secret], ["credentials", "set", secret],
                 ["credentials", "status", "zenodo", secret], ["credentials", "remove", "zenodo", secret],
                 ["credentials", secret], ["credentials", "set", f"--token={secret}"],
                 ["credentials", "set", "zenodo", "--token", secret], ["credentials", secret, "zenodo"],
                 ["credentials", "status", secret]):
        with pytest.raises(SystemExit) as exit_:
            main(argv)
        assert exit_.value.code == 2, argv
        out, err = capsys.readouterr()
        assert not shown(secret, out + err) and secret[-8:] not in out + err, (argv, err)
        assert "never an argument" in err, "the refusal does not say where a token goes"
        asked += 1
    assert asked == 9 and not credentials.credentials_path().exists(), "a refused command stored something"


def test_no_token_is_echoed_by_the_argument_parser(folder: Path, capsys: pytest.CaptureFixture[str]):
    check_no_token_is_echoed_by_the_argument_parser(folder, capsys)


def test_help_still_works(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exit_:
        main(["credentials", "set", "-h"])
    assert exit_.value.code == 0 and "zenodo-sandbox" in capsys.readouterr().out


def make_dotfiles_repository(root: Path, *, core_worktree: bool) -> Path:
    """A bare repository outside the home directory, used as a dotfiles repository with its work tree the home."""
    root.mkdir(parents=True, exist_ok=True)
    bare = root.parent / f"{root.name}-dotfiles.git"
    environment = {**os.environ, "GIT_CONFIG_GLOBAL": str(root / "gitconfig"), "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True, env=environment)
    if core_worktree:
        for key, value in (("core.bare", "false"), ("core.worktree", str(root))):
            subprocess.run(["git", "--git-dir", str(bare), "config", key, value], check=True, env=environment)
    return bare


def check_a_dotfiles_repository_named_by_its_work_tree_is_refused(root: Path) -> None:
    """The environment alone: with no ``git`` to ask, the work tree the environment names is still found."""
    bare = make_dotfiles_repository(root, core_worktree=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_store, "_git_says_inside", lambda directory: None)
        patch.setenv("GIT_DIR", str(bare))
        patch.setenv("GIT_WORK_TREE", str(root))
        assert refused_codes(lambda: put("zenodo", a_token())) == ["credentials.in-git-tree"]
    assert not credentials.credentials_path().exists(), "the refused store wrote the file"


def check_a_dotfiles_repository_git_itself_finds_is_refused(root: Path) -> None:
    bare = make_dotfiles_repository(root, core_worktree=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("GIT_DIR", str(bare))
        assert refused_codes(lambda: put("zenodo", a_token())) == ["credentials.in-git-tree"]
    assert not credentials.credentials_path().exists(), "the refused store wrote the file"


def check_a_work_tree_elsewhere_is_not_a_reason_to_refuse(root: Path) -> None:
    bare = make_dotfiles_repository(root, core_worktree=False)
    elsewhere = root / "elsewhere"
    elsewhere.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("GIT_DIR", str(bare))
        patch.setenv("GIT_WORK_TREE", str(elsewhere))
        assert put("zenodo", a_token()).exists()


def test_a_dotfiles_repository_used_with_a_work_tree_is_refused(folder: Path):
    check_a_dotfiles_repository_named_by_its_work_tree_is_refused(folder)


def test_a_dotfiles_repository_git_finds_through_its_own_configuration_is_refused(tmp_path: Path):
    with home(tmp_path / "second"):
        check_a_dotfiles_repository_git_itself_finds_is_refused(tmp_path / "second")


def test_a_work_tree_that_does_not_hold_the_config_directory_is_not_a_reason_to_refuse(tmp_path: Path):
    with home(tmp_path / "third"):
        check_a_work_tree_elsewhere_is_not_a_reason_to_refuse(tmp_path / "third")


def test_remove_deletes_the_token_and_the_file_when_it_is_the_last(folder: Path, capsys: pytest.CaptureFixture[str]):
    first, second = a_token(), a_token()
    put("zenodo", first)
    put("zenodo-sandbox", second)
    assert credentials.remove("zenodo") is True
    assert credentials.token("zenodo") is None and credentials.token("zenodo-sandbox").reveal() == second  # type: ignore[union-attr]
    assert first not in credentials.credentials_path().read_text(encoding="utf-8")
    assert credentials.remove("zenodo") is False
    assert credentials.remove("zenodo-sandbox") is True and not credentials.credentials_path().exists()
    assert main(["credentials", "remove", "zenodo"]) == 0
    assert "holds no zenodo token" in capsys.readouterr().out


def test_remove_says_when_the_environment_still_holds_a_token(folder: Path, monkeypatch: pytest.MonkeyPatch,
                                                                capsys: pytest.CaptureFixture[str]):
    put("zenodo", a_token())
    monkeypatch.setenv("ZENODO_TOKEN", a_token())
    assert main(["credentials", "remove", "zenodo"]) == 0
    assert "a token is still set by environment variable ZENODO_TOKEN" in capsys.readouterr().out


def test_set_reads_the_token_with_getpass_and_takes_none_on_the_command_line(folder: Path,
                                                                               monkeypatch: pytest.MonkeyPatch):
    secret = a_token()
    asked: list[str] = []
    monkeypatch.setattr("getpass.getpass", lambda prompt="": asked.append(prompt) or secret + "\n")
    assert main(["credentials", "set", "zenodo"]) == 0 and len(asked) == 1
    assert credentials.token("zenodo").reveal() == secret  # type: ignore[union-attr]
    with pytest.raises(SystemExit):
        main(["credentials", "set", "elsewhere"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": (_ for _ in ()).throw(EOFError()))
    assert main(["credentials", "set", "zenodo"]) == 2


def test_a_file_other_users_can_read_is_not_used_and_status_says_so(folder: Path, capsys: pytest.CaptureFixture[str]):
    secret = a_token()
    path = put("zenodo", secret)
    path.chmod(0o644)
    assert refused_codes(lambda: credentials.token("zenodo")) == ["credentials.file-mode"]
    assert main(["credentials", "status", "zenodo"]) == 0
    out = capsys.readouterr().out
    assert "warning" in out and "chmod 600" in out
    assert secret in [each.reveal() for each in credentials.held_tokens()], "the pack's search must still know a token whose file is too open"


def test_the_values_a_pack_searches_for(folder: Path, monkeypatch: pytest.MonkeyPatch):
    stored, from_environment = a_token(), a_token()
    put("zenodo", stored)
    monkeypatch.setenv("ZENODO_SANDBOX_TOKEN", from_environment)
    monkeypatch.setenv("ZENODO_TOKEN", "short")
    assert sorted([each.reveal() for each in credentials.held_tokens()]) == sorted({stored, from_environment}), "a value under eight characters is left out"


def test_an_unreadable_file_does_not_stop_the_pack_search(folder: Path):
    credentials.credentials_path().parent.mkdir(parents=True)
    credentials.credentials_path().write_text("not json", encoding="utf-8")
    assert [each.reveal() for each in credentials.held_tokens()] == []


def test_a_credentials_file_that_is_not_a_service_to_token_object_is_refused(folder: Path):
    path = credentials.credentials_path()
    path.parent.mkdir(parents=True)
    for text in ("not json", "[1, 2]", '{"zenodo": 3}'):
        path.write_text(text, encoding="utf-8")
        assert refused_codes(lambda: credentials.token("zenodo")) == ["credentials.store"]
        assert refused_codes(lambda: put("zenodo", a_token())) == ["credentials.store"]


def test_an_unknown_service_is_refused(folder: Path):
    assert refused_codes(lambda: credentials.token("elsewhere")) == ["credentials.service"]
    assert refused_codes(lambda: put("elsewhere", a_token())) == ["credentials.service"]


def test_the_real_config_directory_is_never_touched(tmp_path: Path):
    """The guard of the helper above: outside it, the config directory is the maintainer's; inside, it is not."""
    with home(tmp_path):
        assert credentials.credentials_path().is_relative_to(tmp_path)


# --- the controls ----------------------------------------------------------------------------------------------------


@contextmanager
def mode_not_set() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_store, "FILE_MODE", 0o644)
        yield


@contextmanager
def directory_mode_not_set() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_store, "DIRECTORY_MODE", 0o755)
        yield


@contextmanager
def mask_removed() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(credentials.Token, "__repr__", lambda self: f"Token({self.reveal()})")
        yield


@contextmanager
def git_check_skipped() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_store, "inside_git_tree", lambda path: None)
        yield


@contextmanager
def file_preferred_to_the_environment() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        honest = _store._from_environment
        patch.setattr(_store, "_from_environment", lambda service: honest(service) if service == "zenodo-sandbox" else None)
        yield


@contextmanager
def status_that_prints_the_token() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(credentials.Token, "last_four", lambda self: self.reveal())
        yield


@contextmanager
def a_refusal_that_quotes_the_value() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        honest = credentials.store

        def quoting(service: str, value: credentials.Token) -> Path:
            try:
                return honest(service, value)
            except BuildRefused as refused:
                raise BuildRefused([Refusal(each.code, each.category, f"{each.message} (read {value.reveal()!r})",
                                            each.action) for each in refused.refusals]) from None

        patch.setattr(credentials, "store", quoting)
        yield


@contextmanager
def a_raw_local_in_the_store() -> Iterator[None]:
    """The error the wrapper guards against: the text of a token under a name of a frame the exception passes through."""
    with pytest.MonkeyPatch.context() as patch:
        honest = credentials.store

        def holding(service: str, value: credentials.Token) -> Path:
            raw = value.reveal()  # noqa: F841 - the plant: a plain local in the traceback
            return honest(service, value)

        patch.setattr(credentials, "store", holding)
        yield


@contextmanager
def the_parser_left_to_echo() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        import mime.cli

        patch.setattr(mime.cli, "_credential_words_problem", lambda after: None)
        yield


@contextmanager
def the_named_work_tree_skipped() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_store, "_named_work_tree", lambda path: None)
        yield


@contextmanager
def git_never_asked() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_store, "_git_says_inside", lambda directory: None)
        yield


@contextmanager
def a_short_token_shown() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_token, "SHOWN_FROM_LENGTH", 0)
        yield


RUNS = 20


def planted_runs(tmp_path: Path, check: Callable[..., None], plant: Callable[[], object],
                 capsys: pytest.CaptureFixture[str] | None = None) -> int:
    detected = 0
    for run in range(RUNS):
        try:
            with home(tmp_path / str(run)), plant():  # type: ignore[attr-defined]
                if capsys is None:
                    check(tmp_path / str(run))
                else:
                    check(tmp_path / str(run), capsys)
        except AssertionError:
            detected += 1
        if capsys is not None:
            capsys.readouterr()
    return detected


def test_the_checks_pass_unplanted_twenty_times(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    for run in range(RUNS):
        for number, check in enumerate([check_the_file_is_private_and_so_is_its_directory,
                                        check_a_wider_directory_is_narrowed,
                                        check_nothing_is_written_inside_a_git_working_tree,
                                        check_the_environment_overrides_the_file,
                                        check_a_token_is_masked_in_every_form]):
            with home(tmp_path / f"{run}-{number}"):
                check(tmp_path / f"{run}-{number}")
        with home(tmp_path / f"s{run}"):
            check_status_shows_its_source_and_four_characters(tmp_path, capsys)
        with home(tmp_path / f"o{run}"):
            check_no_output_or_message_holds_a_token(tmp_path, capsys)


def test_control_a_file_mode_not_set_is_detected_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, check_the_file_is_private_and_so_is_its_directory, mode_not_set) == RUNS


def test_control_a_directory_mode_not_set_is_detected_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, check_the_file_is_private_and_so_is_its_directory, directory_mode_not_set) == RUNS
    assert planted_runs(tmp_path / "wider", check_a_wider_directory_is_narrowed, directory_mode_not_set) == RUNS


def test_control_a_mask_removed_is_detected_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, check_a_token_is_masked_in_every_form, mask_removed) == RUNS


def test_control_a_skipped_git_check_is_detected_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, check_nothing_is_written_inside_a_git_working_tree, git_check_skipped) == RUNS


def test_control_the_file_preferred_to_the_environment_is_detected_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, check_the_environment_overrides_the_file, file_preferred_to_the_environment) == RUNS


def test_control_a_refusal_that_quotes_the_value_is_detected_every_run(tmp_path: Path,
                                                                      capsys: pytest.CaptureFixture[str]):
    assert planted_runs(tmp_path, check_no_output_or_message_holds_a_token, a_refusal_that_quotes_the_value,
                        capsys) == RUNS


def test_control_a_status_that_prints_the_token_is_detected_every_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert planted_runs(tmp_path, check_status_shows_its_source_and_four_characters, status_that_prints_the_token,
                        capsys) == RUNS


def test_control_a_raw_local_in_a_traceback_is_detected_every_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert planted_runs(tmp_path, check_no_traceback_captured_with_its_locals_shows_a_token, a_raw_local_in_the_store,
                        capsys) == RUNS


def test_control_an_argument_parser_left_to_echo_is_detected_every_run(tmp_path: Path,
                                                                      capsys: pytest.CaptureFixture[str]):
    assert planted_runs(tmp_path, check_no_token_is_echoed_by_the_argument_parser, the_parser_left_to_echo,
                        capsys) == RUNS


def test_control_a_skipped_work_tree_check_is_detected_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, check_a_dotfiles_repository_named_by_its_work_tree_is_refused,
                        the_named_work_tree_skipped) == RUNS


def test_control_git_never_asked_is_detected_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, check_a_dotfiles_repository_git_itself_finds_is_refused, git_never_asked) == RUNS


def test_control_a_short_token_shown_is_detected_every_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert planted_runs(tmp_path, check_status_shows_nothing_of_a_short_token, a_short_token_shown, capsys) == RUNS


def test_the_new_checks_pass_unplanted_twenty_times(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    for run in range(RUNS):
        for name, check in (("t", check_no_traceback_captured_with_its_locals_shows_a_token),
                            ("a", check_no_token_is_echoed_by_the_argument_parser),
                            ("s", check_status_shows_nothing_of_a_short_token)):
            with home(tmp_path / f"{name}{run}"):
                check(tmp_path / f"{name}{run}", capsys)
        for name, git_check in (("w", check_a_dotfiles_repository_named_by_its_work_tree_is_refused),
                                ("g", check_a_dotfiles_repository_git_itself_finds_is_refused),
                                ("e", check_a_work_tree_elsewhere_is_not_a_reason_to_refuse)):
            with home(tmp_path / f"{name}{run}"):
                git_check(tmp_path / f"{name}{run}")
