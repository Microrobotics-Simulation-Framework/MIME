"""``mime.auth``: a token is stored where a commit cannot publish it, read from the environment first, and shown
nowhere. Every test sets ``HOME`` and the config directory to a temporary folder, so the maintainer's real credentials
are never read or written; tokens are generated at run time, and every output a token could reach is searched for it.

The checks are functions of a home folder; the controls run each under a plant (a mode not set, a mask removed, a git
check skipped, the file preferred to the environment, a status that prints the token) twenty times, and each time it
must fail.
"""

import os
import pickle
import secrets
import stat
import sys
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from mime import auth
from mime.auth import _store
from mime.build import BuildRefused, Refusal
from mime.cli import main

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="file modes are POSIX")


def a_token() -> str:
    return "tok" + secrets.token_urlsafe(30)


@contextmanager
def home(root: Path) -> Iterator[Path]:
    """``HOME`` and the config directory in ``root``, and no token in the environment."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HOME", str(root))
        patch.setenv("XDG_CONFIG_HOME", str(root / "config"))
        patch.setenv("APPDATA", str(root / "appdata"))
        patch.setenv("USERPROFILE", str(root))
        for variable in auth.SERVICES.values():
            patch.delenv(variable, raising=False)
        assert root in auth.credentials_path().parents, "the config directory is not in the temporary folder"
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
            auth.credentials_path().unlink(missing_ok=True)
            path = auth.login("zenodo", a_token())
        finally:
            os.umask(old)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, f"umask {umask:03o}"
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700, f"umask {umask:03o}"
    assert not [each for each in path.parent.iterdir() if each != path], "a temporary file was left"


def check_a_wider_directory_is_narrowed(root: Path) -> None:
    directory = auth.credentials_path().parent
    directory.mkdir(parents=True)
    directory.chmod(0o755)
    auth.login("zenodo", a_token())
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def check_nothing_is_written_inside_a_git_working_tree(root: Path) -> None:
    repository = root / "repo"
    (repository / ".git").mkdir(parents=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CONFIG_HOME", str(repository / "deep" / "config"))
        assert refused_codes(lambda: auth.login("zenodo", a_token())) == ["auth.in-git-tree"]
        assert not (repository / "deep").exists(), "the refused login wrote into the working tree"
    assert auth.login("zenodo", a_token()).exists(), "the check refused a folder outside a working tree"


def check_the_environment_overrides_the_file(root: Path) -> None:
    stored, from_environment = a_token(), a_token()
    auth.login("zenodo", stored)
    assert auth.token("zenodo").reveal() == stored and auth.source("zenodo") == "the credentials file"  # type: ignore[union-attr]
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ZENODO_TOKEN", from_environment)
        found = auth.token("zenodo")
        assert found is not None and found.reveal() == from_environment
        assert auth.source("zenodo") == "environment variable ZENODO_TOKEN"
        assert auth.token("zenodo-sandbox") is None, "the sandbox read the production variable"
        patch.setenv("ZENODO_SANDBOX_TOKEN", stored[::-1])
        sandbox = auth.token("zenodo-sandbox")
        assert sandbox is not None and sandbox.reveal() == stored[::-1]


def check_a_token_is_masked_in_every_form(root: Path) -> None:
    secret = a_token()
    token = auth.Token("zenodo", secret)
    forms = [repr(token), str(token), f"{token}", f"{token!r}", "%s %r" % (token, token), format(token, ">40"),
             repr([token]), repr({"zenodo": token}), repr(token.service)]
    assert not [form for form in forms if shown(secret, form)], forms
    with pytest.raises(TypeError) as error:
        pickle.dumps(token)
    assert not shown(secret, str(error.value))
    assert token.last_four() == secret[-4:] and token.reveal() == secret


def check_no_output_or_message_holds_a_token(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    secret = a_token()
    outputs: list[str] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("getpass.getpass", lambda prompt="": secret)
        assert main(["auth", "login", "zenodo"]) == 0
    for argv in (["auth", "status", "zenodo"], ["auth", "status"], ["auth", "logout", "zenodo-sandbox"]):
        main(argv)
    outputs += [captured for captured in capsys.readouterr()]
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ZENODO_TOKEN", secret + "x")
        main(["auth", "status", "zenodo"])
        outputs += list(capsys.readouterr())
    assert not [out for out in outputs if shown(secret, out)], outputs
    # a token with a space in it, a short one, and a file that is not JSON: messages name neither the value nor a part
    messages: list[str] = []
    for bad in (secret + " " + secret, "short", ""):
        try:
            auth.login("zenodo", bad)
        except BuildRefused as refused:
            messages += [str(refused), *traceback.format_exception(refused)]
    path = auth.credentials_path()
    path.write_text('{"zenodo": "' + secret, encoding="utf-8")
    for action in (lambda: auth.token("zenodo"), lambda: auth.login("zenodo", a_token()), lambda: auth.logout("zenodo")):
        try:
            action()
        except BuildRefused as refused:
            messages += [str(refused), *traceback.format_exception(refused)]
    path.write_text('["' + secret + '"]', encoding="utf-8")
    try:
        auth.token("zenodo")
    except BuildRefused as refused:
        messages += [str(refused), *traceback.format_exception(refused)]
    path.write_text('{"zenodo": "' + secret + '"}', encoding="utf-8")
    path.chmod(0o644)
    try:
        auth.token("zenodo")
    except BuildRefused as refused:
        messages += [str(refused), *traceback.format_exception(refused)]
    messages.append(auth.describe("zenodo"))
    assert len(messages) >= 8
    assert not [message for message in messages if shown(secret, message)], messages


def check_status_shows_its_source_and_four_characters(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    secret = a_token()
    auth.login("zenodo", secret)
    assert main(["auth", "status", "zenodo"]) == 0
    line = capsys.readouterr().out
    assert "the credentials file" in line and secret[-4:] in line and not shown(secret, line)
    assert main(["auth", "status", "zenodo-sandbox"]) == 0 and "not logged in" in capsys.readouterr().out


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


def test_logout_deletes_the_token_and_the_file_when_it_is_the_last(folder: Path, capsys: pytest.CaptureFixture[str]):
    first, second = a_token(), a_token()
    auth.login("zenodo", first)
    auth.login("zenodo-sandbox", second)
    assert auth.logout("zenodo") is True
    assert auth.token("zenodo") is None and auth.token("zenodo-sandbox").reveal() == second  # type: ignore[union-attr]
    assert first not in auth.credentials_path().read_text(encoding="utf-8")
    assert auth.logout("zenodo") is False
    assert auth.logout("zenodo-sandbox") is True and not auth.credentials_path().exists()
    assert main(["auth", "logout", "zenodo"]) == 0
    assert "holds no zenodo token" in capsys.readouterr().out


def test_logout_says_when_the_environment_still_holds_a_token(folder: Path, monkeypatch: pytest.MonkeyPatch,
                                                                capsys: pytest.CaptureFixture[str]):
    auth.login("zenodo", a_token())
    monkeypatch.setenv("ZENODO_TOKEN", a_token())
    assert main(["auth", "logout", "zenodo"]) == 0
    assert "still logged in from environment variable ZENODO_TOKEN" in capsys.readouterr().out


def test_login_reads_the_token_with_getpass_and_takes_none_on_the_command_line(folder: Path,
                                                                               monkeypatch: pytest.MonkeyPatch):
    secret = a_token()
    asked: list[str] = []
    monkeypatch.setattr("getpass.getpass", lambda prompt="": asked.append(prompt) or secret + "\n")
    assert main(["auth", "login", "zenodo"]) == 0 and len(asked) == 1
    assert auth.token("zenodo").reveal() == secret  # type: ignore[union-attr]
    with pytest.raises(SystemExit):
        main(["auth", "login", "zenodo", secret])
    with pytest.raises(SystemExit):
        main(["auth", "login", "elsewhere"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": (_ for _ in ()).throw(EOFError()))
    assert main(["auth", "login", "zenodo"]) == 2


def test_a_file_other_users_can_read_is_not_used_and_status_says_so(folder: Path, capsys: pytest.CaptureFixture[str]):
    secret = a_token()
    path = auth.login("zenodo", secret)
    path.chmod(0o644)
    assert refused_codes(lambda: auth.token("zenodo")) == ["auth.file-mode"]
    assert main(["auth", "status", "zenodo"]) == 0
    out = capsys.readouterr().out
    assert "warning" in out and "chmod 600" in out
    assert secret in auth.held_values(), "the pack's search must still know a token whose file is too open"


def test_the_values_a_pack_searches_for(folder: Path, monkeypatch: pytest.MonkeyPatch):
    stored, from_environment = a_token(), a_token()
    auth.login("zenodo", stored)
    monkeypatch.setenv("ZENODO_SANDBOX_TOKEN", from_environment)
    monkeypatch.setenv("ZENODO_TOKEN", "short")
    assert sorted(auth.held_values()) == sorted({stored, from_environment}), "a value under eight characters is left out"


def test_an_unreadable_file_does_not_stop_the_pack_search(folder: Path):
    auth.credentials_path().parent.mkdir(parents=True)
    auth.credentials_path().write_text("not json", encoding="utf-8")
    assert auth.held_values() == []


def test_a_credentials_file_that_is_not_a_service_to_token_object_is_refused(folder: Path):
    path = auth.credentials_path()
    path.parent.mkdir(parents=True)
    for text in ("not json", "[1, 2]", '{"zenodo": 3}'):
        path.write_text(text, encoding="utf-8")
        assert refused_codes(lambda: auth.token("zenodo")) == ["auth.store"]
        assert refused_codes(lambda: auth.login("zenodo", a_token())) == ["auth.store"]


def test_an_unknown_service_is_refused(folder: Path):
    assert refused_codes(lambda: auth.token("elsewhere")) == ["auth.service"]
    assert refused_codes(lambda: auth.login("elsewhere", a_token())) == ["auth.service"]


def test_the_real_config_directory_is_never_touched(tmp_path: Path):
    """The guard of the helper above: outside it, the config directory is the maintainer's; inside, it is not."""
    with home(tmp_path):
        assert auth.credentials_path().is_relative_to(tmp_path)


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
        patch.setattr(auth.Token, "__repr__", lambda self: f"Token({self.reveal()})")
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
        patch.setattr(auth.Token, "last_four", lambda self: self.reveal())
        yield


@contextmanager
def a_refusal_that_quotes_the_value() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        honest = auth.login

        def quoting(service: str, value: str) -> Path:
            try:
                return honest(service, value)
            except BuildRefused as refused:
                raise BuildRefused([Refusal(each.code, each.category, f"{each.message} (read {value!r})", each.action)
                                    for each in refused.refusals]) from None

        patch.setattr(auth, "login", quoting)
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
