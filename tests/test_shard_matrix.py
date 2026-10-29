"""How many shards a lane runs as (``scripts/shard_matrix.py``, the ``shards`` job of ``.github/workflows/ci.yml``): the
repository variable ``MIME_CI_SHARDS``, a manual run's input over it, else 6; a whole number from 1 to 12, and anything
else refused by name. The split and the completeness check work for each count in the bounds (``tests/_sharding.py``,
``tests/test_sharding.py``)."""

import importlib.util
import json
from pathlib import Path

import _sharding
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_shard_matrix", ROOT / "scripts" / "shard_matrix.py")
assert spec is not None and spec.loader is not None
shard_matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shard_matrix)


def test_the_bounds_and_the_default_are_one_to_twelve_and_six():
    assert (shard_matrix.MINIMUM, shard_matrix.MAXIMUM, shard_matrix.DEFAULT) == (1, 12, 6)


@pytest.mark.parametrize("variable, manual, expected", [
    ("", "", 6), ("  ", "", 6), ("", "  ", 6), ("4", "", 4), (" 4 ", "", 4), ("", "3", 3), ("4", "3", 3),
    ("4", "", 4), ("1", "", 1), ("12", "", 12), ("", "12", 12), ("06", "", 6), ("4", " ", 4),
])
def test_the_input_overrides_the_variable_which_overrides_the_default(variable, manual, expected):
    assert shard_matrix.resolve(variable, manual)[0] == expected


@pytest.mark.parametrize("value", ["0", "13", "100", "-1", "+3", "abc", "4.5", "1e1", "4 5", "0x4", "٣",
                                   "99999999999999999999"])
@pytest.mark.parametrize("where", ["variable", "input"])
def test_a_count_outside_the_bounds_or_not_a_whole_number_is_refused_naming_where_it_came_from(value, where):
    with pytest.raises(shard_matrix.Refused) as refused:
        shard_matrix.resolve(*((value, "") if where == "variable" else ("", value)))
    assert ("MIME_CI_SHARDS" if where == "variable" else "shards input") in str(refused.value)
    assert "1 to 12" in str(refused.value)


def test_the_input_is_what_the_run_was_told_so_a_bad_one_is_refused_and_a_good_one_wins():
    """A bad input is refused whatever the variable holds, and a good input over a bad variable is used: the input is
    what this run was told."""
    with pytest.raises(shard_matrix.Refused):
        shard_matrix.resolve("4", "13")
    assert shard_matrix.resolve("13", "4")[0] == 4


@pytest.mark.parametrize("count", range(1, 13))
def test_the_job_writes_the_count_and_the_matrix_it_stands_for(count, tmp_path, monkeypatch, capsys):
    output = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert shard_matrix.main(["--variable", str(count), "--input", ""]) == 0
    written = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    assert written["count"] == str(count) and json.loads(written["list"]) == list(range(1, count + 1))
    assert f"{count} shards per lane" in capsys.readouterr().out


@pytest.mark.parametrize("variable, manual", [("0", ""), ("13", ""), ("many", ""), ("", "0"), ("", "13"), ("6", "x")])
def test_a_refusal_fails_the_job_with_an_error_and_writes_no_count(variable, manual, tmp_path, monkeypatch, capsys):
    output = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert shard_matrix.main(["--variable", variable, "--input", manual]) == 1
    assert capsys.readouterr().out.startswith("::error title=Shards::")
    assert not output.exists()


@pytest.mark.parametrize("of", range(1, 13))
def test_the_split_gives_every_shard_files_and_every_file_one_shard_for_any_count(of):
    weights = _sharding.load_weights()
    dealt = _sharding.deal(weights, of)
    assert set(dealt) == set(weights) and set(dealt.values()) == set(range(1, of + 1))
    assert dealt == _sharding.deal(dict(reversed(list(weights.items()))), of)
    load = [sum(s for path, s in weights.items() if dealt[path] == shard) for shard in range(1, of + 1)]
    assert max(load) - min(load) <= max(weights.values())
    unlisted = [f"tests/test_unlisted_{k}.py::test_x" for k in range(200)]
    assert {_sharding.shard_of(nodeid, of, dealt) for nodeid in unlisted} <= set(range(1, of + 1))
    if of == 1:
        assert set(dealt.values()) == {1}
