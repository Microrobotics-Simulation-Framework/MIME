"""Tests for the benchmark suite infrastructure."""

import pytest

from mime.benchmarks.suite import (
    B0, B1, B2, B3, B4_T1, B4_T2, B4_T3, B5,
    BenchmarkDefinition,
    BenchmarkStatus,
    BenchmarkSuite,
    all_benchmarks,
    get_benchmark,
)


class TestBenchmarkRegistry:
    def test_all_benchmarks_registered(self):
        benchmarks = all_benchmarks()
        expected_ids = {"B0", "B1", "B2", "B3", "B4-T1", "B4-T2", "B4-T3", "B5"}
        assert set(benchmarks.keys()) == expected_ids

    def test_get_benchmark_by_id(self):
        b = get_benchmark("B1")
        assert b is not None
        assert b.name == "Step-out frequency detection"

    def test_get_nonexistent_returns_none(self):
        assert get_benchmark("B99") is None

    def test_all_have_pass_criterion(self):
        for bid, defn in all_benchmarks().items():
            assert defn.pass_criterion, f"{bid} has no pass criterion"

    def test_all_have_dependencies(self):
        for bid, defn in all_benchmarks().items():
            assert len(defn.dependencies) > 0, f"{bid} has no dependencies"

    def test_all_start_not_implemented(self):
        for bid, defn in all_benchmarks().items():
            assert defn.status == BenchmarkStatus.NOT_IMPLEMENTED

    def test_b4_tiers_have_correct_phases(self):
        assert get_benchmark("B4-T1").phase == "Phase 3"
        assert get_benchmark("B4-T2").phase == "Phase 3"
        assert get_benchmark("B4-T3").phase == "Advanced"


class TestBenchmarkSuite:
    def test_list_benchmarks_ordered(self):
        suite = BenchmarkSuite()
        benchmarks = suite.list_benchmarks()
        ids = [b.benchmark_id for b in benchmarks]
        assert ids == sorted(ids)

    def test_run_unimplemented_raises(self):
        suite = BenchmarkSuite()
        with pytest.raises(NotImplementedError, match="B1"):
            suite.run("B1")

    def test_run_unknown_raises_key_error(self):
        suite = BenchmarkSuite()
        with pytest.raises(KeyError, match="B99"):
            suite.run("B99")

    def test_run_all_returns_not_implemented_for_all(self):
        suite = BenchmarkSuite()
        results = suite.run_all()
        assert len(results) == 8
        for bid, result in results.items():
            assert isinstance(result, str)
            assert "NOT_IMPLEMENTED" in result
