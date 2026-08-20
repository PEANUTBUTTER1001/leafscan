"""Final-test boundary and GUI bridge tests (no training is performed)."""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.test_evaluation import (final_test_command, list_evaluations,
                                  validate_run)


def _run(tmp_path, dataset="index_csv", done=True):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({
        "dataset": dataset, "index_csv": str(tmp_path / "index.csv"),
        "split_json": str(tmp_path / "split.json")}), encoding="utf-8")
    (run / "best.pt").write_bytes(b"checkpoint")
    (run / "status.json").write_text(json.dumps({"state": "done" if done else "running"}),
                                      encoding="utf-8")
    return run


def test_gui_bridge_requires_explicit_worker_command(tmp_path):
    cmd = final_test_command("python", "evaluate_test.py", tmp_path / "run")
    assert cmd == ["python", "evaluate_test.py", "--run", str(tmp_path / "run")]
    rerun = final_test_command("python", "evaluate_test.py", tmp_path / "run",
                               True, "reviewed leakage concern")
    assert "--reevaluate" in rerun and "reviewed leakage concern" in rerun


def test_non_index_csv_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="index_csv"):
        validate_run(_run(tmp_path, dataset="fake"))


def test_unfinished_run_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="완료"):
        validate_run(_run(tmp_path, done=False))


def test_history_only_counts_complete_evaluations(tmp_path):
    run = _run(tmp_path)
    root = run / "test_evaluations"
    (root / "eval_bad").mkdir(parents=True)
    (root / "eval_good").mkdir(parents=True)
    (root / "eval_good" / "test_metrics.json").write_text("{}", encoding="utf-8")
    assert [p.name for p in list_evaluations(run)] == ["eval_good"]
