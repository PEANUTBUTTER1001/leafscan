"""회귀 기준선 (P0).

골든 config를 현재 train_worker.py로 재실행해, 저장된 골든 metrics의
epoch별 loss·acc·val_loss·val_acc 와 소수점 6자리까지 일치하는지 검증한다.

P1~P6 각 단계 종료 시 반드시 통과해야 한다 (fake · simple_cnn · 단일 head 경로).

    python -m pytest tests/test_regression.py
    또는 단독:  python tests/test_regression.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
GOLDEN = BASE / "tests" / "golden"
WORKER = BASE / "train_worker.py"
KEYS = ("loss", "acc", "val_loss", "val_acc")
TOL = 1e-6


def _run_worker(out_dir):
    env = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    import os
    env = dict(os.environ, **env)
    r = subprocess.run(
        [sys.executable, str(WORKER), "--config", str(GOLDEN / "config.json"),
         "--out", str(out_dir)],
        capture_output=True, text=True, encoding="utf-8", env=env)
    assert r.returncode == 0, f"worker failed:\n{r.stdout}\n{r.stderr}"
    return json.loads((Path(out_dir) / "metrics.json").read_text(encoding="utf-8"))


def compare(golden, fresh):
    gh, fh = golden["history"], fresh["history"]
    assert len(gh) == len(fh), f"epoch 수 불일치 golden={len(gh)} fresh={len(fh)}"
    diffs = []
    for ge, fe in zip(gh, fh):
        for k in KEYS:
            d = abs(ge[k] - fe[k])
            if d > TOL:
                diffs.append(f"epoch {ge['epoch']} {k}: golden={ge[k]!r} fresh={fe[k]!r} Δ={d:.2e}")
    return diffs


def test_regression():
    golden = json.loads((GOLDEN / "metrics.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        fresh = _run_worker(td)
    diffs = compare(golden, fresh)
    assert not diffs, "골든 회귀 실패:\n" + "\n".join(diffs)


if __name__ == "__main__":
    golden = json.loads((GOLDEN / "metrics.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        fresh = _run_worker(td)
    diffs = compare(golden, fresh)
    if diffs:
        print("[FAIL] 골든 회귀 실패:")
        print("\n".join(diffs))
        sys.exit(1)
    print(f"[OK] 골든 회귀 통과 — {len(golden['history'])} epoch, "
          f"final_val_acc={golden['final_val_acc']}")
