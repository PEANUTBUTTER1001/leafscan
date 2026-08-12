"""arch × dataset 조합 매트릭스 검증 (안전장치 A5).

각 (arch, dataset) 조합에서 모델·데이터가 구성되고 forward 1회가 통과하는지 확인.
미검증 조합을 표로 노출한다. (전체 학습이 아니라 파이프라인 결선 점검.)

    python tests/test_matrix.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.data import build_multihead_data
from core.models import build_multihead_model
from tasks.models_registry import list_archs

DATASETS = ["multilabel_fake"]
if (Path("data") / "index.csv").exists():
    DATASETS.append("index_csv")


def _try_combo(arch, dataset):
    cfg = {"dataset": dataset, "arch": arch, "img_size": 32, "subset": 64,
           "val_ratio": 0.3, "seed": 42, "index_csv": "data/index.csv",
           "split_policy": "resplit_all"}
    tr, va, te, meta = build_multihead_data(cfg)
    model = build_multihead_model(cfg, meta["heads"]).eval()
    x, _y = tr[0]
    with torch.no_grad():
        out = model(x.unsqueeze(0))
    assert isinstance(out, dict) and set(out) == set(meta["heads"])
    return True


def run_matrix():
    archs = list_archs()
    results = {}
    for arch in archs:
        for ds in DATASETS:
            try:
                _try_combo(arch, ds)
                results[(arch, ds)] = "OK"
            except Exception as e:  # noqa: BLE001
                results[(arch, ds)] = f"FAIL: {e}"[:60]
    return archs, results


def test_matrix():
    archs, results = run_matrix()
    fails = {k: v for k, v in results.items() if v != "OK"}
    assert not fails, f"조합 실패: {fails}"


if __name__ == "__main__":
    archs, results = run_matrix()
    print(f"{'arch':20s} " + " ".join(f"{d:16s}" for d in DATASETS))
    for arch in archs:
        row = " ".join(f"{results[(arch,d)]:16s}" for d in DATASETS)
        print(f"{arch:20s} {row}")
    fails = [k for k, v in results.items() if v != "OK"]
    print("\n" + ("[OK] 전 조합 통과" if not fails else f"[FAIL] {fails}"))
