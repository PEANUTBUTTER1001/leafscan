"""P6 지표 검증 — 손계산 소규모 예제와 core.metrics 결과 일치.

    python tests/test_metrics.py   또는   pytest tests/test_metrics.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.metrics import head_metrics


def _logits_from_preds(preds, n_classes):
    """argmax 가 preds 가 되도록 하는 간단한 로짓."""
    lg = np.full((len(preds), n_classes), -5.0)
    for i, p in enumerate(preds):
        lg[i, p] = 5.0
    return lg


def test_hand_example():
    # 손계산: labels=[0,0,1,1,2], preds=[0,1,1,1,2]
    labels = np.array([0, 0, 1, 1, 2])
    preds = [0, 1, 1, 1, 2]
    logits = _logits_from_preds(preds, 3)
    m = head_metrics(logits, labels, ["a", "b", "c"])

    assert abs(m["accuracy"] - 0.8) < 1e-6, m["accuracy"]
    pc = m["per_class"]
    # class a: P=1.0 R=0.5 F1=0.6667
    assert abs(pc["a"]["precision"] - 1.0) < 1e-4
    assert abs(pc["a"]["recall"] - 0.5) < 1e-4
    assert abs(pc["a"]["f1"] - 0.6667) < 1e-3
    # class b: P=0.6667 R=1.0 F1=0.8
    assert abs(pc["b"]["precision"] - 0.6667) < 1e-3
    assert abs(pc["b"]["recall"] - 1.0) < 1e-4
    assert abs(pc["b"]["f1"] - 0.8) < 1e-3
    # class c: 완전 일치
    assert abs(pc["c"]["f1"] - 1.0) < 1e-4
    # macro-F1 = (0.6667+0.8+1.0)/3 = 0.8222
    assert abs(m["macro_f1"] - 0.8222) < 1e-3, m["macro_f1"]
    # confusion
    assert m["confusion"] == [[1, 1, 0], [0, 2, 0], [0, 0, 1]]


def test_abstain():
    labels = np.array([0, 1])
    logits = np.array([[5.0, -5.0], [0.1, 0.0]])  # 두번째는 confidence 낮음
    m = head_metrics(logits, labels, ["a", "b"], threshold=0.9)
    assert m["n_abstain"] == 1, m
    assert m["abstain_rate"] == 0.5


if __name__ == "__main__":
    test_hand_example()
    test_abstain()
    print("[OK] P6 지표 손계산 일치 · abstain 처리 정상")
