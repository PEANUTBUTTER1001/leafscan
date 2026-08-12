"""core/metrics.py — 지표 계산 (P6).

head 별로 sklearn classification_report 기반 지표를 계산한다.
  · accuracy, macro-F1 (주 지표), 클래스별 P/R/F1
  · confusion matrix (미판정 열은 threshold 적용 시 추가)
  · 정식기 recall — 소수 클래스 검출력 (stage head 최상위 지표)

같은 계산 결과를 그림 생성기(core/figures.py)와 공유한다 → 화면·그림 숫자 일치.
"""
import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report, confusion_matrix

from tasks.leafscan import PRIMARY_STAGE


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def head_metrics(logits, labels, class_names, threshold=None):
    """단일 head 지표. threshold 지정 시 미판정(abstain) 처리."""
    n_classes = len(class_names)
    probs = softmax_np(logits)
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    abstain_mask = np.zeros(len(preds), dtype=bool)
    if threshold is not None:
        abstain_mask = conf < threshold

    decided = ~abstain_mask
    y_true = labels[decided]
    y_pred = preds[decided]
    label_ids = list(range(n_classes))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        warnings.simplefilter("ignore", UserWarning)
        rep = classification_report(
            y_true, y_pred, labels=label_ids, target_names=class_names,
            output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=label_ids).tolist()
    acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0

    per_class = {}
    for name in class_names:
        r = rep.get(name, {})
        per_class[name] = {
            "precision": round(r.get("precision", 0.0), 4),
            "recall": round(r.get("recall", 0.0), 4),
            "f1": round(r.get("f1-score", 0.0), 4),
            "support": int(r.get("support", 0)),
        }

    out = {
        "accuracy": round(acc, 4),
        "macro_f1": round(rep.get("macro avg", {}).get("f1-score", 0.0), 4),
        "weighted_f1": round(rep.get("weighted avg", {}).get("f1-score", 0.0), 4),
        "per_class": per_class,
        "confusion": cm,
        "class_names": list(class_names),
        "n_total": int(len(preds)),
        "n_abstain": int(abstain_mask.sum()),
        "abstain_rate": round(float(abstain_mask.mean()), 4) if len(preds) else 0.0,
    }
    # 정식기 recall 을 최상위로 승격 (stage head 일 때만 존재)
    if PRIMARY_STAGE in per_class:
        out["primary_recall"] = per_class[PRIMARY_STAGE]["recall"]
        out["primary_class"] = PRIMARY_STAGE
    return out


def compute_metrics(logits: dict, labels: dict, label_names: dict, threshold=None):
    """head 별 지표 dict. label_names 없는 head 는 인덱스 이름으로 대체."""
    result = {}
    for head, lg in logits.items():
        names = label_names.get(head)
        if not names:
            names = [f"{head}_{i}" for i in range(lg.shape[1])]
        result[head] = head_metrics(lg, labels[head], names, threshold=threshold)
    return result
