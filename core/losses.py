"""core/losses.py — 손실 구성 (P3).

멀티헤드 손실 = Σ head_weight[h] · CE(logit[h], label[h], class_weight[h])

class weight (기획서 4.2 — 불균형은 생육단계 쪽 문제):
  · none   — 가중 없음
  · auto   — total / (n_class × count[c])
  · manual — cfg 에서 직접 지정
기본은 stage head 에만 적용한다 (cfg["class_weight_head"]).
"""
from collections import Counter

import torch
import torch.nn as nn


def auto_class_weights(labels, n_classes):
    """auto = total / (n_class × count[c]). 미등장 클래스는 1.0."""
    cnt = Counter(int(x) for x in labels)
    total = sum(cnt.values())
    w = []
    for c in range(n_classes):
        n = cnt.get(c, 0)
        w.append(total / (n_classes * n) if n > 0 else 1.0)
    return torch.tensor(w, dtype=torch.float32)


def build_criteria(heads: dict, cfg, train_labels: dict, device):
    """head 별 CrossEntropyLoss dict 를 만든다.

    heads: {name: n_classes}
    train_labels: {name: [int,...]}  (auto weight 계산용)
    """
    mode = cfg.get("class_weight", "none")
    target_head = cfg.get("class_weight_head", "stage")
    manual = cfg.get("class_weight_manual")  # {head: [w,...]} 또는 [w,...]
    crit = {}
    for name, n in heads.items():
        weight = None
        if mode != "none" and name == target_head:
            if mode == "auto":
                weight = auto_class_weights(train_labels.get(name, []), n).to(device)
            elif mode == "manual" and manual:
                ws = manual.get(name) if isinstance(manual, dict) else manual
                if ws is not None:
                    weight = torch.tensor(ws, dtype=torch.float32).to(device)
        crit[name] = nn.CrossEntropyLoss(weight=weight)
    return crit


def resolve_head_weights(heads: dict, cfg):
    """head 가중치 dict. 미지정 head 는 1.0. 단일 head 는 자동으로 1.0."""
    hw = dict(cfg.get("head_weights") or {})
    return {name: float(hw.get(name, 1.0)) for name in heads}
