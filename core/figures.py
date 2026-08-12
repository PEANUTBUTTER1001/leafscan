"""core/figures.py — 그림 렌더 (matplotlib) · P10a/P10b.

전 run 에서 폰트·색·크기가 일관되도록 스타일 상수를 한 벌만 둔다 (축 C4).
지표 계산은 core/metrics.py 와 공유한다 → 화면·그림 숫자 일치.

각 그림 함수는 내부에서 예외를 삼키고 (성공 여부, 경로|사유) 를 반환한다.
→ 한 그림이 실패해도 나머지·학습 결과는 보존된다 (축 C7).
"""
import json
from pathlib import Path

import numpy as np

# --- 공통 스타일 -----------------------------------------------------------
FIGSIZE = (7.0, 4.5)
DPI = 130
PALETTE = ["#2f6f4f", "#4b7bec", "#b4761c", "#b4432f", "#6b5aa8", "#2c6fa8"]
GRID = "#e6e8ec"
INK = "#1f2328"
MUTED = "#6b7280"
_STYLE_READY = False


def _ensure_style():
    global _STYLE_READY
    if _STYLE_READY:
        return
    import matplotlib
    matplotlib.use("Agg")   # 화면 없이 파일로만
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # 한글 폰트 — Windows Malgun Gothic. 미설정 시 □ 로 깨진다.
    for cand in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if any(cand == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = DPI
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.edgecolor"] = MUTED
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = GRID
    _STYLE_READY = True


def _save(fig, path):
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return True, str(path)


# --------------------------------------------------------------------------
# 1) 손실 곡선
# --------------------------------------------------------------------------
def loss_curve(metrics, out_path, stage_events=None):
    _ensure_style()
    import matplotlib.pyplot as plt
    hist = metrics.get("history") or []
    if not hist:
        return False, "history 없음"
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(ep, [h["loss"] for h in hist], color=PALETTE[1], label="train")
    ax.plot(ep, [h["val_loss"] for h in hist], color=PALETTE[0], label="val",
            linestyle="--")
    be = metrics.get("best_epoch")
    if be:
        ax.axvline(be, color=PALETTE[3], alpha=0.6, linewidth=1)
        ax.text(be, ax.get_ylim()[1], f" best {be}", color=PALETTE[3],
                va="top", fontsize=8)
    for ev in (stage_events or []):
        ax.axvline(ev["epoch"], color=PALETTE[2], linestyle=":", alpha=0.7)
        ax.text(ev["epoch"], ax.get_ylim()[1], f" {ev.get('name','')}",
                color=PALETTE[2], va="top", fontsize=8)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("손실 곡선 (train/val)")
    ax.legend()
    return _save(fig, out_path)


# --------------------------------------------------------------------------
# 2) head 별 혼동행렬
# --------------------------------------------------------------------------
def confusion(head_metric, out_path, head_name=""):
    _ensure_style()
    import matplotlib.pyplot as plt
    cm = np.array(head_metric.get("confusion") or [])
    names = head_metric.get("class_names") or []
    if cm.size == 0:
        return False, "confusion 없음"
    fig, ax = plt.subplots(figsize=(max(4, len(names) * 1.1 + 2),
                                    max(3.5, len(names) * 1.1 + 1.5)))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
    ax.set_xlabel("예측"); ax.set_ylabel("실제")
    ax.set_title(f"혼동행렬 · {head_name}")
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else INK, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, out_path)


# --------------------------------------------------------------------------
# 3) 클래스별 F1
# --------------------------------------------------------------------------
def f1_per_class(head_metric, out_path, head_name="", baseline=None):
    _ensure_style()
    import matplotlib.pyplot as plt
    pc = head_metric.get("per_class") or {}
    if not pc:
        return False, "per_class 없음"
    names = list(pc)
    f1s = [pc[n]["f1"] for n in names]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(names))
    width = 0.38 if baseline else 0.6
    ax.bar(x - (width / 2 if baseline else 0), f1s, width, color=PALETTE[0],
           label="이번 run")
    if baseline:
        bf = [baseline.get(n, {}).get("f1", 0) for n in names]
        ax.bar(x + width / 2, bf, width, color=MUTED, label="기준 run")
        ax.legend()
    macro = head_metric.get("macro_f1")
    if macro is not None:
        ax.axhline(macro, color=PALETTE[3], linestyle="--", alpha=0.7)
        ax.text(len(names) - 1, macro, f" macro {macro:.3f}", color=PALETTE[3],
                fontsize=8, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1")
    ax.set_title(f"클래스별 F1 · {head_name}")
    return _save(fig, out_path)


# --------------------------------------------------------------------------
# 4) 캘리브레이션 (신뢰도 구간별 실제 정확도)
# --------------------------------------------------------------------------
def calibration(logits, labels, out_path, head_name="", n_bins=10):
    _ensure_style()
    import matplotlib.pyplot as plt
    from core.metrics import softmax_np
    if logits is None or len(logits) == 0:
        return False, "logits 없음"
    probs = softmax_np(logits)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = (conf >= bins[b]) & (conf < bins[b + 1] if b < n_bins - 1 else conf <= 1.0001)
        if m.sum() > 0:
            xs.append((bins[b] + bins[b + 1]) / 2)
            ys.append(correct[m].mean())
            ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle=":", label="완벽 보정")
    ax.plot(xs, ys, "o-", color=PALETTE[0], label="실제 정확도")
    for x, y, n in zip(xs, ys, ns):
        ax.text(x, y, str(n), fontsize=7, color=MUTED, va="bottom")
    ax.set_xlabel("예측 신뢰도"); ax.set_ylabel("실제 정확도")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title(f"캘리브레이션 · {head_name}")
    ax.legend()
    return _save(fig, out_path)


# --------------------------------------------------------------------------
# 6) 신뢰도 분포 (판정/보류 근거)
# --------------------------------------------------------------------------
def confidence_hist(logits, labels, out_path, head_name="", n_bins=20):
    _ensure_style()
    import matplotlib.pyplot as plt
    from core.metrics import softmax_np
    if logits is None or len(logits) == 0:
        return False, "logits 없음"
    probs = softmax_np(logits)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == labels
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.linspace(0, 1, n_bins + 1)
    ax.hist(conf[correct], bins=bins, color=PALETTE[0], alpha=0.75, label="정답")
    ax.hist(conf[~correct], bins=bins, color=PALETTE[3], alpha=0.75, label="오분류")
    ax.set_xlabel("예측 신뢰도"); ax.set_ylabel("표본 수")
    ax.set_title(f"신뢰도 분포 · {head_name}")
    ax.legend()
    return _save(fig, out_path)


# --------------------------------------------------------------------------
# 5) 데이터 분포 (분할 후 단계 비율)
# --------------------------------------------------------------------------
def data_distribution(metrics, out_path):
    _ensure_style()
    import matplotlib.pyplot as plt
    report = metrics.get("split_report")
    if not report:
        # 분할 정보가 없으면 stage head support 로 대체
        ph = (metrics.get("per_head") or {}).get("stage")
        if not ph:
            return False, "분포 정보 없음"
        pc = ph["per_class"]
        names = list(pc); vals = [pc[n]["support"] for n in names]
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.bar(names, vals, color=PALETTE[:len(names)])
        ax.set_title("단계 분포 (val)"); ax.set_ylabel("장수")
        return _save(fig, out_path)
    ratios = report["ratios"]
    parts = [p for p in ("train", "val", "test") if ratios.get(p, {}).get("_n")]
    stages = sorted({s for p in parts for s in ratios[p]
                     if not s.startswith("_")})
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(parts))
    bottom = np.zeros(len(parts))
    for si, st in enumerate(stages):
        vals = [ratios[p].get(st, 0) for p in parts]
        ax.bar(x, vals, bottom=bottom, label=st, color=PALETTE[si % len(PALETTE)])
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n(n={ratios[p]['_n']})" for p in parts])
    ax.set_ylabel("단계 비율"); ax.set_ylim(0, 1.02)
    ax.set_title("분할 후 단계 분포")
    ax.legend()
    return _save(fig, out_path)
