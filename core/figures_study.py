"""core/figures_study.py — study 단위 비교 그림 4종 (P10b).

cost_accuracy · loss_overlay · runs_parallel · metric_progress.
member run 들의 metrics.json 만 읽는다 (의존 방향 불변).
"""
import numpy as np

from core.figures import MUTED, PALETTE, _ensure_style, _save


def _stage_f1(m):
    return (m.get("per_head", {}).get("stage", {}) or {}).get("macro_f1")


def _primary_recall(m):
    return (m.get("per_head", {}).get("stage", {}) or {}).get("primary_recall")


def cost_accuracy(members, out_path):
    """파라미터 수 vs stage macro-F1 산점도."""
    _ensure_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (name, m) in enumerate(members):
        x = m.get("params", 0) / 1e6
        y = _stage_f1(m)
        if y is None:
            continue
        ax.scatter(x, y, s=80, color=PALETTE[i % len(PALETTE)])
        ax.annotate(m.get("config", {}).get("arch", name), (x, y),
                    fontsize=8, xytext=(5, 3), textcoords="offset points")
    ax.set_xlabel("파라미터 수 (M)"); ax.set_ylabel("stage macro-F1")
    ax.set_title("비용 vs 정확도")
    return _save(fig, out_path)


def loss_overlay(members, out_path):
    """여러 run 의 val_loss 곡선 겹치기."""
    _ensure_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (name, m) in enumerate(members):
        hist = m.get("history") or []
        if not hist:
            continue
        ax.plot([h["epoch"] for h in hist], [h["val_loss"] for h in hist],
                color=PALETTE[i % len(PALETTE)],
                label=m.get("config", {}).get("arch", name))
    ax.set_xlabel("epoch"); ax.set_ylabel("val_loss")
    ax.set_title("val_loss 곡선 비교"); ax.legend(fontsize=8)
    return _save(fig, out_path)


def runs_parallel(members, out_path):
    """평행좌표 — 설정(lr, batch, img_size)과 지표(macro-F1, recall)의 관계."""
    _ensure_style()
    import matplotlib.pyplot as plt
    axes_keys = ["lr", "batch_size", "img_size", "macro_f1", "recall"]
    data = []
    for name, m in members:
        cfg = m.get("config", {})
        row = [cfg.get("lr", 0), cfg.get("batch_size", 0), cfg.get("img_size", 0),
               _stage_f1(m) or 0, _primary_recall(m) or 0]
        data.append((m.get("config", {}).get("arch", name), row))
    if not data:
        return False, "데이터 없음"
    arr = np.array([r for _, r in data], dtype=float)
    mins, maxs = arr.min(0), arr.max(0)
    rng = np.where(maxs - mins == 0, 1, maxs - mins)
    norm = (arr - mins) / rng
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = range(len(axes_keys))
    for i, (label, _) in enumerate(data):
        ax.plot(xs, norm[i], "o-", color=PALETTE[i % len(PALETTE)], label=label)
    ax.set_xticks(list(xs)); ax.set_xticklabels(axes_keys, rotation=20)
    ax.set_ylabel("정규화 값 (0~1)"); ax.set_title("평행좌표 — 설정↔지표")
    ax.legend(fontsize=8)
    for xi, k in enumerate(axes_keys):
        ax.text(xi, -0.06, f"{mins[xi]:.3g}", fontsize=6, color=MUTED, ha="center")
        ax.text(xi, 1.02, f"{maxs[xi]:.3g}", fontsize=6, color=MUTED, ha="center")
    return _save(fig, out_path)


def metric_progress(members, out_path):
    """실험 순서에 따른 지표 추이."""
    _ensure_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [m.get("config", {}).get("arch", name) for name, m in members]
    f1s = [_stage_f1(m) or 0 for _, m in members]
    rec = [_primary_recall(m) or 0 for _, m in members]
    x = range(len(members))
    ax.plot(x, f1s, "o-", color=PALETTE[0], label="stage macro-F1")
    ax.plot(x, rec, "s--", color=PALETTE[3], label="정식기 recall")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel("지표"); ax.set_ylim(0, 1.05)
    ax.set_title("실험 순서별 지표 추이"); ax.legend(fontsize=8)
    return _save(fig, out_path)


def make_study_figures(members, fig_dir):
    from pathlib import Path
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, fn in (("cost_accuracy", cost_accuracy),
                     ("loss_overlay", loss_overlay),
                     ("runs_parallel", runs_parallel),
                     ("metric_progress", metric_progress)):
        try:
            ok, info = fn(members, fig_dir / f"{name}.png")
        except Exception as e:  # noqa: BLE001
            ok, info = False, f"예외: {e}"
        results.append((name, ok, info))
    return results
