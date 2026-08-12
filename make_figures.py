"""make_figures.py — run 그림 생성 (별도 프로세스) · P10a/P10b.

**학습과 분리된 프로세스.** run 디렉터리의 파일만 읽고 figures/ 에 쓴다.
그림 생성이 실패해도 학습 결과(metrics·logits·best.pt)는 온전히 보존된다 (축 C7).

    python make_figures.py --run runs/exp_007
    python make_figures.py --run runs/exp_007 --advanced   # P10b 심화 포함

의존 방향: 위로 올라가지 않는다. train_worker 는 이 스크립트를 모른다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import figures


def _load_logits(run: Path, head: str):
    p = run / f"logits_{head}.npy"
    lp = run / f"labels_{head}.npy"
    if p.exists() and lp.exists():
        return np.load(p), np.load(lp)
    # 단일 head 하위호환
    if (run / "logits.npy").exists() and (run / "labels.npy").exists():
        return np.load(run / "logits.npy"), np.load(run / "labels.npy")
    return None, None


def make_run_figures(run_dir, advanced=False):
    run = Path(run_dir)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    fig_dir = run / "figures"
    fig_dir.mkdir(exist_ok=True)
    results = []

    def do(name, fn):
        try:
            ok, info = fn()
        except Exception as e:  # noqa: BLE001 — 그림 하나 실패가 전체를 막지 않는다
            ok, info = False, f"예외: {e}"
        results.append((name, ok, info))
        print(f"  {'✓' if ok else '·'} {name}: {info}", flush=True)

    per_head = metrics.get("per_head") or {}
    stage_events = metrics.get("stage_events")

    do("loss_curve", lambda: figures.loss_curve(
        metrics, fig_dir / "loss_curve.png", stage_events))
    do("data_distribution", lambda: figures.data_distribution(
        metrics, fig_dir / "data_distribution.png"))
    for head, hm in per_head.items():
        do(f"confusion_{head}", lambda hm=hm, head=head: figures.confusion(
            hm, fig_dir / f"confusion_{head}.png", head))
        do(f"f1_per_class_{head}", lambda hm=hm, head=head: figures.f1_per_class(
            hm, fig_dir / f"f1_per_class_{head}.png", head))
    # 캘리브레이션 — 주 head(stage 우선)
    main_head = "stage" if "stage" in per_head else (
        next(iter(per_head)) if per_head else None)
    if main_head:
        lg, lb = _load_logits(run, main_head)
        do(f"calibration_{main_head}", lambda: figures.calibration(
            lg, lb, fig_dir / "calibration.png", main_head))
        do(f"confidence_hist_{main_head}", lambda: figures.confidence_hist(
            lg, lb, fig_dir / "confidence_hist.png", main_head))

    if advanced:
        _make_advanced(run, metrics, per_head, fig_dir, do)

    ok_n = sum(1 for _, ok, _ in results if ok)
    print(f"[그림] {ok_n}/{len(results)} 생성 → {fig_dir}", flush=True)
    return results


def _make_advanced(run, metrics, per_head, fig_dir, do):
    """P10b 심화 — 오분류 그리드·Grad-CAM·t-SNE. 데이터/모델이 있을 때만."""
    from core import figures_advanced
    do("misclassified_grid", lambda: figures_advanced.misclassified_grid(
        run, metrics, fig_dir / "misclassified_grid.png"))
    do("gradcam_samples", lambda: figures_advanced.gradcam_samples(
        run, metrics, fig_dir / "gradcam_samples.png"))
    do("embedding_tsne", lambda: figures_advanced.embedding_tsne(
        run, metrics, fig_dir / "embedding_tsne.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--advanced", action="store_true", help="P10b 심화 그림 포함")
    args = ap.parse_args()
    run = Path(args.run)
    if not (run / "metrics.json").exists():
        raise SystemExit(f"[중단] metrics.json 없음: {run}")
    make_run_figures(run, advanced=args.advanced)


if __name__ == "__main__":
    main()
