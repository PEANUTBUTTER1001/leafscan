"""Create figures inside one final-test evaluation history directory."""
import argparse
import json
from pathlib import Path

import numpy as np

from core import figures


def make_evaluation_figures(evaluation_dir):
    d = Path(evaluation_dir)
    result = json.loads((d / "test_metrics.json").read_text(encoding="utf-8"))
    out = d / "figures"
    out.mkdir(exist_ok=True)
    made = []
    for head, metric in (result.get("per_head") or {}).items():
        logits = np.load(d / f"logits_{head}.npy")
        labels = np.load(d / f"labels_{head}.npy")
        jobs = (
            (f"confusion_{head}.png", lambda: figures.confusion(metric, out / f"confusion_{head}.png", head)),
            (f"f1_per_class_{head}.png", lambda: figures.f1_per_class(metric, out / f"f1_per_class_{head}.png", head)),
            (f"calibration_{head}.png", lambda: figures.calibration(logits, labels, out / f"calibration_{head}.png", head)),
            (f"confidence_hist_{head}.png", lambda: figures.confidence_hist(logits, labels, out / f"confidence_hist_{head}.png", head)),
        )
        for name, fn in jobs:
            try:
                ok, _ = fn()
                if ok:
                    made.append(name)
            except Exception as exc:
                print(f"  · {name}: {exc}", flush=True)
    print(f"[테스트 그림] {len(made)}개 생성 → {out}", flush=True)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation", required=True)
    args = ap.parse_args()
    make_evaluation_figures(args.evaluation)


if __name__ == "__main__":
    main()
