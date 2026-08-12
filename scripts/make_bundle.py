"""make_bundle.py — 하류 소비자용 모델 번들 생성 (인계 H1/H2).

표준 번들 폴더를 만든다. LeafScan은 이 번들을 누가 어떻게 쓰는지 알지 못하며,
정해진 폴더 규격으로 내보내는 것까지가 이 프로젝트의 책임이다.

    python scripts/make_bundle.py --run runs/exp_007 --out bundles/v1.0.0 --version 1.0.0

산출:
  bundles/vX/
  ├─ model.ts.pt        # TorchScript (이식용). tflite/pte 변환은 별도(ai-edge-torch)
  ├─ labels.json        # 라벨 + 전처리 규격 (SSOT)
  ├─ validation/
  │  ├─ images/         # 검증 이미지 (최대 100장)
  │  └─ expected.json   # PyTorch 원본 출력 (소비 측 L1~L4 정합성 검증용)
  ├─ REPORT.md
  └─ BUNDLE.json

> **model.tflite/.pte 변환**은 팀 PC에 `ai-edge-torch`(또는 executorch)가 있을 때
> 별도 단계로 수행한다. 이 스크립트는 그 전 단계까지(TorchScript + 검증자료)를 만든다.
> validation/ 이 이 번들의 핵심이며, 이것으로 소비 측이 정합성(L1~L4)을 증명할 수 있다.
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.data import build_multihead_data
from core.metrics import softmax_np
from core.models import build_multihead_model


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def build_bundle(run_dir, out_dir, version, n_val=100):
    run = Path(run_dir)
    out = Path(out_dir)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    cfg = metrics.get("config", {})
    if cfg.get("dataset") != "index_csv":
        raise SystemExit("[중단] 실 데이터(index_csv) run 만 번들로 만들 수 있습니다.")
    heads = metrics.get("heads", {})
    label_names = metrics.get("label_names", {})
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation" / "images").mkdir(parents=True, exist_ok=True)

    # 모델 재구성 + best.pt
    model = build_multihead_model(cfg, heads)
    if (run / "best.pt").exists():
        model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    model.eval()

    # labels.json (라벨·전처리 규격 SSOT)
    labels = {
        "schema_version": 1,
        "model_version": version,
        "trained_run": run.name,
        "heads": {h: label_names.get(h, [f"{h}_{i}" for i in range(n)])
                  for h, n in heads.items()},
        "preprocess": {
            "input_size": int(cfg.get("img_size", 224)),
            "color_space": "RGB",
            "resize_mode": cfg.get("crop_mode", "full"),
            "interpolation": "bilinear",
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "value_range": "0-1",
        },
        "output": {"layout": "NCHW", "tensors": [
            {"index": i, "head": h, "shape": [1, n], "activation": "none"}
            for i, (h, n) in enumerate(heads.items())]},
        "baseline_metrics": _baseline(metrics),
    }
    (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    # validation/ — val 이미지 + PyTorch 출력
    _, val_ds, _test, _meta = build_multihead_data(cfg)
    samples = []
    order = list(heads.keys())
    for i in range(min(n_val, len(val_ds))):
        x, y = val_ds[i]
        with torch.no_grad():
            out_dict = model(x.unsqueeze(0))
        fname = f"images/val_{i:04d}.jpg"
        _save_img(x, cfg, out / "validation" / fname)
        entry = {"file": fname}
        for h in order:
            lg = out_dict[h][0].numpy()
            entry[f"{h}_logits"] = [round(float(v), 4) for v in lg]
            entry[f"{h}_pred"] = int(lg.argmax())
            entry[f"{h}_conf"] = round(float(softmax_np(lg[None])[0].max()), 4)
        samples.append(entry)
    expected = {"generated_from": run.name, "torch_version": torch.__version__,
                "samples": samples}
    (out / "validation" / "expected.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")

    # TorchScript 모델 (이식용)
    model_path = out / "model.ts.pt"
    try:
        example = torch.zeros(1, 3, int(cfg.get("img_size", 224)),
                              int(cfg.get("img_size", 224)))
        ts = torch.jit.trace(_WrapOrdered(model, order), example)
        ts.save(str(model_path))
    except Exception as e:  # noqa: BLE001
        (out / "MODEL_EXPORT_NOTE.txt").write_text(
            f"TorchScript 실패: {e}\nbest.pt 를 그대로 첨부합니다.", encoding="utf-8")
        shutil.copy(run / "best.pt", out / "best.pt")
        model_path = out / "best.pt"

    # REPORT.md
    if (run / "report.md").exists():
        shutil.copy(run / "report.md", out / "REPORT.md")

    # BUNDLE.json
    files = {model_path.name: _sha256(model_path),
             "labels.json": _sha256(out / "labels.json")}
    bundle = {
        "bundle_version": version,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source_run": run.name,
        "source_study": cfg.get("study"),
        "arch": cfg.get("arch"),
        "input_size": int(cfg.get("img_size", 224)),
        "quantization": "none",
        "converter": "torch.jit (tflite/pte 변환 미수행 — 별도 단계)",
        "files": files,
        "notes": "LeafScan make_bundle.py 자동 생성. tflite/pte 변환은 인계 전 별도 수행.",
    }
    (out / "BUNDLE.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"[번들] {out} 생성 · 검증표본 {len(samples)}장 · model={model_path.name}",
          flush=True)
    return out


class _WrapOrdered(torch.nn.Module):
    """dict 출력을 고정 순서 튜플로 — TorchScript/변환 호환."""

    def __init__(self, model, order):
        super().__init__()
        self.model = model
        self.order = order

    def forward(self, x):
        d = self.model(x)
        return tuple(d[h] for h in self.order)


def _baseline(metrics):
    ph = metrics.get("per_head", {})
    out = {}
    if "crop" in ph:
        out["crop_accuracy"] = ph["crop"].get("accuracy")
    if "stage" in ph:
        out["stage_macro_f1"] = ph["stage"].get("macro_f1")
        pc = ph["stage"].get("per_class", {})
        out["stage_recall"] = {k: v.get("recall") for k, v in pc.items()}
    return out


def _save_img(tensor, cfg, path):
    from PIL import Image
    from tasks.models_registry import get_spec
    (mean, std) = get_spec(cfg.get("arch", "resnet18")).norm
    t = tensor.clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    arr = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--n-val", type=int, default=100)
    args = ap.parse_args()
    build_bundle(args.run, args.out, args.version, n_val=args.n_val)


if __name__ == "__main__":
    main()
