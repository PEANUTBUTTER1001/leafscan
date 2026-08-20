"""Final, split-only evaluation for completed ``index_csv`` runs.

This module deliberately does not call the training data factory: doing so could
create a split or apply a training subset.  The evaluator reads the recorded
split and constructs only the test dataset.
"""
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.config import file_hash, load_config
from core.data import (IndexCsvDataset, build_leaf_transforms, load_index_rows)
from core.metrics import compute_metrics
from core.models import build_multihead_model
from core.split import assign_rows, load_split, verify_split
from tasks.leafscan import CANON_CROPS, CANON_STAGES, order_present

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(value, base):
    p = Path(value)
    return p if p.is_absolute() else (Path(base) / p)


def _sha256_bytes(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _json_hash(obj):
    return _sha256_bytes(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode())


def evaluation_dirs(run):
    return Path(run) / "test_evaluations"


def list_evaluations(run):
    root = evaluation_dirs(run)
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir()
                   if p.is_dir() and (p / "test_metrics.json").exists()),
                  key=lambda p: p.name)


def latest_evaluation(run):
    items = list_evaluations(run)
    return items[-1] if items else None


def final_test_command(python, worker, run, reevaluate=False, reason=None):
    """Build the GUI subprocess command without starting a process."""
    cmd = [str(python), str(worker), "--run", str(run)]
    if reevaluate:
        cmd += ["--reevaluate", "--reason", str(reason).strip()]
    return cmd


def validate_run(run):
    """Validate the immutable inputs and return a prepared context."""
    run = Path(run)
    required = {name: run / name for name in ("config.json", "best.pt")}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"필수 파일 없음: {', '.join(missing)}")
    status_path = run / "status.json"
    if not status_path.is_file() or json.loads(status_path.read_text(encoding="utf-8")).get("state") != "done":
        raise RuntimeError("완료(state=done)된 run만 최종 테스트할 수 있습니다.")
    cfg = load_config(required["config.json"])
    if cfg.get("dataset") != "index_csv":
        raise RuntimeError("최종 테스트 대상 dataset은 index_csv뿐입니다.")
    index_value = cfg.get("index_csv") or cfg.get("data_index")
    if not index_value:
        raise RuntimeError("index_csv 설정이 없습니다.")
    index_path = _resolve(index_value, PROJECT_ROOT)
    split_value = cfg.get("split_json")
    candidates = []
    if split_value:
        candidates.extend([_resolve(split_value, PROJECT_ROOT),
                           _resolve(split_value, run),
                           _resolve(split_value, run.parent)])
    candidates.extend([run / "split.json", run.parent / "split.json"])
    split_path = next((p for p in candidates if p.is_file()), None)
    if split_path is None:
        raise RuntimeError("고정 split.json이 없습니다.")
    if not index_path.is_file():
        raise RuntimeError(f"index.csv 없음: {index_path}")
    rows = load_index_rows(index_path)
    split = load_split(split_path)
    if not all(k in split for k in ("train", "val", "test")):
        raise RuntimeError("split.json에 train/val/test가 모두 필요합니다.")
    ok, report = verify_split(rows, split)
    if not ok:
        raise RuntimeError(f"분할 그룹 중복 발견: {report['overlaps']}")
    indices = assign_rows(rows, split)["test"]
    if not indices:
        raise RuntimeError("test 이미지가 1장 이상이어야 합니다.")
    # Read every path before starting output so a broken index cannot leave history.
    broken = [rows[i]["path"] for i in indices
              if not (_resolve(rows[i]["path"], index_path.parent)).is_file()]
    if broken:
        raise RuntimeError(f"test 이미지 없음: {broken[0]}")
    return {"run": run, "cfg": cfg, "index_path": index_path,
            "split_path": split_path, "rows": rows, "split": split,
            "report": report, "indices": indices}


def _label_names(rows):
    return {"crop": order_present({r["crop"] for r in rows}, CANON_CROPS),
            "stage": order_present({r["stage"] for r in rows}, CANON_STAGES)}


def _build_test_loader(ctx):
    cfg = ctx["cfg"]
    names = _label_names(ctx["rows"])
    from tasks.models_registry import get_spec
    spec = get_spec(cfg.get("arch", "resnet18"))
    img_size = int(cfg.get("img_size") or spec.default_size)
    _, eval_tf = build_leaf_transforms(img_size, spec.norm, augment=False)
    ds = IndexCsvDataset(ctx["rows"], ctx["indices"], names["crop"], names["stage"],
                         eval_tf, cfg.get("crop_mode", "full"), ctx["index_path"].parent,
                         cache_dir=None, cache_size=int(cfg.get("cache_size", 256)))
    return ds, names


@torch.no_grad()
def _predict(model, loader, device):
    logits, labels = {}, {}
    model.eval()
    for images, batch_labels in loader:
        out = model(images.to(device))
        out = out if isinstance(out, dict) else {"label": out}
        batch_labels = batch_labels if isinstance(batch_labels, dict) else {"label": batch_labels}
        for head, value in out.items():
            logits.setdefault(head, []).append(value.cpu())
        for head, value in batch_labels.items():
            labels.setdefault(head, []).append(value.cpu())
    return ({h: torch.cat(v).numpy() for h, v in logits.items()},
            {h: torch.cat(v).numpy() for h, v in labels.items()})


def evaluate_run(run, allow_rerun=False, reason=None, device=None, batch_size=None):
    ctx = validate_run(run)
    existing = list_evaluations(ctx["run"])
    if existing and not allow_rerun:
        raise RuntimeError(f"이미 최종 테스트 평가가 있습니다: {existing[-1].name} (재평가 사유 필요)")
    if allow_rerun and not str(reason or "").strip():
        raise RuntimeError("재평가에는 사유가 필요합니다.")
    run = ctx["run"]
    cfg = ctx["cfg"]
    ds, label_names = _build_test_loader(ctx)
    heads = {"crop": len(label_names["crop"]), "stage": len(label_names["stage"])}
    model_cfg = dict(cfg)
    model_cfg["pretrained"] = False  # architecture only; checkpoint is the source of weights
    model = build_multihead_model(model_cfg, heads)
    dev = device or cfg.get("device", "auto")
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    checkpoint_hash_before = file_hash(run / "best.pt")
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev))
    loader = DataLoader(ds, batch_size=int(batch_size or cfg.get("batch_size", 32)),
                        shuffle=False, num_workers=0)
    logits, labels = _predict(model, loader, dev)
    metrics = compute_metrics(logits, labels, label_names)
    checkpoint_hash_after = file_hash(run / "best.pt")
    if checkpoint_hash_before != checkpoint_hash_after:
        raise RuntimeError("평가 중 best.pt 해시가 변경되었습니다.")
    evaluation_id = datetime.now(timezone.utc).strftime("eval_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    final_dir = evaluation_dirs(run) / evaluation_id
    tmp_dir = evaluation_dirs(run) / (f".tmp_{evaluation_id}_{os.getpid()}")
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        for head in logits:
            np.save(tmp_dir / f"logits_{head}.npy", logits[head])
            np.save(tmp_dir / f"labels_{head}.npy", labels[head])
        meta = {
            "evaluation_id": evaluation_id, "run_id": run.name,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_hash": checkpoint_hash_after,
            "config_hash": file_hash(run / "config.json"),
            "split_hash": file_hash(ctx["split_path"]),
            "index_hash": file_hash(ctx["index_path"]),
            "test_images": len(ctx["indices"]),
            "test_groups": ctx["report"]["ratios"]["test"]["_groups"],
            "group_overlap": ctx["report"]["overlaps"],
            "reason": str(reason).strip() if reason else None,
        }
        result = {"evaluation_id": evaluation_id, "metadata": meta,
                  "per_head": metrics, "label_names": label_names}
        (tmp_dir / "test_metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (tmp_dir / "evaluation_metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        evaluation_dirs(run).mkdir(exist_ok=True)
        os.replace(tmp_dir, final_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return final_dir
