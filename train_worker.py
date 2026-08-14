"""
train_worker.py — LeafScan Lab 학습 워커 (Plan B / PyTorch)

GUI(lab_gui.py)가 subprocess로 실행하는 학습 스크립트.
GUI 없이 터미널에서 단독 실행해도 완전히 동일하게 동작한다.

    python train_worker.py --config runs/exp_001/config.json --out runs/exp_001

구조도 7.1 의 계약 C1~C5 를 지킨다.
  C1. 모든 설정을 --config 하나로 받는다 (하드코딩 금지)
  C2. 모든 출력을 --out 아래에만 쓴다
  C3. epoch마다 JSON 한 줄을 flush=True로 출력한다
  C4. 종료 시 metrics.json / logits*.npy / labels*.npy / status.json 을 남긴다
  C5. 그림·리포트를 생성하지 않는다

멀티헤드(P3): 학습 루프는 head dict 를 순회하므로 head 개수에 무관하게 동작한다.
  단일 head(fake/cifar)는 {"label": ...} 특수 케이스로 통일되며, 골든 수치가 보존된다.
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.config import (classify_failure, collect_env, config_diff,
                         load_config, load_parent_config)
from core.data import build_data, build_multihead_data
from core.losses import build_criteria, resolve_head_weights
from core.metrics import compute_metrics  # P6
from core.models import build_model, build_multihead_model
from core.schedule import build_optimizer, set_backbone_frozen, trainable_params
from core.tracking import Tracker

MULTIHEAD_DATASETS = ("index_csv", "multilabel_fake")


def emit(**payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def log(msg):
    emit(event="log", message=msg)


def write_status(out_dir, **fields):
    fields.setdefault("pid", os.getpid())
    (Path(out_dir) / "status.json").write_text(
        json.dumps(fields, ensure_ascii=False, indent=2), encoding="utf-8")


def _fp16_broken_gpu():
    """fp16 연산 결함(NaN)이 알려진 GPU 이면 이름을 반환, 아니면 None.

    GTX 16xx(Turing, 텐서코어 없음)는 특정 conv 형태에서 cuDNN fp16 커널이
    NaN 을 내는 결함이 널리 보고되어 있다 (실측: GTX 1650 + resnet18 32px 의
    layer3.0.downsample 1x1 conv 출력 24.5% NaN → 가중치 전체 오염).
    텐서코어가 없어 AMP 속도 이득도 거의 없으므로 끄는 것이 안전하다.
    """
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return None
    return name if re.search(r"GTX 16\d0", name) else None


def _vram_warn(cfg, device):
    arch = cfg.get("arch", "simple_cnn")
    try:
        from tasks.models_registry import get_spec, suggested_batch_for
        spec = get_spec(arch)
    except Exception:  # noqa: BLE001
        return
    vram = 4
    if device == "cuda":
        try:
            vram = int(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
        except Exception:  # noqa: BLE001
            pass
    rec = suggested_batch_for(spec, vram)
    bs = int(cfg["batch_size"])
    if bs > rec:
        emit(event="warn", message=(
            f"batch_size {bs} 가 {arch} 의 {vram}GB VRAM 권장 배치({rec})를 초과합니다. "
            f"OOM 시 배치를 {rec} 이하로 낮추세요."))


# --------------------------------------------------------------------------
# head dict 정규화 — 단일 head 텐서/int 라벨을 {"label": ...} 로 통일
# --------------------------------------------------------------------------
def _as_out_dict(out):
    return out if isinstance(out, dict) else {"label": out}


def _labels_to_device(labels, device):
    if isinstance(labels, dict):
        return {k: v.to(device) for k, v in labels.items()}
    return {"label": labels.to(device)}


# --------------------------------------------------------------------------
# 학습 / 검증 루프 (head 개수 무관)
# --------------------------------------------------------------------------
def run_epoch(model, loader, criteria, head_weights, device,
              optimizer=None, scaler=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    amp_on = scaler is not None and scaler.is_enabled()
    total_loss, total = 0.0, 0
    correct = Counter()
    ctx = torch.enable_grad() if train_mode else torch.no_grad()

    def compute(images, labels):
        outs = _as_out_dict(model(images))
        loss = 0.0
        for h, logit in outs.items():
            loss = loss + head_weights[h] * criteria[h](logit, labels[h])
        return outs, loss

    with ctx:
        for images, labels in loader:
            images = images.to(device)
            labels = _labels_to_device(labels, device)
            if train_mode and amp_on:
                optimizer.zero_grad()
                with torch.autocast(device_type="cuda"):
                    outs, loss = compute(images, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outs, loss = compute(images, labels)
                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            bs = images.size(0)
            batch_loss = loss.item()
            if not math.isfinite(batch_loss):
                raise RuntimeError(
                    f"손실이 NaN/Inf 입니다 (batch loss={batch_loss}) — 즉시 중단. "
                    f"lr 과대, AMP fp16 결함, 데이터 이상 여부를 확인하세요")
            total_loss += batch_loss * bs
            total += bs
            for h, logit in outs.items():
                correct[h] += (logit.argmax(1) == labels[h]).sum().item()

    n_heads = max(1, len(correct))
    mean_acc = sum(correct[h] / max(total, 1) for h in correct) / n_heads
    per_head_acc = {h: correct[h] / max(total, 1) for h in correct}
    return total_loss / max(total, 1), mean_acc, per_head_acc


@torch.no_grad()
def collect_predictions(model, loader, device):
    """head 별 logits/labels numpy dict 반환."""
    model.eval()
    logits = {}
    labels_acc = {}
    for images, labels in loader:
        outs = _as_out_dict(model(images.to(device)))
        labels = labels if isinstance(labels, dict) else {"label": labels}
        for h, logit in outs.items():
            logits.setdefault(h, []).append(logit.cpu())
        for h, lab in labels.items():
            labels_acc.setdefault(h, []).append(
                lab if torch.is_tensor(lab) else torch.tensor(lab))
    logits = {h: torch.cat(v).numpy() for h, v in logits.items()}
    labels_acc = {h: torch.cat(v).numpy() for h, v in labels_acc.items()}
    return logits, labels_acc


# --------------------------------------------------------------------------
# 데이터·모델 구성
# --------------------------------------------------------------------------
def setup_data_and_model(cfg, device):
    """(train_ds, val_ds, model, heads, label_names, extra) 반환.

    단일 head(fake/cifar)와 멀티헤드(index_csv/multilabel_fake)를 통일한다.
    """
    if cfg["dataset"] in MULTIHEAD_DATASETS:
        train_ds, val_ds, test_ds, meta = build_multihead_data(cfg, log=log)
        heads = meta["heads"]
        label_names = {}
        if "crop" in heads:
            label_names["crop"] = meta["crop_names"]
        if "stage" in heads:
            label_names["stage"] = meta["stage_names"]
        # config 검증: head_weights 키가 heads 안에 있는지
        for k in (cfg.get("head_weights") or {}):
            if k not in heads:
                raise ValueError(f"head_weights 의 head {k!r} 가 데이터 heads {list(heads)} 에 없음")
        cwh = cfg.get("class_weight_head", "stage")
        if cfg.get("class_weight", "none") != "none" and cwh not in heads:
            raise ValueError(f"class_weight_head {cwh!r} 가 heads {list(heads)} 에 없음")
        model = build_multihead_model(cfg, heads)
        extra = {"split": meta.get("split"), "split_report": meta.get("split_report"),
                 "test_ds": test_ds}
        return train_ds, val_ds, model, heads, label_names, extra

    # 단일 head 경로 (골든 보존)
    train_ds, val_ds, classes = build_data(cfg, log=log)
    heads = {"label": len(classes)}
    label_names = {"label": classes}
    model = build_model(cfg, num_classes=len(classes))   # 기존 경로 (동작 불변)
    return train_ds, val_ds, model, heads, label_names, {"test_ds": val_ds}


def _collect_train_labels(loader, heads):
    """auto class weight 계산용 head 별 라벨 수집."""
    acc = {h: [] for h in heads}
    for _images, labels in loader:
        labels = labels if isinstance(labels, dict) else {"label": labels}
        for h in heads:
            v = labels[h]
            acc[h].extend(v.tolist() if torch.is_tensor(v) else list(v))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)   # v1 → v2 마이그레이션 (원본 파일 미수정)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    write_status(out, state="running", epoch=0)
    env = collect_env(cfg, device)
    (out / "env.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    runs_dir = out.parent
    parent_cfg = load_parent_config(cfg, runs_dir)
    if parent_cfg is not None:
        (out / "diff.json").write_text(
            json.dumps(config_diff(cfg, parent_cfg), ensure_ascii=False, indent=2),
            encoding="utf-8")

    train_ds, val_ds, model, heads, label_names, extra = setup_data_and_model(cfg, device)
    model = model.to(device)

    emit(event="start", total_epochs=int(cfg["epochs"]), device=device,
         arch=cfg.get("arch", "simple_cnn"), heads=heads,
         torch_version=torch.__version__, torchvision_version=torchvision.__version__)

    nw = cfg.get("num_workers", 0)
    if nw in (None, "auto"):
        # JPEG 디코딩이 병목인 실이미지 데이터셋에서만 병렬 로딩 기본 적용.
        # Windows 의 worker spawn(프로세스 생성+torch import)이 수십 초라,
        # 작은 데이터(빠른확인 등)에서는 병렬화가 오히려 크게 느리다 (실측:
        # 500장 12초 → 150초). 충분히 커서 상쇄될 때만 켠다.
        big_enough = cfg["dataset"] == "index_csv" and len(train_ds) >= 3000
        nw = min(6, max(2, (os.cpu_count() or 4) - 2)) if big_enough else 0
    nw = int(nw)
    dl_kw = dict(num_workers=nw, pin_memory=(device == "cuda"),
                 persistent_workers=(nw > 0 and int(cfg["epochs"]) > 1))
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]),
                              shuffle=True, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]),
                            shuffle=False, **dl_kw)
    if nw:
        log(f"DataLoader 병렬 로딩 num_workers={nw}")
    log(f"train {len(train_ds)}장 / val {len(val_ds)}장 · device={device} · heads={heads}")

    n_params = sum(p.numel() for p in model.parameters())
    flat = getattr(model, "flat_features", None) or getattr(model, "feature_dim", None)
    if flat is not None:
        log(f"feature 차원 = {flat} · 총 파라미터 {n_params:,}개")
    else:
        log(f"총 파라미터 {n_params:,}개")

    _vram_warn(cfg, device)

    # 손실: head 별 criterion + head 가중치
    train_labels = ({} if cfg.get("class_weight", "none") == "none"
                    else _collect_train_labels(train_loader, heads))
    criteria = build_criteria(heads, cfg, train_labels, device)
    head_weights = resolve_head_weights(heads, cfg)
    if len(heads) > 1:
        log(f"head 가중치 {head_weights} · class_weight={cfg.get('class_weight','none')}")

    lr = float(cfg["lr"])
    # 전이학습 freeze 스케줄 (P5) — freeze_epochs>0 이면 백본 동결로 시작
    freeze_epochs = int(cfg.get("freeze_epochs", 0))
    lr_finetune = float(cfg.get("lr_finetune", lr))
    stage_events = []
    if freeze_epochs > 0:
        if set_backbone_frozen(model, True):
            log(f"백본 동결 · freeze {freeze_epochs}epoch · "
                f"학습가능 파라미터 {trainable_params(model):,}개")
        else:
            log("백본이 없어 freeze 를 건너뜁니다 (simple_cnn 단일 head).")
            freeze_epochs = 0
    optimizer = build_optimizer(cfg, model, lr)

    amp_enabled = bool(cfg.get("amp", True)) and device == "cuda"
    broken_gpu = _fp16_broken_gpu() if amp_enabled else None
    if broken_gpu:
        amp_enabled = False
        log(f"AMP 자동 비활성 — {broken_gpu} 는 fp16 결함(NaN)이 알려진 "
            f"GPU 입니다 (GTX 16xx). fp32 로 학습합니다.")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if amp_enabled:
        log("AMP(autocast + GradScaler) 활성")

    # 추적 (P8) — MLflow/wandb 미설치 시 조용히 no-op. run 이름 = 디렉터리명.
    tracker = Tracker(out.name, cfg, enable=bool(cfg.get("tracking", True)))
    if tracker.active:
        log("실험 추적 활성 (MLflow/wandb)")

    patience = int(cfg.get("patience", 0))
    best_val, best_epoch, bad = float("inf"), 0, 0
    history = []
    t0 = time.time()

    for e in range(1, int(cfg["epochs"]) + 1):
        # freeze 해제 시점 — 백본 해제 + optimizer 재생성(필수) + lr 전환
        if freeze_epochs and e == freeze_epochs + 1:
            set_backbone_frozen(model, False)
            optimizer = build_optimizer(cfg, model, lr_finetune)
            n_train = trainable_params(model)
            log(f"[freeze 해제] epoch {e} · 학습가능 파라미터 {n_train:,}개 · "
                f"lr {lr} → {lr_finetune}")
            emit(event="stage", name="freeze_release", epoch=e,
                 trainable_params=n_train, lr=lr_finetune)
            stage_events.append({"name": "freeze_release", "epoch": e,
                                 "trainable_params": n_train})
        tr_loss, tr_acc, tr_ph = run_epoch(model, train_loader, criteria,
                                           head_weights, device, optimizer, scaler)
        va_loss, va_acc, va_ph = run_epoch(model, val_loader, criteria,
                                           head_weights, device)
        rec = dict(epoch=e, loss=tr_loss, acc=tr_acc,
                   val_loss=va_loss, val_acc=va_acc)
        if len(heads) > 1:   # 골든(단일 head)에는 추가 키를 넣지 않는다
            rec["per_head_acc"] = va_ph
        history.append(rec)
        improved = va_loss < best_val - 1e-6
        if improved:
            best_val, best_epoch, bad = va_loss, e, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1

        emit(event="epoch", epoch=e, total_epochs=int(cfg["epochs"]),
             loss=tr_loss, acc=tr_acc, val_loss=va_loss, val_acc=va_acc,
             per_head_acc=va_ph, best_epoch=best_epoch, improved=improved,
             elapsed=round(time.time() - t0, 1))
        tracker.log_metrics({"loss": tr_loss, "val_loss": va_loss,
                             "acc": tr_acc, "val_acc": va_acc}, step=e)
        write_status(out, state="running", epoch=e)

        if patience and bad >= patience:
            log(f"Early stopping — val_loss가 {patience}epoch 연속 개선되지 않음 "
                f"(최적 epoch {best_epoch})")
            break

    if (out / "best.pt").exists():
        model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    logits, labels = collect_predictions(model, val_loader, device)

    # head 별 저장 + 단일 head 하위호환(logits.npy/labels.npy)
    for h in logits:
        np.save(out / f"logits_{h}.npy", logits[h])
        np.save(out / f"labels_{h}.npy", labels[h])
    if len(heads) == 1:
        only = next(iter(logits))
        np.save(out / "logits.npy", logits[only])
        np.save(out / "labels.npy", labels[only])

    # P6 지표
    per_head_metrics = compute_metrics(logits, labels, label_names)

    final = history[-1] if history else {}
    single_classes = label_names.get("label") or label_names.get(
        "stage") or next(iter(label_names.values()), [])
    metrics = dict(
        best_epoch=best_epoch,
        best_val_loss=best_val if history else None,
        best_val_acc=max((h["val_acc"] for h in history), default=None),
        final_train_acc=final.get("acc"),
        final_val_acc=final.get("val_acc"),
        epochs_ran=len(history),
        elapsed_sec=round(time.time() - t0, 1),
        params=n_params,
        classes=single_classes,
        heads=heads,
        label_names=label_names,
        per_head=per_head_metrics,
        stage_events=stage_events,
        history=history,
        config=cfg,
    )
    if extra.get("split") is not None:
        metrics["split"] = extra["split"]
        metrics["split_report"] = extra["split_report"]
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    tracker.end()
    write_status(out, state="done", epoch=len(history), failure_kind=None)
    emit(event="done", run_id=out.name,
         metrics={k: v for k, v in metrics.items()
                  if k not in ("history", "config", "per_head", "label_names",
                               "split", "split_report")})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:                                   # noqa: BLE001
        kind = classify_failure(exc)
        emit(event="error", message=str(exc), traceback=traceback.format_exc(),
             kind=kind)
        try:
            o = sys.argv[sys.argv.index("--out") + 1]
            write_status(o, state="failed", error=str(exc), failure_kind=kind)
        except Exception:                                      # noqa: BLE001
            pass
        sys.exit(1)
