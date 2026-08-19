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


class ProgressEmitter:
    """진행 이벤트 스로틀러 — 기본 1초에 한 번. 처음·마지막은 항상 낸다.

    학습 로직은 건드리지 않고 '지금 어디까지 왔는지'만 알린다.
    긴 무출력 구간(라벨 수집·epoch 내부·예측 수집)에서 GUI 가 멈춘 것처럼
    보이던 문제를 없애기 위한 것이다.

    clock·sink 를 주입할 수 있어 테스트에서 결정론적으로 검증한다.
    """

    def __init__(self, interval=1.0, clock=None, sink=None):
        self.interval = interval
        self.clock = clock or time.monotonic
        self.sink = sink or emit
        self._last = None

    def phase(self, name):
        """총량을 모르는 단계의 시작. 스로틀을 초기화한다."""
        self._last = None
        self.sink(event="phase", name=name)

    def step(self, name, done, total=0, **extra):
        """진행 1틱. 실제로 보냈으면 True."""
        now = self.clock()
        if self._last is not None:
            if not (total and done >= total):
                if (now - self._last) < self.interval:
                    return False
        self._last = now
        self.sink(event="progress", phase=name, done=done, total=total, **extra)
        return True


def write_status(out_dir, **fields):
    fields.setdefault("pid", os.getpid())
    (Path(out_dir) / "status.json").write_text(
        json.dumps(fields, ensure_ascii=False, indent=2), encoding="utf-8")


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
              optimizer=None, scaler=None,
              prog=None, phase="", frac_base=0.0, frac_span=0.0):
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

    n_steps = len(loader)
    with ctx:
        for step, (images, labels) in enumerate(loader, 1):
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
            total_loss += loss.item() * bs
            total += bs
            for h, logit in outs.items():
                correct[h] += (logit.argmax(1) == labels[h]).sum().item()
            if prog is not None and n_steps:
                extra = ({"epoch_frac": frac_base + frac_span * step / n_steps}
                         if frac_span else {})
                prog.step(phase, step, n_steps, **extra)

    n_heads = max(1, len(correct))
    mean_acc = sum(correct[h] / max(total, 1) for h in correct) / n_heads
    per_head_acc = {h: correct[h] / max(total, 1) for h in correct}
    return total_loss / max(total, 1), mean_acc, per_head_acc


@torch.no_grad()
def collect_predictions(model, loader, device, prog=None):
    """head 별 logits/labels numpy dict 반환."""
    model.eval()
    logits = {}
    labels_acc = {}
    n_steps = len(loader)
    for step, (images, labels) in enumerate(loader, 1):
        if prog is not None and n_steps:
            prog.step("검증 예측 수집", step, n_steps)
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


def _collect_train_labels(loader, heads, prog=None):
    """auto class weight 계산용 head 별 라벨 수집.

    라벨만 쓰지만 로더가 이미지까지 전부 디코딩·변환하므로 시간이 오래 걸린다.
    학습 시작 전인데 아무 출력이 없어 멈춘 것처럼 보이던 구간이라 진행률을 낸다.
    """
    acc = {h: [] for h in heads}
    n_steps = len(loader)
    for step, (_images, labels) in enumerate(loader, 1):
        labels = labels if isinstance(labels, dict) else {"label": labels}
        for h in heads:
            v = labels[h]
            acc[h].extend(v.tolist() if torch.is_tensor(v) else list(v))
        if prog is not None and n_steps:
            prog.step("학습 라벨 수집 (class weight)", step, n_steps)
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

    nw = int(cfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]),
                              shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]),
                            shuffle=False, num_workers=nw)
    log(f"train {len(train_ds)}장 / val {len(val_ds)}장 · device={device} · heads={heads}")

    n_params = sum(p.numel() for p in model.parameters())
    flat = getattr(model, "flat_features", None) or getattr(model, "feature_dim", None)
    if flat is not None:
        log(f"feature 차원 = {flat} · 총 파라미터 {n_params:,}개")
    else:
        log(f"총 파라미터 {n_params:,}개")

    _vram_warn(cfg, device)

    prog = ProgressEmitter(interval=float(cfg.get("progress_interval", 1.0)))

    # 손실: head 별 criterion + head 가중치
    if cfg.get("class_weight", "none") == "none":
        train_labels = {}
    else:
        prog.phase("학습 라벨 수집 (class weight)")
        train_labels = _collect_train_labels(train_loader, heads, prog)
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
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if amp_enabled:
        log("AMP(autocast + GradScaler) 활성")

    # 추적 (P8) — MLflow/wandb 미설치 시 조용히 no-op. run 이름 = 디렉터리명.
    tracker = Tracker(out.name, cfg, enable=bool(cfg.get("tracking", True)))
    for m in tracker.messages:
        log(m)
    if tracker.run_url:
        # GUI 의 'wandb 보기' 버튼용 — 라이브 이벤트 + run 폴더에 영구 기록
        emit(event="wandb", url=tracker.run_url)
        (out / "wandb.json").write_text(
            json.dumps({"run_url": tracker.run_url}, ensure_ascii=False, indent=2),
            encoding="utf-8")

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
        # 진행바는 epoch 하나를 학습 절반 · 검증 절반으로 나눠 채운다
        tr_loss, tr_acc, tr_ph = run_epoch(
            model, train_loader, criteria, head_weights, device, optimizer, scaler,
            prog=prog, phase=f"epoch {e} 학습", frac_base=e - 1, frac_span=0.5)
        va_loss, va_acc, va_ph = run_epoch(
            model, val_loader, criteria, head_weights, device,
            prog=prog, phase=f"epoch {e} 검증", frac_base=e - 0.5, frac_span=0.5)

        # 발산 감지 — NaN/inf 는 이후 epoch 에서 회복되지 않는다.
        # 남은 epoch 을 헛돌지 않게 끊고, metrics.json 을 남기지 않아
        # run_study 가 '완료' 로 집계하지 못하게 한다 (가짜 지표 차단).
        if not (math.isfinite(tr_loss) and math.isfinite(va_loss)):
            raise RuntimeError(
                f"학습이 발산했습니다 — epoch {e} 에서 loss={tr_loss}, "
                f"val_loss={va_loss} (NaN/inf). 이후 epoch 에서 회복되지 않으므로 "
                f"중단합니다. AMP 를 끄거나(고급 › AMP 사용 해제) lr 을 낮춰 "
                f"다시 시도하세요. "
                f"[arch={cfg.get('arch')} · lr={lr} · batch_size={cfg.get('batch_size')} "
                f"· amp={amp_enabled} · device={device}]")

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
        epoch_metrics = {"loss": tr_loss, "val_loss": va_loss,
                         "acc": tr_acc, "val_acc": va_acc}
        if len(heads) > 1:   # head 별 val acc 도 함께 남긴다 (예: val_acc_stage)
            epoch_metrics.update({f"val_acc_{h}": v for h, v in va_ph.items()})
        tracker.log_metrics(epoch_metrics, step=e)
        write_status(out, state="running", epoch=e)

        if patience and bad >= patience:
            log(f"Early stopping — val_loss가 {patience}epoch 연속 개선되지 않음 "
                f"(최적 epoch {best_epoch})")
            break

    if (out / "best.pt").exists():
        prog.phase("최적 가중치 로드")
        model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    prog.phase("검증 예측 수집")
    logits, labels = collect_predictions(model, val_loader, device, prog)

    # head 별 저장 + 단일 head 하위호환(logits.npy/labels.npy)
    prog.phase("예측 결과 저장")
    for h in logits:
        np.save(out / f"logits_{h}.npy", logits[h])
        np.save(out / f"labels_{h}.npy", labels[h])
    if len(heads) == 1:
        only = next(iter(logits))
        np.save(out / "logits.npy", logits[only])
        np.save(out / "labels.npy", labels[only])

    # P6 지표
    prog.phase("지표 계산")
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
    prog.phase("metrics.json 저장")
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 최종 지표 요약을 추적 백엔드(run summary)에도 남긴다 — run 비교표의 기준값
    summary = {"best_epoch": best_epoch, "epochs_ran": len(history),
               "params": n_params, "elapsed_sec": metrics["elapsed_sec"]}
    if history:
        summary["best_val_loss"] = best_val
        summary["best_val_acc"] = metrics["best_val_acc"]
    for h, pm in (per_head_metrics or {}).items():
        for k in ("accuracy", "macro_f1", "primary_recall", "abstain_rate"):
            if isinstance(pm.get(k), (int, float)):
                summary[f"{h}_{k}"] = pm[k]
    tracker.log_summary(summary)
    tracker.log_artifact(out / "metrics.json")
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
