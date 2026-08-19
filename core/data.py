"""core/data.py — 데이터 팩토리.

P1: build_transforms / FakeCIFAR / build_datasets 를 train_worker.py 에서
    **변경 없이** 이동. build_data(cfg) 도입. 동작 불변.

이후 P4b 에서 index_csv 데이터셋·StratifiedGroupKFold·crop_mode 가 붙는다.
"""
import csv
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Subset, random_split
from torchvision import datasets, transforms

from tasks.cifar10 import CIFAR10_CLASSES
from tasks.leafscan import CANON_CROPS, CANON_STAGES, order_present


def _log(msg):
    # 워커가 주입할 수도 있으나, 기본은 무음(모듈 단독 사용 대비).
    import json
    import sys
    print(json.dumps({"event": "log", "message": msg}, ensure_ascii=False),
          flush=True, file=sys.stdout)


def build_transforms(augment):
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_tf = [transforms.ToTensor(), norm]
    if augment:
        train_tf = [transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip()] + train_tf
    return transforms.Compose(train_tf), transforms.Compose([transforms.ToTensor(), norm])


class FakeCIFAR(torch.utils.data.Dataset):
    """네트워크 없이 파이프라인만 점검할 때 쓰는 더미 데이터셋."""

    def __init__(self, n=512, num_classes=10):
        g = torch.Generator().manual_seed(0)
        self.y = torch.randint(0, num_classes, (n,), generator=g)
        base = torch.randn(num_classes, 3, 32, 32, generator=g)
        self.x = base[self.y] + 0.3 * torch.randn(n, 3, 32, 32, generator=g)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def build_datasets(cfg, log=None):
    log = log or _log
    if cfg["dataset"] == "fake":
        full = FakeCIFAR(n=max(256, cfg.get("subset") or 1024))
        classes = [f"class_{i}" for i in range(10)]
    else:
        train_tf, eval_tf = build_transforms(cfg["augment"])
        log("CIFAR-10 준비 중… (최초 1회 다운로드, 약 170MB)")
        full = datasets.CIFAR10(root=cfg["data_root"], download=True,
                                train=True, transform=train_tf)
        classes = CIFAR10_CLASSES

    if cfg.get("subset"):
        n = min(int(cfg["subset"]), len(full))
        idx = list(range(len(full)))
        random.Random(cfg["seed"]).shuffle(idx)
        full = Subset(full, idx[:n])

    val_ratio = float(cfg["val_ratio"])
    n_val = max(1, int(len(full) * val_ratio))
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(cfg["seed"])
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=g)
    return train_ds, val_ds, classes


def build_data(cfg, log=None):
    """(train_ds, val_ds, classes) 반환. 단일 head 경로(fake/cifar10)."""
    return build_datasets(cfg, log=log)


# ==========================================================================
# P4b: index.csv 데이터셋 · 멀티라벨 · StratifiedGroupKFold · crop_mode · ambiguous
# ==========================================================================
def _tf_norm(norm):
    (mean, std) = norm
    return transforms.Normalize(mean, std)


def build_leaf_transforms(img_size, norm, augment):
    """train(augmentation)/eval(리사이즈+정규화) transform 분리."""
    norm_t = _tf_norm(norm)
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), norm_t])
    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(), norm_t])
    else:
        train_tf = eval_tf
    return train_tf, eval_tf


def load_index_rows(index_csv):
    """index.csv → list[dict]. width/height int, ambiguous bool, bbox tuple|None."""
    rows = []
    with open(index_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["width"] = int(r["width"]) if r.get("width") else 0
            r["height"] = int(r["height"]) if r.get("height") else 0
            r["ambiguous"] = str(r.get("ambiguous", "")).lower() in ("true", "1")
            bb = r.get("bbox", "")
            if bb:
                try:
                    r["bbox_t"] = tuple(float(x) for x in bb.split(","))
                except ValueError:
                    r["bbox_t"] = None
            else:
                r["bbox_t"] = None
            rows.append(r)
    return rows


def crop_by_mode(img, bbox, mode):
    """PIL 이미지에 crop_mode 적용. bbox=(x,y,w,h). 없으면 full 로 폴백."""
    if mode == "full" or not bbox:
        return img
    x, y, w, h = bbox
    if mode == "bbox_expand":
        cx, cy = x + w / 2, y + h / 2
        w, h = w * 1.25, h * 1.25   # 20~30% 확장
        x, y = cx - w / 2, cy - h / 2
    W, H = img.size
    left = max(0, int(x)); top = max(0, int(y))
    right = min(W, int(x + w)); bottom = min(H, int(y + h))
    if right - left < 8 or bottom - top < 8:
        return img
    return img.crop((left, top, right, bottom))


def cache_image(r, data_root, cache_dir, cache_size=256, crop_mode="full"):
    """row 하나의 사전 리사이즈 캐시를 보장한다. 반환: True=새로 생성, False=이미 있음.

    IndexCsvDataset 의 지연 캐시와 scripts/preprocess.py 의 일괄 생성이 공유하는
    단일 구현. bbox 좌표는 원본 해상도 기준이므로 crop 을 리사이즈 전에 적용한다.
    tmp 파일 + os.replace 로 원자적 교체 — 병렬 worker 동시 접근 안전.
    """
    from PIL import Image
    cpath = Path(cache_dir) / r["path"]
    if cpath.exists():
        return False
    with Image.open(Path(data_root) / r["path"]) as im:
        im = im.convert("RGB")
        im = crop_by_mode(im, r.get("bbox_t"), crop_mode)
    im = im.resize((int(cache_size), int(cache_size)), Image.BILINEAR)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = cpath.with_name(cpath.name + f".tmp{os.getpid()}")
    im.save(tmp, "JPEG", quality=92)
    os.replace(tmp, cpath)
    return True


class IndexCsvDataset(torch.utils.data.Dataset):
    """index.csv 기반 멀티헤드 데이터셋. (image, {"crop":int,"stage":int}) 반환.

    · 이미지는 convert('RGB') 강제 (흑백/RGBA 혼재 대응)
    · crop_mode: full / bbox / bbox_expand
    · cache_dir: 사전 리사이즈 캐시. 원본(1500×2000급) JPEG 디코딩이 epoch 시간의
      병목이므로, 최초 접근 시 crop_mode 적용 후 cache_size²로 줄여 저장하고
      이후에는 캐시만 읽는다. bbox 좌표는 원본 해상도 기준이므로 반드시
      리사이즈 전에 crop 을 적용한다 (캐시 경로가 crop_mode 별로 분리되는 이유).
    """

    def __init__(self, rows, indices, crop_names, stage_names, transform,
                 crop_mode="full", data_root=".", cache_dir=None, cache_size=256):
        self.rows = rows
        self.indices = list(indices)
        self.crop_idx = {c: i for i, c in enumerate(crop_names)}
        self.stage_idx = {s: i for i, s in enumerate(stage_names)}
        self.transform = transform
        self.crop_mode = crop_mode
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_size = int(cache_size)

    def __len__(self):
        return len(self.indices)

    def _load_original(self, r):
        from PIL import Image
        with Image.open(self.data_root / r["path"]) as im:
            im = im.convert("RGB")
            return crop_by_mode(im, r.get("bbox_t"), self.crop_mode)

    def _load_image(self, r):
        from PIL import Image
        if self.cache_dir is None:
            return self._load_original(r)
        cpath = self.cache_dir / r["path"]
        if not cpath.exists():
            cache_image(r, self.data_root, self.cache_dir,
                        self.cache_size, self.crop_mode)
        with Image.open(cpath) as im:
            return im.convert("RGB")

    def __getitem__(self, i):
        r = self.rows[self.indices[i]]
        x = self.transform(self._load_image(r))
        y = {"crop": self.crop_idx[r["crop"]], "stage": self.stage_idx[r["stage"]]}
        return x, y


class MultiLabelFake(torch.utils.data.Dataset):
    """P3 테스트용 멀티라벨 더미. (image, {"crop":int,"stage":int}) 반환. 결정적."""

    def __init__(self, n=256, n_crop=4, n_stage=3, img_size=32):
        g = torch.Generator().manual_seed(0)
        self.yc = torch.randint(0, n_crop, (n,), generator=g)
        self.ys = torch.randint(0, n_stage, (n,), generator=g)
        base_c = torch.randn(n_crop, 3, img_size, img_size, generator=g)
        base_s = torch.randn(n_stage, 3, img_size, img_size, generator=g)
        self.x = (base_c[self.yc] + base_s[self.ys]
                  + 0.3 * torch.randn(n, 3, img_size, img_size, generator=g))
        self.n_crop, self.n_stage = n_crop, n_stage

    def __len__(self):
        return len(self.yc)

    def __getitem__(self, i):
        return self.x[i], {"crop": int(self.yc[i]), "stage": int(self.ys[i])}


def _apply_ambiguous(rows, indices, policy):
    """ambiguous 정책 적용. exclude=제외, include/downweight=유지(가중은 손실단계에서).

    반환: (사용할 indices, per-sample weight dict or None)"""
    if policy == "exclude":
        keep = [i for i in indices if not rows[i]["ambiguous"]]
        return keep, None
    return list(indices), None  # include/downweight 은 인덱스 유지


def build_multihead_data(cfg, log=None):
    """index.csv / multilabel_fake 로부터 멀티헤드 학습용 데이터를 만든다.

    반환: (train_ds, val_ds, test_ds, meta)
      meta = {heads, crop_names, stage_names, split, split_report}
    """
    from core.split import (assign_rows, load_split, make_split, save_split,
                            verify_split)
    log = log or _log
    dataset = cfg["dataset"]

    if dataset == "multilabel_fake":
        n = max(64, int(cfg.get("subset") or 256))
        crop_names = CANON_CROPS[:]
        stage_names = CANON_STAGES[:]
        full = MultiLabelFake(n=n, img_size=int(cfg.get("img_size", 32)))
        val_ratio = float(cfg.get("val_ratio", 0.3))
        n_val = max(1, int(n * val_ratio))
        g = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
        tr, va = random_split(full, [n - n_val, n_val], generator=g)
        meta = {"heads": {"crop": len(crop_names), "stage": len(stage_names)},
                "crop_names": crop_names, "stage_names": stage_names,
                "split": None, "split_report": None}
        return tr, va, va, meta

    # index_csv 경로
    index_csv = cfg.get("index_csv") or cfg.get("data_index") or "data/index.csv"
    index_csv = Path(index_csv)
    rows = load_index_rows(index_csv)
    crop_names = order_present({r["crop"] for r in rows}, CANON_CROPS)
    stage_names = order_present({r["stage"] for r in rows}, CANON_STAGES)

    # 분할 (split.json 고정·재사용)
    seed = int(cfg.get("seed", 42))
    policy = cfg.get("split_policy", "respect_provided")
    split_path = cfg.get("split_json")
    if split_path and Path(split_path).exists():
        split = load_split(split_path)
        log(f"기존 split 재사용: {split_path}")
    else:
        split = make_split(rows, policy=policy, seed=seed)
        if split_path:
            save_split(split_path, split)

    ok, report = verify_split(rows, split)
    log(f"분할 정책={policy} · 그룹중복 {'없음(0건)' if ok else '발견!'} · "
        f"train/val/test 그룹={report['ratios']['train']['_groups']}/"
        f"{report['ratios']['val']['_groups']}/{report['ratios']['test']['_groups']}")
    for part in ("train", "val", "test"):
        rr = report["ratios"][part]
        log(f"  {part}: {rr['_n']}장 · 단계비율 "
            f"{ {k:v for k,v in rr.items() if not k.startswith('_')} }")
    if not ok:
        raise RuntimeError(f"분할 그룹 중복 발견: {report['overlaps']}")

    idx = assign_rows(rows, split)
    amb_policy = cfg.get("ambiguous_policy", "include")
    idx["train"], _ = _apply_ambiguous(rows, idx["train"], amb_policy)

    # subset: 전체 N장으로 제한 (빠른확인용). split 별 비율 유지, 시드 고정.
    # 그룹 분할이 끝난 뒤 각 split 안에서만 뽑으므로 그룹 누수는 생기지 않는다.
    subset = int(cfg.get("subset") or 0)
    if subset and subset < len(rows):
        frac = subset / len(rows)
        rng = random.Random(seed)
        for part in ("train", "val", "test"):
            k = max(1, round(len(idx[part]) * frac))
            if k < len(idx[part]):
                idx[part] = sorted(rng.sample(list(idx[part]), k))
        log(f"subset={subset} 적용 — train {len(idx['train'])}장 / "
            f"val {len(idx['val'])}장 / test {len(idx['test'])}장 "
            f"(전체 {len(rows)}장에서 축소)")

    # transform: arch 의 norm/size 를 따른다
    from tasks.models_registry import get_spec
    arch = cfg.get("arch", "resnet18")
    spec = get_spec(arch)
    img_size = int(cfg.get("img_size") or spec.default_size)
    train_tf, eval_tf = build_leaf_transforms(img_size, spec.norm,
                                              bool(cfg.get("augment", False)))
    crop_mode = cfg.get("crop_mode", "full")
    data_root = index_csv.parent

    # 사전 리사이즈 캐시. 빠른 학습(fast_train, 기본 체크) 이면 원본
    # (예: 915×1060)을 최초 epoch 에 cache_size(256px)로 축소·저장해 재사용한다.
    # 체크 해제 시 캐시 없이 원본 이미지를 매번 그대로 디코딩해 학습한다
    # (품질 우선 · epoch 시간이 크게 늘어난다).
    cache_size = int(cfg.get("cache_size", 256))
    cache_dir = None
    fast_train = bool(cfg.get("fast_train", cfg.get("image_cache", True)))
    if fast_train and img_size <= cache_size:
        cache_dir = data_root / "_cache" / f"{crop_mode}_{cache_size}"
        log(f"빠른 학습 — 원본을 {cache_size}px 로 축소한 캐시 사용: {cache_dir} "
            f"(최초 epoch 에 생성, 이후 재사용)")
    elif not fast_train:
        log("빠른 학습 해제 — 원본 해상도 이미지를 그대로 디코딩해 학습합니다 "
            "(품질 우선 · epoch 시간이 크게 늘어납니다)")
    mk = lambda ilist, tf: IndexCsvDataset(
        rows, ilist, crop_names, stage_names, tf, crop_mode, data_root,
        cache_dir=cache_dir, cache_size=cache_size)
    train_ds = mk(idx["train"], train_tf)
    val_ds = mk(idx["val"], eval_tf)
    test_ds = mk(idx["test"], eval_tf)
    meta = {"heads": {"crop": len(crop_names), "stage": len(stage_names)},
            "crop_names": crop_names, "stage_names": stage_names,
            "split": split, "split_report": report}
    return train_ds, val_ds, test_ds, meta
