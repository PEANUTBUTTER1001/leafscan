"""core/split.py — 데이터 분할 (P4b).

이 데이터는 두 제약이 충돌한다 (05_데이터_명세서 §5):
  · 그룹 분할 — 한 개체를 한 달간 27~40장 연속 촬영 → 개체가 양쪽에 섞이면 과대평가
  · 층화 분할 — 한 아카이브에 한 단계만 있을 수 있음 → 단계 비율 유지 필요
둘을 동시에 만족시키는 것은 **StratifiedGroupKFold** 뿐이다.
(GroupShuffleSplit 는 층화 불가, train_test_split(stratify) 는 그룹 유지 불가.)

split_policy:
  · respect_provided (기본) — val = validation 폴더, test = training 에서 그룹 분리
  · resplit_all — 전체를 모아 StratifiedGroupKFold 로 70:15:15
분할 결과는 group_id 목록으로 split.json 에 고정 → study 내 재사용(공정성).
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def _combo_labels(rows):
    """작물×단계 조합을 정수 라벨로. 층화 대상."""
    combos = sorted({(r["crop"], r["stage"]) for r in rows})
    idx = {c: i for i, c in enumerate(combos)}
    return np.array([idx[(r["crop"], r["stage"])] for r in rows])


def _group_array(rows):
    return np.array([r["group_id"] for r in rows])


def _sgk_holdout(y, groups, frac, seed):
    """StratifiedGroupKFold 로 대략 frac 비율의 홀드아웃(그룹 단위) 인덱스를 뽑는다."""
    n_groups = len(set(groups))
    k = max(2, min(n_groups, round(1 / max(frac, 1e-6))))
    sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    X = np.zeros((len(y), 1))
    tr, ho = next(iter(sgk.split(X, y, groups)))
    return tr, ho


def resplit_all(rows, seed=42, val_frac=0.15, test_frac=0.15):
    """전체를 StratifiedGroupKFold 로 train/val/test 로 나눈다 (그룹 단위)."""
    y = _combo_labels(rows)
    groups = _group_array(rows)
    idx_all = np.arange(len(rows))

    # 1) test 홀드아웃
    trv_local, test_local = _sgk_holdout(y, groups, test_frac, seed)
    test_idx = idx_all[test_local]
    trv_idx = idx_all[trv_local]

    # 2) 남은 것에서 val 홀드아웃
    y2, g2 = y[trv_local], groups[trv_local]
    if len(set(g2)) >= 2:
        rel_val_frac = val_frac / max(1e-6, (1 - test_frac))
        tr_local2, val_local2 = _sgk_holdout(y2, g2, rel_val_frac, seed + 1)
        train_idx = trv_idx[tr_local2]
        val_idx = trv_idx[val_local2]
    else:
        # 그룹이 너무 적으면 val=test 로 대체(스모크 한정)
        train_idx, val_idx = trv_idx, test_idx
    return _to_groups(rows, train_idx, val_idx, test_idx)


def respect_provided(rows, seed=42, test_frac=0.15):
    """val = split_source==validation, test = training 에서 그룹 단위 분리.

    어느 한쪽이 비면 resplit_all 로 폴백(경고)."""
    tr_rows = [i for i, r in enumerate(rows) if r["split_source"] == "training"]
    va_rows = [i for i, r in enumerate(rows) if r["split_source"] == "validation"]
    if not tr_rows or not va_rows:
        return None  # 호출측에서 resplit_all 폴백
    rows_arr = np.array(rows, dtype=object)
    tr_sub = [rows[i] for i in tr_rows]
    y = _combo_labels(tr_sub)
    groups = _group_array(tr_sub)
    if len(set(groups)) >= 2:
        keep_local, test_local = _sgk_holdout(y, groups, test_frac, seed)
        train_idx = np.array(tr_rows)[keep_local]
        test_idx = np.array(tr_rows)[test_local]
    else:
        train_idx = np.array(tr_rows)
        test_idx = np.array(va_rows)
    val_idx = np.array(va_rows)
    return _to_groups(rows, train_idx, val_idx, test_idx)


def _to_groups(rows, train_idx, val_idx, test_idx):
    def gset(idx):
        return sorted({rows[int(i)]["group_id"] for i in idx})
    return {
        "train": gset(train_idx),
        "val": gset(val_idx),
        "test": gset(test_idx),
    }


def make_split(rows, policy="respect_provided", seed=42,
               val_frac=0.15, test_frac=0.15):
    """정책에 따라 group_id 단위 분할을 만든다. {"train":[gid],"val":[..],"test":[..]}."""
    if policy == "respect_provided":
        res = respect_provided(rows, seed=seed, test_frac=test_frac)
        if res is None:
            res = resplit_all(rows, seed=seed, val_frac=val_frac, test_frac=test_frac)
            res["_note"] = "respect_provided 불가(한쪽 split 비어있음) → resplit_all 폴백"
    elif policy == "resplit_all":
        res = resplit_all(rows, seed=seed, val_frac=val_frac, test_frac=test_frac)
    else:
        raise ValueError(f"미지의 split_policy: {policy}")
    res["policy"] = policy
    res["seed"] = seed
    return res


def assign_rows(rows, split):
    """분할(그룹 목록)을 실제 행 인덱스로 매핑. train/val/test 인덱스 리스트 반환."""
    gsplit = {"train": set(split["train"]), "val": set(split["val"]),
              "test": set(split["test"])}
    out = {"train": [], "val": [], "test": []}
    for i, r in enumerate(rows):
        for part, gs in gsplit.items():
            if r["group_id"] in gs:
                out[part].append(i)
                break
    return out


def verify_split(rows, split):
    """그룹 중복 0건 검증 + 단계 비율 계산. (ok, report) 반환."""
    parts = {k: set(split[k]) for k in ("train", "val", "test")}
    overlaps = {
        "train∩val": parts["train"] & parts["val"],
        "train∩test": parts["train"] & parts["test"],
        "val∩test": parts["val"] & parts["test"],
    }
    idx = assign_rows(rows, split)
    ratios = {}
    for part, ilist in idx.items():
        c = Counter(rows[i]["stage"] for i in ilist)
        tot = sum(c.values())
        ratios[part] = {k: round(v / tot, 3) for k, v in c.items()} if tot else {}
        ratios[part]["_n"] = tot
        ratios[part]["_groups"] = len(parts[part])
    ok = all(len(v) == 0 for v in overlaps.values())
    return ok, {"overlaps": {k: sorted(v) for k, v in overlaps.items()},
                "ratios": ratios}


def save_split(path, split):
    Path(path).write_text(json.dumps(split, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def load_split(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
