"""run_study.py — 비교 실행기 (P11).

base config 에 --vary 조합을 얹어 여러 run 을 순차 실행하고 STUDY.md 를 만든다.
공정 비교(기획서 §4.3)를 코드로 강제한다:

  · 분할 동일성 — study 첫 run 에서 split.json 을 만들고 전 run 이 재사용
  · 조건 누락 방지 — arch/lr/batch 외 차이가 있으면 공정성 경고
  · lr 공정성 — --lr-search 로 arch 별 3점 탐색(예산 30%) 후 최적 lr 로 본 실험
  · 중단 복구 — 완료된 run(metrics.json 존재)은 건너뜀
  · 팀 분담 — --only arch=convnext_tiny 로 일부만

    python run_study.py --study study_01_backbone \
      --base configs/base_leafscan.json \
      --vary arch=resnet18,mobilenet_v3_small,efficientnet_b0 --lr-search
"""
import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.config import migrate

BASE = Path(__file__).resolve().parent
WORKER = BASE / "train_worker.py"
ALLOWED_OVERRIDE = {"arch", "lr", "lr_finetune", "batch_size"}


def log(m):
    print(m, flush=True)


def parse_vary(vary_list):
    """['arch=a,b', 'lr=1e-3,3e-3'] → {'arch':['a','b'], 'lr':['1e-3','3e-3']}."""
    out = {}
    for item in vary_list or []:
        k, _, vals = item.partition("=")
        out[k.strip()] = [v.strip() for v in vals.split(",") if v.strip()]
    return out


def _coerce(key, val):
    if key in ("lr", "lr_finetune", "val_ratio"):
        return float(val)
    if key in ("batch_size", "epochs", "img_size", "freeze_epochs", "seed"):
        return int(val)
    return val


def combos(vary):
    keys = list(vary)
    for values in product(*(vary[k] for k in keys)):
        yield {k: _coerce(k, v) for k, v in zip(keys, values)}


def run_name(study, override):
    parts = [f"{k}-{override[k]}" for k in sorted(override)]
    return f"{study}__" + "__".join(parts) if parts else f"{study}__base"


def fairness_check(base, override):
    """override 키가 허용 범위(arch/lr/batch)를 벗어나면 경고 목록 반환."""
    warns = []
    for k in override:
        if k not in ALLOWED_OVERRIDE:
            warns.append(f"공정성 경고: '{k}' 는 모델별 허용 항목이 아닙니다 "
                         f"(허용: {sorted(ALLOWED_OVERRIDE)}). 비교 해석에 주의.")
    return warns


def run_worker(config, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    import os
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, str(WORKER), "--config", str(cfg_path),
                        "--out", str(out_dir)], env=env)
    return r.returncode == 0 and (out_dir / "metrics.json").exists()


def ensure_split(base_cfg, study_dir):
    """index_csv 데이터셋이면 study split.json 을 1회 생성해 경로를 반환."""
    if base_cfg.get("dataset") != "index_csv":
        return None
    split_path = study_dir / "split.json"
    if split_path.exists():
        return split_path
    from core.data import load_index_rows
    from core.split import make_split, save_split, verify_split
    rows = load_index_rows(base_cfg.get("index_csv", "data/index.csv"))
    split = make_split(rows, policy=base_cfg.get("split_policy", "respect_provided"),
                       seed=int(base_cfg.get("seed", 42)))
    ok, _ = verify_split(rows, split)
    if not ok:
        raise SystemExit("[중단] study 분할에서 그룹 중복 발견")
    save_split(split_path, split)
    log(f"[split] {split_path} 생성 (전 run 공유)")
    return split_path


def lr_search(base_cfg, arch, study_dir, split_path):
    """arch 별 3점 lr 탐색(예산 30% epoch). 최적 lr 반환."""
    from tasks.models_registry import get_spec
    lrs = get_spec(arch).suggested_lr
    budget = max(1, int(int(base_cfg["epochs"]) * 0.3))
    best_lr, best_val = lrs[0], float("inf")
    log(f"  [lr탐색] {arch}: {lrs} · {budget}epoch")
    for lr in lrs:
        cfg = dict(base_cfg, arch=arch, lr=float(lr), epochs=budget,
                   study=study_dir.name, tags=(base_cfg.get("tags") or []) + ["lr_search"])
        if split_path:
            cfg["split_json"] = str(split_path)
        out = study_dir / f"_lrsearch__{arch}__lr-{lr}"
        if not (out / "metrics.json").exists():
            run_worker(cfg, out)
        try:
            m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            v = m.get("best_val_loss")
            if v is not None and v < best_val:
                best_val, best_lr = v, float(lr)
        except Exception:  # noqa: BLE001
            pass
    log(f"  [lr탐색] {arch} 최적 lr={best_lr} (val_loss {best_val:.4f})")
    return best_lr


def build_study_md(study_dir, members):
    lines = [f"# {study_dir.name} — 비교 결과", "",
             "> run_study.py 자동 생성. 공정 비교 프로토콜(기획서 §4.3) 적용.", "",
             "| run | arch | lr | stage macro-F1 | 정식기 recall | crop acc "
             "| params(M) | 소요(s) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, m in members:
        cfg = m.get("config", {})
        ph = m.get("per_head", {})
        st = ph.get("stage", {})
        cr = ph.get("crop", {})
        lines.append(
            f"| {name.split('__',1)[-1]} | {cfg.get('arch','')} | {cfg.get('lr','')} "
            f"| {st.get('macro_f1','—')} | {st.get('primary_recall','—')} "
            f"| {cr.get('accuracy','—')} | {m.get('params',0)/1e6:.1f} "
            f"| {m.get('elapsed_sec','—')} |")
    lines += ["", "## 결론", "", "_(사람이 작성)_", ""]
    (study_dir / "STUDY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[STUDY.md] {study_dir/'STUDY.md'} 생성")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--vary", action="append", default=[],
                    help="key=v1,v2,... (여러 번 지정 가능)")
    ap.add_argument("--only", default=None, help="key=value 로 일부 조합만 실행")
    ap.add_argument("--lr-search", action="store_true")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    base_cfg = migrate(json.loads(Path(args.base).read_text(encoding="utf-8")))
    study_dir = Path(args.runs_dir) / args.study
    study_dir.mkdir(parents=True, exist_ok=True)
    split_path = ensure_split(base_cfg, study_dir)

    vary = parse_vary(args.vary)
    only_k = only_v = None
    if args.only:
        only_k, _, only_v = args.only.partition("=")

    members = []
    for override in (combos(vary) if vary else [{}]):
        if only_k and str(override.get(only_k)) != only_v:
            continue
        for w in fairness_check(base_cfg, override):
            log(w)
        arch = override.get("arch", base_cfg.get("arch", "simple_cnn"))
        if args.lr_search and "lr" not in override:
            override["lr"] = lr_search(base_cfg, arch, study_dir, split_path)

        cfg = dict(base_cfg, study=study_dir.name)
        cfg.update(override)
        if split_path:
            cfg["split_json"] = str(split_path)
        name = run_name(args.study, override)
        out = study_dir / name
        if (out / "metrics.json").exists():
            log(f"[건너뜀] {name} — 이미 완료됨")
        else:
            log(f"[실행] {name} · {override}")
            if not run_worker(cfg, out):
                log(f"[실패] {name}")
                continue
        try:
            members.append((name, json.loads(
                (out / "metrics.json").read_text(encoding="utf-8"))))
        except Exception:  # noqa: BLE001
            pass

    if members:
        build_study_md(study_dir, members)
    log(f"[완료] {len(members)}개 run · {study_dir}")


if __name__ == "__main__":
    main()
