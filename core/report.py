"""core/report.py — report.md 생성 · 자동 관찰 규칙 (P9b).

구조도 §8 의 9섹션. 사람이 채우는 칸은 §1 가설·§7 결론뿐이다.
자동 생성부(auto)와 사람 작성부(fill)를 앵커로 나눠 서로 덮어쓰지 않는다.
  <!--auto:results-start--> … 재생성 시 교체 <!--auto:results-end-->
  <!--fill:conclusion-start--> … 사람이 쓴 것, 보존 <!--fill:conclusion-end-->

자동 관찰(§6)은 **사실만 짚고 판단하지 않는다** (구조도 §8.1). 해석은 사람의 몫.
"""
import json
import re
from pathlib import Path

from tasks.leafscan import TARGETS


# --------------------------------------------------------------------------
# 앵커 유틸
# --------------------------------------------------------------------------
def _fill_block(kind, name, content):
    return (f"<!--{kind}:{name}-start-->\n{content}\n<!--{kind}:{name}-end-->")


def _extract(text, kind, name, default=""):
    m = re.search(rf"<!--{kind}:{name}-start-->\n?(.*?)\n?<!--{kind}:{name}-end-->",
                  text or "", re.DOTALL)
    return m.group(1).strip() if m else default


# --------------------------------------------------------------------------
# 자동 관찰 규칙 (사실만)
# --------------------------------------------------------------------------
def auto_observations(metrics):
    obs = []
    hist = metrics.get("history") or []
    per_head = metrics.get("per_head") or {}

    # 학습 실패 — loss NaN 또는 초기값 유지
    if hist:
        losses = [h["loss"] for h in hist]
        if any(l != l for l in losses):  # NaN
            obs.append("🔴 loss 가 NaN — 학습이 발산했습니다.")
        elif len(losses) >= 2 and abs(losses[-1] - losses[0]) < 1e-6:
            obs.append("🔴 loss 가 초기값에서 변하지 않음 — 학습이 진행되지 않았습니다.")

    # 과적합 — val_loss 연속 상승
    if len(hist) >= 4:
        vl = [h["val_loss"] for h in hist]
        rises = 0; start = None
        for i in range(1, len(vl)):
            if vl[i] > vl[i - 1]:
                rises += 1
                start = start or hist[i]["epoch"]
            else:
                rises = 0; start = None
            if rises >= 3:
                obs.append(f"🟡 epoch {start}부터 val_loss 가 연속 상승 — 과적합 신호.")
                break

    # head 별 최약 클래스 / 오분류 편중
    for head, hm in per_head.items():
        macro = hm.get("macro_f1", 0)
        pc = hm.get("per_class") or {}
        if pc:
            weak = min(pc, key=lambda n: pc[n]["f1"])
            if macro - pc[weak]["f1"] >= 0.10:
                obs.append(f"🔴 [{head}] 최약 클래스 '{weak}' F1={pc[weak]['f1']:.3f} "
                           f"(macro {macro:.3f}보다 0.10 이상 낮음).")
        cm = hm.get("confusion")
        names = hm.get("class_names") or []
        if cm and len(names) >= 2:
            for i, row in enumerate(cm):
                tot = sum(row)
                if tot == 0:
                    continue
                for j, v in enumerate(row):
                    if i != j and v / tot > 0.60:
                        obs.append(f"🔴 [{head}] '{names[i]}'→'{names[j]}' 오분류가 "
                                   f"{v/tot*100:.0f}% 로 한 방향에 쏠림.")

    if not obs:
        obs.append("특이 관찰 없음 (규칙 기반).")
    return obs


# --------------------------------------------------------------------------
# 섹션 렌더
# --------------------------------------------------------------------------
def _metric_table(metrics, baseline=None):
    per_head = metrics.get("per_head") or {}
    lines = ["| 지표 | 값 | 기준Δ | 목표 |", "|---|---|---|---|"]

    def row(label, val, target_key=None, base=None, hib=True, fmt="{:.3f}"):
        d = ""
        if base is not None and val is not None:
            delta = val - base
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "＝")
            d = f"{arrow}{delta:+.3f}"
        tg = ""
        if target_key and target_key in TARGETS:
            tg = f"{TARGETS[target_key]:.2f}"
        vs = fmt.format(val) if val is not None else "—"
        lines.append(f"| {label} | {vs} | {d} | {tg} |")

    bph = (baseline or {}).get("per_head", {}) if baseline else {}
    if "stage" in per_head:
        sm = per_head["stage"]
        bm = bph.get("stage", {})
        row("stage macro-F1", sm.get("macro_f1"), "stage_macro_f1",
            bm.get("macro_f1"))
        if "primary_recall" in sm:
            row(f"{sm.get('primary_class','정식기')} recall", sm["primary_recall"],
                "stage_recall_정식기",
                bm.get("per_class", {}).get(sm.get("primary_class", ""), {}).get("recall"))
    if "crop" in per_head:
        cm = per_head["crop"]
        row("crop accuracy", cm.get("accuracy"), "crop_accuracy",
            bph.get("crop", {}).get("accuracy"))
    for head in per_head:
        if head not in ("crop", "stage"):
            row(f"{head} macro-F1", per_head[head].get("macro_f1"))
    return "\n".join(lines)


def _per_head_prf(metrics):
    per_head = metrics.get("per_head") or {}
    blocks = []
    for head, hm in per_head.items():
        pc = hm.get("per_class") or {}
        lines = [f"**{head}** (acc {hm.get('accuracy')}, macro-F1 {hm.get('macro_f1')})",
                 "", "| 클래스 | P | R | F1 | n |", "|---|---|---|---|---|"]
        for name, m in pc.items():
            lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} "
                         f"| {m['f1']:.3f} | {m['support']} |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(head 지표 없음)"


def _figures_embed(run: Path):
    fig_dir = run / "figures"
    if not fig_dir.is_dir():
        return "(그림 없음 — `python make_figures.py --run <run>` 실행)"
    pngs = sorted(fig_dir.glob("*.png"))
    if not pngs:
        return "(그림 없음)"
    return "\n\n".join(f"### {p.stem}\n\n![{p.stem}](figures/{p.name})" for p in pngs)


def _config_summary(cfg):
    keys = ["arch", "dataset", "img_size", "epochs", "batch_size", "lr",
            "optimizer", "crop_mode", "split_policy", "class_weight",
            "head_weights", "freeze_epochs", "pretrained"]
    lines = ["| 항목 | 값 |", "|---|---|"]
    for k in keys:
        if k in cfg and cfg[k] not in (None, ""):
            lines.append(f"| {k} | `{cfg[k]}` |")
    return "\n".join(lines)


def _diff_summary(run: Path, cfg):
    dp = run / "diff.json"
    if not dp.exists():
        parent = cfg.get("parent_run")
        return f"부모 run: {parent}" if parent else "부모 run 없음 (최초 실험)."
    diff = json.loads(dp.read_text(encoding="utf-8"))
    if not diff:
        return "부모와 설정 차이 없음."
    lines = ["| 항목 | 부모 | 이번 |", "|---|---|---|"]
    for k, (a, b) in diff.items():
        lines.append(f"| {k} | `{a}` | `{b}` |")
    return "\n".join(lines)


def _next_suggestions(metrics):
    per_head = metrics.get("per_head") or {}
    cand = []
    sm = per_head.get("stage")
    if sm and sm.get("primary_recall", 1) < TARGETS.get("stage_recall_정식기", 0.65):
        cand.append("- 정식기 recall 이 목표 미달 → `class_weight=auto` 또는 head 가중치 조정")
    if metrics.get("config", {}).get("crop_mode", "full") == "full":
        cand.append("- `crop_mode=bbox_expand` 로 입력 crop 실험 (study 04)")
    if not metrics.get("config", {}).get("pretrained"):
        cand.append("- `pretrained=true` + `freeze_epochs` 전이학습 스케줄 비교 (study 06)")
    if not cand:
        cand.append("- (자동 제안 없음)")
    return "\n".join(cand)


def _reproduce(run: Path, cfg):
    env = {}
    ep = run / "env.json"
    if ep.exists():
        env = json.loads(ep.read_text(encoding="utf-8"))
    lines = [
        "```bash",
        f"python train_worker.py --config {run.name}/config.json --out {run.name}",
        "```", "",
        f"- torch {env.get('torch','?')} / torchvision {env.get('torchvision','?')}",
        f"- device {env.get('device','?')} · GPU {env.get('gpu')}",
        f"- git {env.get('git_commit')}",
    ]
    idx = env.get("index_hash")
    if idx:
        lines.append(f"- 데이터 인덱스 해시 {idx[:23]}…")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 메인
# --------------------------------------------------------------------------
def generate_report(run_dir, baseline=None):
    run = Path(run_dir)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    cfg = metrics.get("config", {})
    report_path = run / "report.md"
    prev = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

    # 사람 작성부 보존 (없으면 config 의 hypothesis 로 시드)
    hypo = _extract(prev, "fill", "hypothesis",
                    cfg.get("hypothesis", "") or "_(가설을 입력하세요)_")
    concl = _extract(prev, "fill", "conclusion", "⚠️ 미작성")

    tags = ", ".join(cfg.get("tags") or []) or "—"
    header = (f"# Run `{run.name}`\n\n"
              f"- study: {cfg.get('study') or '—'} · parent: "
              f"{cfg.get('parent_run') or '—'} · tags: {tags}\n"
              f"- 소요: {metrics.get('elapsed_sec','?')}초 · epochs "
              f"{metrics.get('epochs_ran','?')} · params {metrics.get('params','?'):,}\n")

    obs = "\n".join(f"- {o}" for o in auto_observations(metrics))

    parts = [
        header,
        "## 1. 가설\n\n" + _fill_block("fill", "hypothesis", hypo),
        "## 2. 무엇을 바꿨나\n\n"
        + _fill_block("auto", "diff", _diff_summary(run, cfg)),
        "## 3. 설정 요약\n\n" + _fill_block("auto", "config", _config_summary(cfg)),
        "## 4. 결과\n\n" + _fill_block(
            "auto", "results",
            _metric_table(metrics, baseline) + "\n\n" + _per_head_prf(metrics)),
        "## 5. 그림\n\n" + _fill_block("auto", "figures", _figures_embed(run)),
        "## 6. 자동 관찰\n\n" + _fill_block("auto", "observations", obs),
        "## 7. 결론\n\n" + _fill_block("fill", "conclusion", concl),
        "## 8. 다음 실험 제안\n\n"
        + _fill_block("auto", "suggestions", _next_suggestions(metrics)),
        "## 9. 재현 정보\n\n"
        + _fill_block("auto", "reproduce", _reproduce(run, cfg)),
    ]
    report_path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return report_path, ("⚠️ 미작성" not in concl)


# --------------------------------------------------------------------------
# runs/index.csv 카탈로그
# --------------------------------------------------------------------------
def update_catalog(runs_dir):
    import csv
    runs_dir = Path(runs_dir)
    rows = []
    for run in sorted(runs_dir.glob("*/")):
        mp = run / "metrics.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        cfg = m.get("config", {})
        ph = m.get("per_head", {})
        stage = ph.get("stage", {})
        concl_done = ""
        rp = run / "report.md"
        if rp.exists():
            concl_done = "" if "⚠️ 미작성" in _extract(
                rp.read_text(encoding="utf-8"), "fill", "conclusion", "⚠️ 미작성") else "✓"
        rows.append({
            "run": run.name,
            "study": cfg.get("study", ""),
            "arch": cfg.get("arch", ""),
            "dataset": cfg.get("dataset", ""),
            "stage_macro_f1": stage.get("macro_f1", ""),
            "primary_recall": stage.get("primary_recall", ""),
            "val_acc": m.get("best_val_acc", ""),
            "hypothesis": (cfg.get("hypothesis", "") or "")[:40],
            "conclusion": concl_done or "⚠️",
        })
    if not rows:
        return None
    cat = runs_dir / "index.csv"
    with open(cat, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return cat
