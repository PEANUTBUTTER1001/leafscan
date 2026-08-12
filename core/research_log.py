"""core/research_log.py — RESEARCH_LOG.md 생성 (P8).

전체 연구 서사 인덱스. study 목록 + 지표 진행 타임라인.
RESEARCH_LOG.md 만 읽고 연구 흐름을 이해할 수 있어야 한다 (축 B7).
"""
import json
from datetime import datetime
from pathlib import Path


def _load(run: Path):
    try:
        return json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _stage_f1(m):
    return (m.get("per_head", {}).get("stage", {}) or {}).get("macro_f1")


def _recall(m):
    return (m.get("per_head", {}).get("stage", {}) or {}).get("primary_recall")


def _conclusion_done(run: Path):
    rp = run / "report.md"
    if not rp.exists():
        return "⚠️"
    return "⚠️" if "⚠️ 미작성" in rp.read_text(encoding="utf-8") else "✓"


def build_research_log(runs_dir):
    runs_dir = Path(runs_dir)
    studies = {}          # study_name -> [(run, metrics)]
    solo = []
    seen = set()
    # 최상위 run 과 study 하위의 member run(runs/{study}/{member}/) 모두 수집
    for mp in sorted(runs_dir.rglob("metrics.json")):
        run = mp.parent
        if run in seen or any(part.startswith("_") for part in run.relative_to(runs_dir).parts):
            continue
        seen.add(run)
        m = _load(run)
        if m is None:
            continue
        study = (m.get("config", {}) or {}).get("study")
        if study:
            studies.setdefault(study, []).append((run, m))
        else:
            solo.append((run, m))

    lines = ["# LeafScan 연구 로그 (RESEARCH_LOG)", "",
             f"> 자동 생성 {datetime.now().isoformat(timespec='seconds')}. "
             "study 목록과 지표 진행을 한 장에 모은다.", ""]

    # study 목록
    lines += ["## Study 목록", ""]
    if studies:
        lines += ["| study | run 수 | 최고 stage macro-F1 | 최고 정식기 recall |",
                  "|---|---|---|---|"]
        for name, runs in sorted(studies.items()):
            f1s = [x for x in (_stage_f1(m) for _, m in runs) if x is not None]
            recs = [x for x in (_recall(m) for _, m in runs) if x is not None]
            sd = runs_dir / name / "STUDY.md"
            link = f"[{name}]({name}/STUDY.md)" if sd.exists() else name
            lines.append(f"| {link} | {len(runs)} | "
                         f"{max(f1s) if f1s else '—'} | {max(recs) if recs else '—'} |")
    else:
        lines.append("_(아직 study 없음. run_study.py 로 study 를 실행하세요.)_")

    # 지표 진행 타임라인 (전체 run, 시간순)
    lines += ["", "## 지표 진행 타임라인", "",
              "| run | study | arch | stage macro-F1 | 정식기 recall | 결론 |",
              "|---|---|---|---|---|---|"]
    allruns = [(r, m) for rs in studies.values() for r, m in rs] + solo
    allruns.sort(key=lambda rm: rm[0].name)
    for run, m in allruns:
        cfg = m.get("config", {})
        rel = run.relative_to(runs_dir).as_posix()
        lines.append(
            f"| [{run.name}]({rel}/report.md) | {cfg.get('study') or '—'} "
            f"| {cfg.get('arch','')} | {_stage_f1(m) or '—'} "
            f"| {_recall(m) or '—'} | {_conclusion_done(run)} |")

    out = runs_dir / "RESEARCH_LOG.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
