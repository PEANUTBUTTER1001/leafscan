"""make_report.py — report.md 생성 (별도 프로세스) · P9b.

run 디렉터리의 metrics·diff·env·figures 를 읽어 report.md 를 만든다.
사람이 쓴 §1 가설·§7 결론은 앵커로 보존되고, 자동 섹션만 재생성된다.
runs/index.csv 카탈로그도 갱신한다.

    python make_report.py --run runs/exp_007
    python make_report.py --run runs/exp_007 --baseline runs/exp_005
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.report import generate_report, update_catalog
from core.research_log import build_research_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline", default=None, help="기준 run 디렉터리 (델타 표시)")
    ap.add_argument("--no-catalog", action="store_true")
    args = ap.parse_args()
    run = Path(args.run)
    if not (run / "metrics.json").exists():
        raise SystemExit(f"[중단] metrics.json 없음: {run}")

    baseline = None
    if args.baseline:
        bp = Path(args.baseline) / "metrics.json"
        if bp.exists():
            baseline = json.loads(bp.read_text(encoding="utf-8"))

    path, has_conclusion = generate_report(run, baseline=baseline)
    print(f"[리포트] {path} · 결론 {'작성됨' if has_conclusion else '⚠️ 미작성'}",
          flush=True)
    if not args.no_catalog:
        cat = update_catalog(run.parent)
        if cat:
            print(f"[카탈로그] {cat} 갱신", flush=True)
        rlog = build_research_log(run.parent)
        print(f"[RESEARCH_LOG] {rlog} 갱신", flush=True)


if __name__ == "__main__":
    main()
