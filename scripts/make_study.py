"""make_study.py — study 비교 그림 생성 (별도 프로세스) · P10b.

study 디렉터리의 member run metrics.json 을 모아 비교 그림 4종을 만든다.
run_study.py 가 만든 STUDY.md 와 함께 study 결론의 근거가 된다.

    python scripts/make_study.py --study runs/study_01_backbone
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.figures_study import make_study_figures


def collect_members(study_dir: Path):
    members = []
    for run in sorted(study_dir.glob("*/")):
        if run.name.startswith("_"):     # _lrsearch 등 보조 run 제외
            continue
        mp = run / "metrics.json"
        if mp.exists():
            try:
                members.append((run.name, json.loads(mp.read_text(encoding="utf-8"))))
            except Exception:  # noqa: BLE001
                pass
    return members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    args = ap.parse_args()
    study_dir = Path(args.study)
    members = collect_members(study_dir)
    if not members:
        raise SystemExit(f"[중단] {study_dir} 에 member run 이 없습니다.")
    results = make_study_figures(members, study_dir / "figures")
    ok = sum(1 for _, o, _ in results if o)
    for name, o, info in results:
        print(f"  {'✓' if o else '·'} {name}: {info}", flush=True)
    print(f"[study 그림] {ok}/{len(results)} · {study_dir/'figures'}", flush=True)


if __name__ == "__main__":
    main()
