"""CLI worker for the immutable final test-set evaluation."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from core.test_evaluation import evaluate_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reevaluate", action="store_true")
    ap.add_argument("--reason", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()
    child_kw = {}
    if sys.platform == "win32":
        child_kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    evaluation = evaluate_run(args.run, allow_rerun=args.reevaluate,
                              reason=args.reason, device=args.device,
                              batch_size=args.batch_size)
    subprocess.run([sys.executable, str(Path(__file__).with_name("make_test_figures.py")),
                    "--evaluation", str(evaluation)], check=True, **child_kw)
    subprocess.run([sys.executable, str(Path(__file__).with_name("make_report.py")),
                    "--run", str(Path(args.run)), "--no-catalog"], check=True,
                    **child_kw)
    result = json.loads((evaluation / "test_metrics.json").read_text(encoding="utf-8"))
    print(json.dumps({"event": "final_test_done", "evaluation": str(evaluation),
                      "metrics": result["per_head"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"event": "error", "message": str(exc)}, ensure_ascii=False), flush=True)
        raise SystemExit(1)
