"""check_data.py — 데이터 버전·무결성 확인 (P4a).

팀원 간 index.csv 가 같은 데이터에서 나왔는지 해시로 대조한다.

    python scripts/check_data.py --index data/index.csv          # 해시 출력
    python scripts/check_data.py --index data/index.csv --expect sha256:abcd...  # 대조
"""
import argparse
import csv
import hashlib
from pathlib import Path


def index_hash(index_csv: Path) -> str:
    h = hashlib.sha256()
    with open(index_csv, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def summarize(index_csv: Path):
    import collections
    crops, stages, groups = collections.Counter(), collections.Counter(), set()
    n = 0
    with open(index_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            crops[r["crop"]] += 1
            stages[r["stage"]] += 1
            groups.add(r["group_id"])
            n += 1
    return n, dict(crops), dict(stages), len(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--expect", default=None, help="기대 해시와 대조")
    args = ap.parse_args()
    idx = Path(args.index)
    if not idx.exists():
        raise SystemExit(f"[중단] index 파일 없음: {idx}")
    h = index_hash(idx)
    n, crops, stages, groups = summarize(idx)
    print(f"index : {idx}")
    print(f"hash  : {h}")
    print(f"규모  : {n}장 · 개체 {groups}개")
    print(f"작물  : {crops}")
    print(f"단계  : {stages}")
    if args.expect:
        if h == args.expect:
            print("[일치] 기대 해시와 동일 — 같은 데이터 버전입니다.")
        else:
            print("[불일치] ⚠️ 데이터 버전이 다릅니다.")
            print(f"  기대: {args.expect}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
