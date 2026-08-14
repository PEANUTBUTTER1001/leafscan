"""setup_data_dirs.py — 최초 설치 시 data/ 뼈대 폴더 생성.

`data/` 는 `.gitignore` 대상이라 저장소를 새로 clone하면 아예 존재하지 않는다.
**다른 무엇보다 먼저 이 스크립트를 실행해** 작물별 표준 폴더 구조를 만든다.
그 다음 각자 AI 허브에서 받은 tar 를 정해진 위치에 넣기만 하면 된다
(04_팀_협업_규약.md §4 — 원본 데이터는 공유·재배포하지 않고 각자 직접 받는다).

    python scripts/setup_data_dirs.py
    python scripts/setup_data_dirs.py --data-root D:/leafscan_data   # 디스크 분리 시
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks.leafscan import FOLDER_TO_CROP

SPLITS = ["training", "validation"]
ROLES = ["labeled", "source"]
ARCHIVE_PREFIX = {("training", "labeled"): "TL", ("training", "source"): "TS",
                  ("validation", "labeled"): "VL", ("validation", "source"): "VS"}


def setup(data_root="data"):
    root = Path(data_root)
    created = []
    for crop_folder in FOLDER_TO_CROP:
        for split in SPLITS:
            for role in ROLES:
                d = root / crop_folder / split / role
                if not d.exists():
                    d.mkdir(parents=True, exist_ok=True)
                    created.append(d)
    extracted = root / "_extracted"
    if not extracted.exists():
        extracted.mkdir(parents=True, exist_ok=True)
        created.append(extracted)

    print(f"[완료] data/ 뼈대 폴더 준비 — 신규 {len(created)}개, 루트: {root.resolve()}")
    print()
    print("다음 위치에 원본 tar 를 그대로 넣으세요 (파일명 변경·압축 해제 금지):")
    for crop_folder, crop_kr in FOLDER_TO_CROP.items():
        print(f"  [{crop_folder}] ({crop_kr})")
        for split in SPLITS:
            for role in ROLES:
                prefix = ARCHIVE_PREFIX[(split, role)]
                print(f"    data/{crop_folder}/{split}/{role}/   "
                     f"← {prefix}_3.{crop_kr}NN.tar")
    print()
    print("지금은 data/lettuce/ 만 채워도 된다 — 나머지 3작물은 준비되는 대로 넣으면 된다.")
    print("배치 후 (전처리 원스톱 — 검증·추출·index·이미지캐시):")
    print("  python scripts/preprocess.py")
    return created


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data",
                    help="디스크 여유가 부족하면 다른 드라이브 경로를 지정 (예: D:/leafscan_data)")
    args = ap.parse_args()
    setup(args.data_root)


if __name__ == "__main__":
    main()
