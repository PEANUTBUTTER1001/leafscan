"""preprocess.py — 전처리 원스톱 파이프라인 (tar → index.csv → 이미지 캐시).

4개 작물(상추·케일·겨자채·근대) 모두 동일하게 처리한다. AI Hub tar 를
data/{crop}/{split}/{labeled|source}/ 에 넣기만 하면 된다 (폴더 구조는
scripts/setup_data_dirs.py 가 만든다).

  ① 작물별 tar 배치 현황 확인
  ② 인제스트 — 검증(전역 stem 조인, 판단 D-02) · 추출 · index.csv/labels.json/MANIFEST.md
     (scripts/prepare_dataset.py 의 build_index 를 그대로 사용)
  ③ 이미지 캐시 일괄 생성 — 원본(1500×2000급)을 256px 로 사전 리사이즈.
     학습 시 첫 epoch 의 캐시 생성 대기가 사라진다 (epoch당 ~12분 → ~26초).

    python scripts/preprocess.py                          # 전 작물, 전체 파이프라인
    python scripts/preprocess.py --crops 근대,상추         # 일부 작물만 index 에 포함
    python scripts/preprocess.py --no-cache               # 캐시 생성 생략
    python scripts/preprocess.py --skip-broken            # 매칭 안 되는 라벨 건너뛰고 진행
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.data import cache_image, load_index_rows          # noqa: E402
from prepare_dataset import build_index                     # noqa: E402
from tasks.leafscan import FOLDER_TO_CROP                   # noqa: E402


def report_tar_layout(data_root: Path):
    """작물별 tar 배치 현황을 출력하고, tar 가 있는 작물(한글) 목록을 반환."""
    print("① 작물별 tar 배치 현황")
    present = []
    for folder, crop_kr in FOLDER_TO_CROP.items():
        n = len(list((data_root / folder).rglob("*.tar"))) \
            if (data_root / folder).is_dir() else 0
        mark = "○" if n else "×"
        print(f"   {mark} {crop_kr:4s} data/{folder:13s} tar {n}개"
              + ("" if n else "  — 건너뜀 (tar 를 넣으면 자동 포함)"))
        if n:
            present.append(crop_kr)
    if not present:
        raise SystemExit("[중단] tar 가 하나도 없습니다. "
                         "scripts/setup_data_dirs.py 로 폴더를 만들고 tar 를 넣으세요.")
    return present


def _cache_worker(args):
    r, data_root, cache_dir, cache_size, crop_mode = args
    try:
        return cache_image(r, data_root, cache_dir, cache_size, crop_mode)
    except Exception as ex:  # noqa: BLE001
        return f"{r['path']}: {ex}"


def prebuild_cache(index_csv: Path, cache_size: int, crop_mode: str, workers: int):
    rows = load_index_rows(index_csv)
    data_root = index_csv.parent
    cache_dir = data_root / "_cache" / f"{crop_mode}_{cache_size}"
    print(f"③ 이미지 캐시 일괄 생성 — {len(rows)}장 → {cache_dir} "
          f"(worker {workers}개)")
    jobs = [(r, data_root, cache_dir, cache_size, crop_mode) for r in rows]
    made = skipped = 0
    errors = []
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_cache_worker, jobs,
                                                    chunksize=64), 1):
            if res is True:
                made += 1
            elif res is False:
                skipped += 1
            else:
                errors.append(res)
            if i % 5000 == 0:
                rate = i / max(time.time() - t0, 1e-9)
                print(f"   … {i}/{len(rows)} ({rate:.0f}장/초)")
    print(f"   완료 — 신규 {made}장 · 기존 {skipped}장 · 실패 {len(errors)}건 "
          f"· {time.time() - t0:.0f}초")
    for e in errors[:10]:
        print(f"   [경고] 캐시 실패: {e}")
    if len(errors) > 10:
        print(f"   [경고] …외 {len(errors) - 10}건")


def main():
    ap = argparse.ArgumentParser(description="LeafScan 전처리 원스톱 파이프라인")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="data/index.csv")
    ap.add_argument("--crops", default="",
                    help="쉼표구분 한글 작물명만 포함 (예: 근대,상추). 기본: tar 있는 전부")
    ap.add_argument("--skip-broken", action="store_true",
                    help="라벨↔이미지 매칭 이슈를 건너뛰고 매칭분만 진행")
    ap.add_argument("--no-verify-images", action="store_true",
                    help="인제스트 단계의 이미지 손상 검사 생략 (속도)")
    ap.add_argument("--no-cache", action="store_true", help="③ 캐시 생성 생략")
    ap.add_argument("--cache-size", type=int, default=256)
    ap.add_argument("--crop-mode", default="full",
                    choices=["full", "bbox", "bbox_expand"],
                    help="캐시에 적용할 crop_mode (학습 config 와 일치해야 함)")
    ap.add_argument("--workers", type=int, default=max(2, (mp.cpu_count() or 4) - 2))
    args = ap.parse_args()

    data_root, out_csv = Path(args.data_root), Path(args.out)
    present = report_tar_layout(data_root)
    crops_only = [c for c in args.crops.split(",") if c] or None
    if crops_only:
        missing = [c for c in crops_only if c not in present]
        if missing:
            raise SystemExit(f"[중단] --crops {missing} 는 tar 가 없습니다 "
                             f"(있는 작물: {present})")

    print()
    print("② 인제스트 (검증 · 추출 · index.csv)")
    build_index(data_root, out_csv, skip_broken=args.skip_broken,
                crops_only=crops_only, verify_images=not args.no_verify_images)

    if not args.no_cache:
        print()
        prebuild_cache(out_csv, args.cache_size, args.crop_mode, args.workers)

    print()
    print("[완료] 전처리 끝 — GUI 에서 바로 학습을 시작하면 됩니다.")


if __name__ == "__main__":
    main()
