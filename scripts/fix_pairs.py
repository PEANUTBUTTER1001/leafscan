"""labeled/source tar 짝 어긋남 진단 및 복구 — 엽채류 4종 공통.

배경
----
AI 허브 "수직농장 통합데이터 — 엽채류"는 라벨(JSON)과 이미지(JPG)가 각각
별도의 tar로 배포되며, 두 tar는 **끝 번호로 짝을 이룬다**.

    TL_4.케일12.tar  (라벨)  ↔  TS_4.케일12.tar  (이미지)

`scripts/prepare_dataset.py` 는 이 번호를 신뢰해 짝을 맺는다. 그런데 실제
배포본에는 **번호는 같은데 내용이 서로 다른** 아카이브가 존재한다. 이 경우
교집합이 0건이 되어 해당 묶음 전체(약 500장)가 조용히 버려진다.

실측 사례 (케일 training, 2026-08)
    · 15번 ↔ 16번이 서로 교차
    · 19~26번이 source 기준으로 한 칸씩 밀림
    · 59~64번이 역순으로 대응
    · 18번 라벨 tar 하나가 1,000건(두 아카이브 몫)을 보유
    → 16개 아카이브, 약 11,500장이 유실되고 있었음

해결 방식
--------
파일명(stem)은 라벨과 이미지가 **완전히 동일**하다. 따라서 아카이브 번호를
믿지 말고 stem으로 직접 대조하면 실제 짝을 알 수 있다. 이 스크립트는

  1) 번호 기준 매칭량과 stem 기준 매칭량을 비교해 **손실량을 진단**하고,
  2) `--apply` 시 한 split의 tar들을 **하나로 병합**해 번호 문제를 무력화한다.

병합 후에는 라벨 tar 1개 ↔ 이미지 tar 1개가 되므로 번호가 항상 일치하고,
인제스트의 stem 조인이 가능한 모든 짝을 찾아낸다.

주의
----
· 병합하면 index.csv 의 `archive` 컬럼이 단일 값으로 뭉뚱그려져
  아카이브 단위 추적성이 사라진다. **팀 전체가 같은 방식을 쓸지 합의할 것.**
· 병합은 tar를 새로 쓰므로 원본 용량만큼의 **여유 디스크**가 추가로 필요하다.
· 원본 tar는 삭제하지 않고 `data/_raw_backup/` 으로 이동한다.
· 애초에 파일이 누락된 경우(짝 자체가 없음)는 복구 불가 — AI 허브에서 재다운로드해야 한다.

사용법 (leafscan 저장소 루트에서 실행)
------------------------------------
    python scripts/fix_pairs.py                        # 전체 작물 진단만
    python scripts/fix_pairs.py --crop kale            # 케일만 진단
    python scripts/fix_pairs.py --crop kale --apply    # 케일 복구
    python scripts/fix_pairs.py --apply --yes          # 전체 복구 (확인 생략)

복구 후에는 반드시 인덱스를 다시 생성한다.

    rmdir /s /q data\\_extracted
    del data\\index.csv
    python scripts/prepare_dataset.py --data-root data --out data/index.csv
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tarfile

# --------------------------------------------------------------------------
# 상수
# --------------------------------------------------------------------------

DATA_ROOT = "data"
BACKUP_DIR = os.path.join(DATA_ROOT, "_raw_backup")
TMP_DIR = os.path.join(DATA_ROOT, "_merge_tmp")

# 폴더명(영문) → 표시용 한글명. JSON 의 crops 값과는 별개다.
CROPS: dict[str, str] = {
    "lettuce": "상추",
    "chard": "근대",
    "kale": "케일",
    "leaf_mustard": "겨자채",
}

SPLITS = ("training", "validation")

LABEL_EXT = ".json"
IMAGE_EXT = ".jpg"

# TL_4.케일12.tar → ('TL', '4', '케일', '12')
#   1: TL/TS/VL/VS   2: 데이터 그룹 번호   3: 작물 한글명   4: 묶음 번호
ARCHIVE_RE = re.compile(r"^([A-Z]{2})_(\d+)\.(.+?)(\d+)\.tar$")

# 진단 시 화면에 나열할 어긋난 묶음의 최대 개수
MAX_LISTED = 20


# --------------------------------------------------------------------------
# 공용 유틸
# --------------------------------------------------------------------------

def list_tars(directory: str) -> list[str]:
    """디렉터리 안의 .tar 파일명을 정렬해 반환. 없으면 빈 리스트."""
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(".tar"))


def read_stems(tar_path: str, ext: str) -> set[str]:
    """tar 안에서 해당 확장자를 가진 파일의 stem(확장자 제외 이름) 집합을 반환.

    stem 이 라벨↔이미지를 잇는 유일한 키다. JSON 내부의 `fname` 필드는
    실제 파일명이 아니므로 조인에 사용하지 않는다.
    """
    stems: set[str] = set()
    try:
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(ext):
                    stems.add(os.path.splitext(os.path.basename(member.name))[0])
    except (tarfile.TarError, OSError) as exc:
        print(f"      [!] {os.path.basename(tar_path)} 읽기 실패 — {exc}")
    return stems


def archive_number(filename: str) -> str | None:
    """파일명에서 묶음 번호를 추출. 규칙에 맞지 않으면 None."""
    matched = ARCHIVE_RE.match(filename)
    return matched.group(4) if matched else None


def merged_name(sample_filename: str) -> str | None:
    """원본 명명 규칙을 따르는 병합 tar 이름을 만든다.

    TL_4.케일12.tar 를 샘플로 받으면 TL_4.케일1.tar 를 돌려준다.
    """
    matched = ARCHIVE_RE.match(sample_filename)
    if not matched:
        return None
    prefix, group, crop_ko, _ = matched.groups()
    return f"{prefix}_{group}.{crop_ko}1.tar"


def human_gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.1f}GB"


def dir_size(directory: str) -> int:
    """디렉터리 안 tar 파일의 총 바이트 수."""
    return sum(os.path.getsize(os.path.join(directory, f))
               for f in list_tars(directory))


# --------------------------------------------------------------------------
# 진단
# --------------------------------------------------------------------------

def diagnose(crop: str, split: str) -> dict | None:
    """번호 기준 매칭과 stem 기준 매칭을 비교해 회복 가능량을 계산한다.

    Returns:
        {'by_number': int, 'by_stem': int, 'recoverable': int} 또는
        데이터가 없어 판단할 수 없으면 None.
    """
    base = os.path.join(DATA_ROOT, crop, split)
    label_dir = os.path.join(base, "labeled")
    image_dir = os.path.join(base, "source")

    label_tars = list_tars(label_dir)
    image_tars = list_tars(image_dir)
    if not label_tars and not image_tars:
        return None

    print(f"\n─── {CROPS.get(crop, crop)} / {split} "
          f"— labeled {len(label_tars)}개 · source {len(image_tars)}개")

    if not label_tars or not image_tars:
        print("      한쪽 폴더가 비어 있어 매칭할 수 없습니다. tar 배치를 확인하세요.")
        return None

    print("      tar 내용을 읽는 중… (용량에 따라 수 분 소요)")
    labels = {archive_number(f): read_stems(os.path.join(label_dir, f), LABEL_EXT)
              for f in label_tars}
    images = {archive_number(f): read_stems(os.path.join(image_dir, f), IMAGE_EXT)
              for f in image_tars}
    labels.pop(None, None)   # 명명 규칙에 맞지 않는 파일은 제외
    images.pop(None, None)

    if not labels or not images:
        print("      파일명 규칙(TL_4.작물NN.tar)에 맞는 tar가 없습니다.")
        return None

    # ① 현재 인제스트 방식 — 같은 번호끼리만 대조
    by_number = sum(len(stems & images.get(num, set()))
                    for num, stems in labels.items())

    # ② 번호를 무시하고 stem 전체로 대조
    all_labels: set[str] = set().union(*labels.values())
    all_images: set[str] = set().union(*images.values())
    by_stem = len(all_labels & all_images)

    recoverable = by_stem - by_number
    print(f"      번호 기준 매칭 : {by_number:>7,}장   ← 지금 학습에 쓰이는 양")
    print(f"      stem 기준 매칭 : {by_stem:>7,}장   ← 병합하면 쓸 수 있는 양")
    print(f"      회복 가능      : {recoverable:>7,}장")

    _report_broken(labels, images)

    return {"by_number": by_number, "by_stem": by_stem, "recoverable": recoverable}


def _report_broken(labels: dict[str, set[str]], images: dict[str, set[str]]) -> None:
    """번호는 같은데 교집합이 0인 묶음과, 그 묶음의 실제 짝을 출력한다."""
    broken: list[str] = []
    for num in sorted(labels, key=int):
        if labels[num] & images.get(num, set()):
            continue
        best_count, best_num = max(
            ((len(labels[num] & stems), other) for other, stems in images.items()),
            default=(0, None),
        )
        if best_num is None or best_count == 0:
            broken.append(f"        labeled {num}번 — 대응하는 이미지 없음 (복구 불가)")
        else:
            broken.append(f"        labeled {num}번 ↔ source {num}번 교집합 0 "
                          f"→ 실제 짝은 source {best_num}번 ({best_count:,}장)")

    if not broken:
        return
    print("      [번호가 어긋난 묶음]")
    print("\n".join(broken[:MAX_LISTED]))
    if len(broken) > MAX_LISTED:
        print(f"        … 외 {len(broken) - MAX_LISTED}건")


# --------------------------------------------------------------------------
# 병합
# --------------------------------------------------------------------------

def merge_side(src_dir: str, out_name: str, ext: str) -> tuple[str, int]:
    """한 폴더의 tar들을 하나로 병합한다. stem 중복은 첫 번째만 남긴다.

    중복 제거가 필요한 이유: 내용이 동일한 아카이브가 번호만 다르게
    배포된 경우가 있다(케일 26 == 27). 그대로 두면 같은 이미지가
    두 번 학습에 들어간다.

    Returns:
        (생성된 tar 경로, 담긴 파일 수)
    """
    sources = list_tars(src_dir)
    inner_dir = os.path.splitext(out_name)[0]   # tar 내부 폴더명
    out_path = os.path.join(TMP_DIR, out_name)
    seen: set[str] = set()

    with tarfile.open(out_path, "w") as out_tar:
        for idx, filename in enumerate(sources, 1):
            with tarfile.open(os.path.join(src_dir, filename)) as in_tar:
                for member in in_tar.getmembers():
                    if not member.isfile() or not member.name.lower().endswith(ext):
                        continue
                    stem = os.path.splitext(os.path.basename(member.name))[0]
                    if stem in seen:
                        continue
                    seen.add(stem)
                    # 내부 경로를 병합 tar 이름 기준으로 통일
                    member.name = f"{inner_dir}/{os.path.basename(member.name)}"
                    out_tar.addfile(member, in_tar.extractfile(member))
            print(f"        ({idx}/{len(sources)}) {filename} → 누적 {len(seen):,}개")

    return out_path, len(seen)


def apply_merge(crop: str, split: str) -> bool:
    """한 작물·split의 labeled/source tar를 각각 하나로 병합한다.

    Returns:
        실제로 병합했으면 True.
    """
    base = os.path.join(DATA_ROOT, crop, split)
    label_dir = os.path.join(base, "labeled")
    image_dir = os.path.join(base, "source")
    label_tars = list_tars(label_dir)
    image_tars = list_tars(image_dir)

    if len(label_tars) <= 1 and len(image_tars) <= 1:
        print("      이미 병합된 상태입니다 — 건너뜁니다.")
        return False

    label_out = merged_name(label_tars[0])
    image_out = merged_name(image_tars[0])
    if not label_out or not image_out:
        print(f"      [!] 파일명 규칙을 알 수 없어 건너뜁니다 "
              f"({label_tars[0]} / {image_tars[0]})")
        return False

    # 디스크 여유 확인 — 병합본을 새로 쓰므로 원본만큼 추가 공간이 필요하다
    need = dir_size(label_dir) + dir_size(image_dir)
    free = shutil.disk_usage(DATA_ROOT).free
    print(f"      필요 공간 약 {human_gb(need)} · 여유 {human_gb(free)}")
    if free < need * 1.1:
        print("      [!] 디스크 여유가 부족합니다. 공간을 확보한 뒤 다시 실행하세요.")
        return False

    os.makedirs(TMP_DIR, exist_ok=True)

    print(f"      [labeled] {len(label_tars)}개 병합")
    label_path, label_count = merge_side(label_dir, label_out, LABEL_EXT)
    print(f"      [source]  {len(image_tars)}개 병합")
    image_path, image_count = merge_side(image_dir, image_out, IMAGE_EXT)

    # 원본은 삭제하지 않고 백업으로 이동 (같은 드라이브면 즉시 완료)
    backup = os.path.join(BACKUP_DIR, crop, split)
    for kind, directory in (("labeled", label_dir), ("source", image_dir)):
        os.makedirs(os.path.join(backup, kind), exist_ok=True)
        for filename in list_tars(directory):
            shutil.move(os.path.join(directory, filename),
                        os.path.join(backup, kind, filename))

    shutil.move(label_path, os.path.join(label_dir, label_out))
    shutil.move(image_path, os.path.join(image_dir, image_out))
    try:
        os.rmdir(TMP_DIR)
    except OSError:
        pass

    print(f"      완료 — 라벨 {label_count:,}개 · 이미지 {image_count:,}개")
    print(f"      원본 tar는 {backup} 으로 이동했습니다.")
    return True


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="labeled/source tar 짝 어긋남 진단 및 복구 (엽채류 4종 공통)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="예시:\n"
               "  python scripts/fix_pairs.py\n"
               "  python scripts/fix_pairs.py --crop kale --apply\n",
    )
    parser.add_argument("--crop", default="all",
                        choices=[*CROPS, "all"],
                        help="대상 작물 (기본: all)")
    parser.add_argument("--split", default="all",
                        choices=[*SPLITS, "all"],
                        help="대상 분할 (기본: all)")
    parser.add_argument("--apply", action="store_true",
                        help="실제로 tar를 병합한다 (기본은 진단만)")
    parser.add_argument("--yes", action="store_true",
                        help="병합 전 확인 질문을 생략한다")
    return parser.parse_args()


def confirm(total_recoverable: int) -> bool:
    print(f"\n{total_recoverable:,}장을 회복할 수 있습니다. tar를 병합합니다.")
    print("원본은 삭제되지 않고 data/_raw_backup/ 으로 이동합니다.")
    try:
        return input("계속할까요? [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def main() -> None:
    args = parse_args()

    if not os.path.isdir(DATA_ROOT):
        print(f"{DATA_ROOT}/ 폴더가 없습니다. leafscan 저장소 루트에서 실행하세요.")
        sys.exit(1)

    crops = list(CROPS) if args.crop == "all" else [args.crop]
    splits = list(SPLITS) if args.split == "all" else [args.split]

    print("=" * 62)
    print("labeled/source tar 짝 진단" + ("  (--apply: 병합 수행)" if args.apply else "  (진단 전용 — 파일을 변경하지 않습니다)"))
    print("=" * 62)

    # 1단계 — 전체 진단
    results: list[tuple[str, str, dict]] = []
    for crop in crops:
        if not os.path.isdir(os.path.join(DATA_ROOT, crop)):
            continue
        for split in splits:
            result = diagnose(crop, split)
            if result:
                results.append((crop, split, result))

    if not results:
        print("\n대상 데이터를 찾지 못했습니다. data/{작물}/{split}/ 배치를 확인하세요.")
        return

    # 2단계 — 요약
    total_recoverable = sum(r["recoverable"] for _, _, r in results)
    print("\n" + "=" * 62)
    print(f"{'작물':<10}{'split':<12}{'현재':>10}{'병합 후':>10}{'회복':>10}")
    print("-" * 62)
    for crop, split, r in results:
        print(f"{CROPS.get(crop, crop):<10}{split:<12}"
              f"{r['by_number']:>10,}{r['by_stem']:>10,}{r['recoverable']:>10,}")
    print("-" * 62)
    print(f"{'합계':<22}{'':>10}{'':>10}{total_recoverable:>10,}")
    print("=" * 62)

    if not args.apply:
        if total_recoverable > 0:
            print("\n실제로 복구하려면 --apply 를 붙여 다시 실행하세요.")
        else:
            print("\n모든 짝이 정상입니다. 조치할 것이 없습니다.")
        return

    if total_recoverable <= 0:
        print("\n회복할 데이터가 없어 병합하지 않습니다.")
        return

    if not args.yes and not confirm(total_recoverable):
        print("취소했습니다.")
        return

    # 3단계 — 병합
    for crop, split, result in results:
        if result["recoverable"] <= 0:
            continue
        print(f"\n─── {CROPS.get(crop, crop)} / {split} 병합")
        apply_merge(crop, split)

    print("\n" + "=" * 62)
    print("병합 완료. 인덱스를 다시 생성하세요:")
    print("  rmdir /s /q data\\_extracted")
    print("  del data\\index.csv")
    print("  python scripts/prepare_dataset.py --data-root data --out data/index.csv")
    print("=" * 62)


if __name__ == "__main__":
    main()
