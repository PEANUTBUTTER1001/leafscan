"""prepare_dataset.py — 데이터 인제스트 (tar → index.csv) · P4a.

05_데이터_명세서 §6 의 파이프라인:
  ① 아카이브 짝 검증 → ② 추출 → ③ 파싱·조인 → ④ 품질검사 → ⑤⑥⑦ 산출

    python scripts/prepare_dataset.py --data-root data --out data/index.csv

주의 (명세서의 함정):
  · 조인 키 = 파일명 stem (JSON 의 fname 은 실제 파일명이 아님)
  · crops_id 는 JSON 필드 사용 (파일명 파싱 금지 — 길이가 일정하지 않음)
  · width/height 는 문자열 → int 변환
  · 한 아카이브에 한 단계만 있을 수 있음, 개체 단위 시계열
  · (작물, split) 전체의 labeled↔source stem 교집합이 0 이면 데이터가 깨진 것

짝 검증 전역화 (판단 D-02): 같은 번호의 TL/TS tar 가 같은 내용물을 담는다는
  보장이 없다. 실측(겨자채 training)으로 TL_1.겨자채1 의 stem 들이
  TS_1.겨자채60/61 에 들어 있는 등, 라벨과 원본이 서로 다른 순서로 묶여
  배포된다. 따라서 번호별 1:1 짝 검증 대신 (작물, split) 단위로 모든 tar 의
  stem 을 모아 전역 조인한다. 라벨 없는 이미지는 경고 후 제외(labeled tar
  누락분), 이미지 없는 라벨은 hard 오류(--skip-broken 시 매칭분만 진행).

group_id 견고화 (판단 D-01): 실제 TL_42 는 stem 이 JSON crops_id 로 시작하지
  않는다(stem=C26_L01_07_..., crops_id=C26_L04_01). 이 경우에도 개체 단위 누수를
  막도록, image_id 를 제거한 body 를 개체 식별자로 사용한다.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks.leafscan import (CANON_CROPS, CANON_STAGES, FOLDER_TO_CROP,
                            order_present)

ARCHIVE_RE = re.compile(r"^(TL|TS|VL|VS)_[1234]\.(.+?)(\d+)\.tar$")
ROLE = {"TL": ("training", "labeled"), "TS": ("training", "source"),
        "VL": ("validation", "labeled"), "VS": ("validation", "source")}
INDEX_COLUMNS = ["path", "crop", "stage", "group_id", "archive", "split_source",
                 "kind_type", "width", "height", "captured_at", "image_id",
                 "bbox", "ambiguous"]
MIN_RES = 200  # px — 해상도 이상치 경고 기준


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# ① 아카이브 발견 · 짝 검증
# --------------------------------------------------------------------------
class Archive:
    def __init__(self, path: Path, crop_folder, split, role, number):
        self.path, self.crop_folder = path, crop_folder
        self.split, self.role, self.number = split, role, number
        self.stem = path.name[:-4]   # TL_3.상추42

    def __repr__(self):
        return f"<{self.stem}>"


def discover_archives(data_root: Path):
    """data/{crop}/{split}/{labeled|source}/*.tar 를 훑어 Archive 목록 반환."""
    archives = []
    for crop_folder in sorted(FOLDER_TO_CROP):
        base = data_root / crop_folder
        if not base.is_dir():
            continue
        for tar in base.rglob("*.tar"):
            m = ARCHIVE_RE.match(tar.name)
            if not m:
                log(f"[경고] 아카이브 명명 규칙 불일치, 건너뜀: {tar.name}")
                continue
            prefix, _crop_kr, number = m.group(1), m.group(2), m.group(3)
            split, role = ROLE[prefix]
            archives.append(Archive(tar, crop_folder, split, role, number))
    return archives


def _tar_stems(tar_path: Path, ext: str):
    with tarfile.open(tar_path) as t:
        return {Path(n).name[:-len(ext)] for n in t.getnames() if n.endswith(ext)}


def match_archives(archives):
    """(groups, errors) 반환. 전역 stem 조인 (판단 D-02).

    같은 번호의 TL/TS tar 가 같은 내용물을 담는다는 보장이 없으므로,
    (작물, split) 단위로 모든 labeled tar 의 json stem 과 모든 source tar 의
    jpg stem 을 모아 전역으로 조인한다.

    groups 원소: dict(crop, split, labs, srcs, lab_map, src_map, stems)
      · lab_map/src_map: stem -> Archive (어느 tar 에 들어 있는지)
      · stems: 조인된(라벨·이미지 모두 있는) stem 집합
    """
    grouped = defaultdict(lambda: {"labeled": [], "source": []})
    for a in archives:
        grouped[(a.crop_folder, a.split)][a.role].append(a)

    groups, errors = [], []
    for (crop, split), roles in sorted(grouped.items()):
        labs, srcs = roles["labeled"], roles["source"]
        if labs and not srcs:
            errors.append(dict(kind="missing_source", crop=crop, split=split,
                               detail=f"labeled tar {len(labs)}개에 대응하는 source tar 없음"))
            continue
        if srcs and not labs:
            errors.append(dict(kind="missing_labeled", crop=crop, split=split,
                               detail=f"source tar {len(srcs)}개에 대응하는 labeled tar 없음"))
            continue
        lab_map, src_map = {}, {}   # stem -> Archive
        for a in labs:
            for s in _tar_stems(a.path, ".json"):
                lab_map[s] = a
        for a in srcs:
            for s in _tar_stems(a.path, ".jpg"):
                src_map[s] = a
        matched = set(lab_map) & set(src_map)
        n_no_image = len(lab_map) - len(matched)
        n_unlabeled = len(src_map) - len(matched)
        if not matched:
            errors.append(dict(kind="zero_intersection", crop=crop, split=split,
                               detail=(f"라벨 {len(lab_map)}건 ↔ 이미지 {len(src_map)}건 "
                                       f"교집합 0건 — 라벨과 이미지가 전혀 매칭되지 "
                                       f"않습니다. 데이터를 다시 받으세요")))
            continue
        if n_no_image:
            errors.append(dict(kind="label_missing_image", crop=crop, split=split,
                               detail=(f"이미지 없는 라벨 {n_no_image}건 — source tar "
                                       f"누락 가능성. --skip-broken 으로 매칭된 "
                                       f"{len(matched)}건만 진행 가능")))
        if n_unlabeled:
            errors.append(dict(kind="unlabeled_image", crop=crop, split=split, soft=True,
                               detail=(f"라벨 없는 이미지 {n_unlabeled}건 제외 "
                                       f"(labeled tar 누락분으로 추정)")))
        log(f"[정보] {crop}/{split} — labeled tar {len(labs)}개(라벨 {len(lab_map)}건) "
            f"↔ source tar {len(srcs)}개(이미지 {len(src_map)}건) → 매칭 {len(matched)}건")
        groups.append(dict(crop=crop, split=split, labs=labs, srcs=srcs,
                           lab_map=lab_map, src_map=src_map, stems=matched))
    return groups, errors


# --------------------------------------------------------------------------
# ② 추출 (재실행 시 건너뜀)
# --------------------------------------------------------------------------
def extract_archive(archive: Archive, extract_root: Path):
    dest = extract_root / archive.crop_folder / archive.stem
    ext = ".json" if archive.role == "labeled" else ".jpg"
    if dest.is_dir():
        have = len(list(dest.rglob(f"*{ext}")))
        with tarfile.open(archive.path) as t:
            want = sum(1 for n in t.getnames() if n.endswith(ext))
        if have >= want and want > 0:
            return dest   # 이미 추출됨
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive.path) as t:
        for member in t.getmembers():
            if member.isfile() and member.name.endswith((".json", ".jpg")):
                member.name = Path(member.name).name   # 평탄화
                t.extract(member, dest, filter="data")
    return dest


# --------------------------------------------------------------------------
# ③ 파싱 · 조인 · group_id
# --------------------------------------------------------------------------
def derive_group_id(crops_id: str, stem: str):
    """group_id = {crops_id}_{개체번호}. 판단 D-01 견고화 적용.

    stem = ..._{image_id}. image_id 를 떼어낸 body 가 개체 식별자.
    stem 이 crops_id 로 시작하면 명세서 규칙대로 crops_id 를 벗겨 개체번호만 남기고,
    아니면(실측 TL_42) body 전체를 개체 식별자로 써서 누수만은 확실히 막는다.
    """
    if "_" not in stem:
        return f"{crops_id}_{stem}", stem
    body, _image_id = stem.rsplit("_", 1)
    if stem.startswith(crops_id + "_"):
        indiv = body[len(crops_id) + 1:]
    else:
        indiv = body   # 파일명 접두가 crops_id 와 다름 — body 전체를 개체키로
    return f"{crops_id}_{indiv}", indiv


def parse_labeled(json_path: Path):
    d = json.loads(json_path.read_text(encoding="utf-8"))
    im = d.get("images", {})
    bbox = ""
    anns = d.get("annotations") or []
    if anns and isinstance(anns[0], dict) and anns[0].get("bbox"):
        bbox = ",".join(str(x) for x in anns[0]["bbox"])

    def as_int(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 0
    return dict(
        crops=im.get("crops"),
        stage=im.get("growth_stage"),
        crops_id=im.get("crops_id", ""),
        kind_type=im.get("kind_type", ""),
        width=as_int(im.get("width")),
        height=as_int(im.get("height")),
        captured_at=im.get("date_captured", ""),
        image_id=im.get("image_id", ""),
        bbox=bbox,
    )


# --------------------------------------------------------------------------
# 메인 파이프라인
# --------------------------------------------------------------------------
def build_index(data_root: Path, out_csv: Path, skip_broken=False,
                crops_only=None, verify_images=True):
    extract_root = data_root / "_extracted"
    archives = discover_archives(data_root)
    if not archives:
        raise SystemExit(f"[중단] {data_root} 에서 아카이브(*.tar)를 찾지 못했습니다.")

    groups, errors = match_archives(archives)
    hard = [e for e in errors if not e.get("soft")]
    for e in errors:
        tag = "[경고]" if e.get("soft") else "[!!]"
        log(f"{tag} {e['crop']}/{e['split']} — {e['detail']}")

    if hard and not skip_broken:
        log("")
        log("[중단] 라벨↔이미지 매칭에 문제가 있습니다. 위 항목을 해결하거나")
        log("       --skip-broken 으로 매칭된 것만으로 진행하세요.")
        raise SystemExit(2)
    if hard and skip_broken:
        log(f"[진행] --skip-broken: 이슈 {len(hard)}건을 무시하고 "
            f"매칭된 stem 만으로 계속합니다.")
    if not groups:
        raise SystemExit("[중단] 사용 가능한(매칭되는) 아카이브가 없습니다.")

    rows = []
    excluded = Counter()
    manifest_rows = []
    for group in groups:
        crop_kr = FOLDER_TO_CROP[group["crop"]]
        if crops_only and crop_kr not in crops_only:
            continue
        lab_dirs = {a.stem: extract_archive(a, extract_root) for a in group["labs"]}
        src_dirs = {a.stem: extract_archive(a, extract_root) for a in group["srcs"]}
        per_archive = Counter()   # labeled tar 별 건수 (MANIFEST 용)
        for stem in sorted(group["stems"]):
            archive_name = group["lab_map"][stem].stem
            lab_dir = lab_dirs[archive_name]
            src_dir = src_dirs[group["src_map"][stem].stem]
            jpath = lab_dir / f"{stem}.json"
            ipath = src_dir / f"{stem}.jpg"
            if not jpath.exists():
                excluded["json_missing"] += 1
                continue
            if not ipath.exists():
                excluded["jpg_missing"] += 1
                continue
            try:
                info = parse_labeled(jpath)
            except Exception as ex:  # noqa: BLE001
                excluded["json_parse"] += 1
                log(f"[경고] JSON 파싱 실패 {stem}: {ex}")
                continue
            # 미지 라벨 → 중단 (라벨 체계 재확인 필요)
            if info["crops"] not in CANON_CROPS:
                raise SystemExit(f"[중단] 미지의 작물 라벨 {info['crops']!r} @ {stem} "
                                 f"(허용: {CANON_CROPS})")
            if info["stage"] not in CANON_STAGES:
                raise SystemExit(f"[중단] 미지의 생육단계 {info['stage']!r} @ {stem} "
                                 f"(허용: {CANON_STAGES})")
            # 폴더 ↔ crops 교차검증
            if info["crops"] != crop_kr:
                log(f"[경고] 폴더({crop_kr})와 JSON crops({info['crops']}) 불일치 @ {stem}")
            if verify_images:
                try:
                    from PIL import Image
                    with Image.open(ipath) as im:
                        w, h = im.size
                    if min(w, h) < MIN_RES:
                        excluded["small_res"] += 0  # 제외 아님, 경고만
                        log(f"[경고] 해상도 이상치 {w}x{h} @ {stem}")
                except Exception as ex:  # noqa: BLE001
                    excluded["image_corrupt"] += 1
                    log(f"[경고] 이미지 열기 실패 {stem}: {ex}")
                    continue
            gid, _indiv = derive_group_id(info["crops_id"], stem)
            rel = (src_dir / f"{stem}.jpg").resolve().relative_to(
                out_csv.parent.resolve())
            rows.append({
                "path": rel.as_posix(),
                "crop": info["crops"],
                "stage": info["stage"],
                "group_id": gid,
                "archive": archive_name,
                "split_source": group["split"],
                "kind_type": info["kind_type"],
                "width": info["width"],
                "height": info["height"],
                "captured_at": info["captured_at"],
                "image_id": info["image_id"],
                "bbox": info["bbox"],
                "ambiguous": False,
            })
            per_archive[archive_name] += 1
        for name in sorted(per_archive):
            manifest_rows.append((name, group["split"], per_archive[name]))

    if not rows:
        raise SystemExit("[중단] 조인 결과가 0건입니다. 짝·경로를 확인하세요.")

    # ⑤ index.csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # ⑥ labels.json  ⑦ MANIFEST.md
    write_labels(out_csv.parent / "labels.json", rows)
    write_manifest(out_csv.parent / "MANIFEST.md", out_csv, rows, manifest_rows,
                   excluded, errors)

    _summary(rows, excluded)
    return rows


def write_labels(path: Path, rows):
    crops = order_present({r["crop"] for r in rows}, CANON_CROPS)
    stages = order_present({r["stage"] for r in rows}, CANON_STAGES)
    labels = {
        "schema_version": 1,
        "heads": {"crop": crops, "stage": stages},
        "preprocess": {
            "input_size": 224,
            "color_space": "RGB",
            "resize_mode": "center_crop",
            "interpolation": "bilinear",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "value_range": "0-1",
        },
    }
    path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path: Path, index_csv: Path, rows, manifest_rows, excluded, errors):
    crops = Counter(r["crop"] for r in rows)
    stages = Counter(r["stage"] for r in rows)
    groups = len({r["group_id"] for r in rows})
    lines = [
        "# 데이터 MANIFEST", "",
        f"- 생성일: {datetime.now().isoformat(timespec='seconds')}",
        f"- index.csv: `{index_csv.name}` (sha256 `{_sha256(index_csv)[:16]}…`)",
        f"- 총 {len(rows)}장 · 개체(group) {groups}개",
        f"- 작물 분포: {dict(crops)}",
        f"- 단계 분포: {dict(stages)}", "",
        "## 아카이브별 건수", "",
        "| 아카이브 | split | 건수 |", "|---|---|---|",
    ]
    for name, split, n in manifest_rows:
        lines.append(f"| {name} | {split} | {n} |")
    lines += ["", "## 제외/경고 건수", ""]
    lines.append(f"- 제외: {dict(excluded) if excluded else '없음'}")
    if errors:
        lines.append("- 매칭 검증 이슈:")
        for e in errors:
            lines.append(f"  - [{'경고' if e.get('soft') else '오류'}] "
                         f"{e['crop']}/{e['split']}: {e['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(rows, excluded):
    log("")
    log(f"[완료] index.csv 생성 — {len(rows)}장")
    log(f"       작물: {dict(Counter(r['crop'] for r in rows))}")
    log(f"       단계: {dict(Counter(r['stage'] for r in rows))}")
    log(f"       개체(group): {len({r['group_id'] for r in rows})}개")
    if excluded:
        log(f"       제외: {dict(excluded)}")


def main():
    ap = argparse.ArgumentParser(description="LeafScan 데이터 인제스트")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="data/index.csv")
    ap.add_argument("--skip-broken", action="store_true",
                    help="짝이 깨진 아카이브를 건너뛰고 나머지로 index 생성")
    ap.add_argument("--crops", default="", help="쉼표구분 한글 작물명만 포함 (예: 상추)")
    ap.add_argument("--no-verify-images", action="store_true",
                    help="이미지 손상 검사 생략 (대용량에서 속도)")
    args = ap.parse_args()
    crops_only = [c for c in args.crops.split(",") if c] or None
    build_index(Path(args.data_root), Path(args.out),
                skip_broken=args.skip_broken, crops_only=crops_only,
                verify_images=not args.no_verify_images)


if __name__ == "__main__":
    main()
