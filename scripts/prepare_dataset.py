"""prepare_dataset.py — 데이터 인제스트 (tar → index.csv) · P4a.

04_데이터_명세서 §6 의 파이프라인:
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
  번호별 1:1 진단이 필요하면 validate_pairs·number_gaps 를 쓴다(진단·테스트용).

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
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks.leafscan import (CANON_CROPS, CANON_STAGES, FOLDER_TO_CROP,
                            order_present)

# 로그 문구에 '—' 같은 비ASCII 가 있어 출력을 파일·파이프로 넘기면
# 시스템 코드페이지(cp949)로 떨어져 UnicodeEncodeError 로 죽는다.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):   # 이미 닫혔거나 재설정 불가
            pass

# AI Hub archive names in this project are supplied in both v2 and v3 forms,
# training/validation or labeled/source role, which is determined by TL/TS/VL/VS.
ARCHIVE_RE = re.compile(r"^(TL|TS|VL|VS)_[1234]\.(.+?)(\d+)\.tar$")
ROLE = {"TL": ("training", "labeled"), "TS": ("training", "source"),
        "VL": ("validation", "labeled"), "VS": ("validation", "source")}
INDEX_COLUMNS = ["path", "crop", "stage", "group_id", "archive", "split_source",
                 "kind_type", "width", "height", "captured_at", "image_id",
                 "bbox", "ambiguous"]
MIN_RES = 200  # px — 해상도 이상치 경고 기준


# --------------------------------------------------------------------------
# 진행 표시 — 터미널이면 한 줄 갱신, 파일·파이프면 주기적 새 줄
# --------------------------------------------------------------------------
SPINNER = "-\\|/"
TOTAL_STEPS = 5


def fmt_hms(sec):
    sec = max(0, int(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def fmt_eta(elapsed, done, total):
    """남은 예상 시간. 추정할 수 없으면 빈 문자열."""
    if not total or done <= 0 or done >= total or elapsed <= 0:
        return ""
    return fmt_hms(elapsed / done * (total - done))


class Reporter:
    """단계 헤더·진행률·경고를 한 곳에서 찍는다.

    같은 스트림에 진행 줄(\\r)과 경고가 섞이면 화면이 깨지므로,
    영구 줄을 찍기 전에 진행 줄을 먼저 지운다.

    TTY(터미널)  : \\r 로 제자리 덮어쓰기 + 스피너, 0.5초 주기
    비TTY(파일)  : \\r 없이 새 줄, 10초 주기 — \\r 은 파일에서 줄을 바꾸지 않아
                   수천 번의 갱신이 한 줄에 뭉치면 로그를 못 쓰게 된다.

    stream·clock 을 주입할 수 있어 테스트에서 결정론적으로 검증한다.
    """

    def __init__(self, stream=None, clock=None, interval=0.5,
                 quiet=False, spinner=True, tty=None):
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock or time.monotonic
        self.interval = interval
        self.quiet = quiet
        if tty is None:
            try:
                tty = bool(self.stream.isatty())
            except (AttributeError, ValueError):
                tty = False
        self.tty = tty
        self.spinner = bool(spinner) and tty
        self.started = self.clock()
        self.phase_start = self.started
        self._live = ""
        self._last = None
        self._tick = 0

    # -- 기본 출력 ---------------------------------------------------
    def _write(self, s):
        self.stream.write(s)
        try:
            self.stream.flush()
        except (AttributeError, ValueError):
            pass

    def _clear_live(self):
        if self._live:
            if self.tty:
                self._write("\r" + " " * len(self._live) + "\r")
            self._live = ""

    def line(self, msg=""):
        """경고·정보 등 계속 남는 줄."""
        self._clear_live()
        self._write(msg + "\n")

    def step(self, n, title):
        self.phase_start = self.clock()
        self._last = None
        self.line("")
        self.line(f"[단계 {n}/{TOTAL_STEPS}] {title}")

    def info(self, msg):
        self.line(f"[정보] {msg}")

    def note(self, msg):
        self.line(f"[진행] {msg}")

    # -- 진행률 ------------------------------------------------------
    def _due(self, done, total, now):
        """스로틀 — 처음·마지막은 항상, 그 사이는 주기마다."""
        if self._last is None:
            return True
        if total and done >= total:
            return True
        gap = self.interval if self.tty else max(self.interval * 20, 10.0)
        return (now - self._last) >= gap

    def progress(self, done, total, label, extra="", eta=True):
        if self.quiet:
            return
        now = self.clock()
        if not self._due(done, total, now):
            return
        self._last = now
        pct = (done / total * 100) if total else 0.0
        body = f"[진행] {label} — {done:,} / {total:,}건 ({pct:.1f}%)"
        if extra:
            body += f" · {extra}"
        if eta:
            elapsed = now - self.phase_start
            remain = fmt_eta(elapsed, done, total)
            if remain:
                body += f" · 경과 {fmt_hms(elapsed)} · 남은 예상 {remain}"
        self._draw(body)

    def spin(self, label):
        """총량을 모르는 작업 — 살아 있다는 것만 보여준다."""
        if self.quiet:
            return
        now = self.clock()
        if not self._due(1, 0, now):
            return
        self._last = now
        self._draw(f"[진행] {label} ...")

    def _draw(self, body):
        if not self.tty:
            self._write(body + "\n")
            return
        if self.spinner:
            self._tick += 1
            body += "  " + SPINNER[self._tick % len(SPINNER)]
        pad = max(0, len(self._live) - len(body))
        self._write("\r" + body + " " * pad)
        self._live = body

    def elapsed(self):
        return self.clock() - self.started


# 활성 Reporter — 기존 log() 호출이 전부 진행 줄을 지우고 찍게 한다
_REPORT = None


def log(msg):
    if _REPORT is not None:
        _REPORT.line(msg)
    else:
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
    """data/{crop}/{split}/{labeled|source}/*.tar 를 훑어 (archives, skipped) 반환.

    skipped 는 명명 규칙에 맞지 않아 제외한 파일명. 한쪽만 제외되면 뒤에서
    '짝 없음' 오류로 번지므로, 흩어 찍지 않고 모아서 요약에 낸다.
    """
    archives, skipped = [], []
    for crop_folder in sorted(FOLDER_TO_CROP):
        base = data_root / crop_folder
        if not base.is_dir():
            continue
        for tar in base.rglob("*.tar"):
            m = ARCHIVE_RE.match(tar.name)
            if not m:
                skipped.append(tar.name)
                continue
            prefix, _crop_kr, number = m.group(1), m.group(2), m.group(3)
            split, role = ROLE[prefix]
            archives.append(Archive(tar, crop_folder, split, role, number))
    return archives, skipped


def _numbers_by_split(archives):
    """{(crop, split): {'labeled': {번호…}, 'source': {번호…}}} — 문자열 번호."""
    present = defaultdict(lambda: {"labeled": set(), "source": set()})
    for a in archives:
        present[(a.crop_folder, a.split)][a.role].add(a.number)
    return present


def number_gaps(archives):
    """{(crop, split): [빠진 번호…]} — 양쪽 다 없는 번호만 '공백'으로 본다.

    원본 아카이브 번호는 연속이 아닐 수 있다. 1,2,3,6,7,8 처럼 중간이 비는 것은
    데이터 불량이 아니라 번호 공백일 뿐이므로 오류로 취급하지 않는다.
    한쪽 role 에만 있는 번호는 공백이 아니라 결손이며 validate_pairs 가 오류로 낸다.
    """
    out = {}
    for key, roles in _numbers_by_split(archives).items():
        nums = set()
        for n in roles["labeled"] | roles["source"]:
            try:
                nums.add(int(n))
            except ValueError:
                continue
        if len(nums) < 2:
            out[key] = []
            continue
        out[key] = sorted(set(range(min(nums), max(nums) + 1)) - nums)
    return out


def _tar_stems(tar_path: Path, ext: str):
    with tarfile.open(tar_path) as t:
        return {Path(n).name[:-len(ext)] for n in t.getnames() if n.endswith(ext)}


def validate_pairs(archives, report=None):
    """(pairs, errors) 반환. 번호별 1:1 짝 진단 (진단·테스트용).

    본 파이프라인(build_index)은 전역 조인(match_archives, 판단 D-02)을 쓴다.
    이 함수는 번호별 결손·교집합을 사람이 확인할 때와 테스트에서 쓴다.

    검사 기준은 **실제 존재하는 tar 번호**다. 번호가 연속인지는 보지 않는다.
    오류로 보는 경우는 셋뿐이다.
      · 어떤 번호에 labeled 만 있고 source 가 없음
      · 어떤 번호에 source 만 있고 labeled 가 없음
      · 같은 번호의 TL/TS 내부 stem 교집합이 0건
    """
    grouped = defaultdict(dict)   # (crop, split, number) -> {role: Archive}
    for a in archives:
        grouped[(a.crop_folder, a.split, a.number)][a.role] = a
    present = _numbers_by_split(archives)

    def _compo(crop, split):
        """오해 방지용 — 이 split 에 실제로 있는 번호 구성."""
        def fmt(s):
            return "[" + ", ".join(sorted(s, key=lambda x: (len(x), x))) + "]"
        roles = present[(crop, split)]
        return f"labeled: {fmt(roles['labeled'])} · source: {fmt(roles['source'])}"

    GAP_HINT = ("번호 공백이 아니라 한쪽 tar 결손입니다. "
                "해당 번호의 {} tar 를 받으세요")

    pairs, errors = [], []
    items = sorted(grouped.items())
    for i, ((crop, split, number), roles) in enumerate(items, 1):
        lab, src = roles.get("labeled"), roles.get("source")
        if report is not None:
            report.progress(i, len(items), "짝 검증",
                            extra=(lab or src).stem if (lab or src) else "")
        if lab and not src:
            errors.append(dict(kind="missing_source", crop=crop, split=split,
                               number=number,
                               detail=(f"labeled {lab.stem} 는 있는데 같은 번호의 "
                                       f"source tar 가 없습니다"),
                               hint=GAP_HINT.format("source"),
                               compo=_compo(crop, split)))
            continue
        if src and not lab:
            errors.append(dict(kind="missing_labeled", crop=crop, split=split,
                               number=number,
                               detail=(f"source {src.stem} 는 있는데 같은 번호의 "
                                       f"labeled tar 가 없습니다"),
                               hint=GAP_HINT.format("labeled"),
                               compo=_compo(crop, split)))
            continue
        # 짝은 있음 — stem 교집합 검사
        lab_stems = _tar_stems(lab.path, ".json")
        src_stems = _tar_stems(src.path, ".jpg")
        inter = lab_stems & src_stems
        ratio = len(inter) / max(1, min(len(lab_stems), len(src_stems)))
        if len(inter) == 0:
            errors.append(dict(kind="zero_intersection", crop=crop, split=split,
                               number=number,
                               detail=(f"{lab.stem}({len(lab_stems)}건) ↔ "
                                       f"{src.stem}({len(src_stems)}건) 교집합 0건 "
                                       f"— 짝이 실제로 맞지 않습니다. 올바른 번호의 tar 를 받으세요")))
            pairs.append(dict(crop=crop, split=split, number=number,
                              labeled=lab, source=src, stems=inter))
            continue
        if ratio < 0.95:
            errors.append(dict(kind="low_intersection", crop=crop, split=split,
                               number=number, soft=True,
                               detail=(f"{lab.stem} ↔ {src.stem} 교집합 {ratio*100:.1f}% "
                                       f"({len(inter)}건) — 95% 미만")))
        pairs.append(dict(crop=crop, split=split, number=number,
                          labeled=lab, source=src, stems=inter))
    return pairs, errors


def match_archives(archives, report=None):
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

    n_tars = sum(len(r["labeled"]) + len(r["source"]) for r in grouped.values())
    t_i = 0
    groups, errors = [], []
    for (crop, split), roles in sorted(grouped.items()):
        labs, srcs = roles["labeled"], roles["source"]
        if labs and not srcs:
            errors.append(dict(kind="missing_source", crop=crop, split=split,
                               detail=f"labeled tar {len(labs)}개에 대응하는 source tar 없음"))
            t_i += len(labs)
            continue
        if srcs and not labs:
            errors.append(dict(kind="missing_labeled", crop=crop, split=split,
                               detail=f"source tar {len(srcs)}개에 대응하는 labeled tar 없음"))
            t_i += len(srcs)
            continue
        lab_map, src_map = {}, {}   # stem -> Archive
        for a in labs:
            t_i += 1
            if report is not None:
                report.progress(t_i, n_tars, "stem 수집", extra=a.stem)
            for s in _tar_stems(a.path, ".json"):
                lab_map[s] = a
        for a in srcs:
            t_i += 1
            if report is not None:
                report.progress(t_i, n_tars, "stem 수집", extra=a.stem)
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
def extract_archive(archive: Archive, extract_root: Path, report=None, head=""):
    dest = extract_root / archive.crop_folder / archive.stem
    ext = ".json" if archive.role == "labeled" else ".jpg"
    tag = f"{head} {archive.stem}".strip()
    if dest.is_dir():
        if report is not None:
            report.spin(f"{tag} 추출본 확인 중")
        have = len(list(dest.rglob(f"*{ext}")))
        with tarfile.open(archive.path) as t:
            want = sum(1 for n in t.getnames() if n.endswith(ext))
        if have >= want and want > 0:
            return dest   # 이미 추출됨
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive.path) as t:
        if report is not None:
            report.spin(f"{tag} 아카이브 목록 읽는 중")
        members = [m for m in t.getmembers()
                   if m.isfile() and m.name.endswith((".json", ".jpg"))]
        for i, member in enumerate(members, 1):
            member.name = Path(member.name).name   # 평탄화
            t.extract(member, dest, filter="data")
            if report is not None:
                report.progress(i, len(members), f"{tag} 추출 중")
    return dest


def reconcile_extracted_pair(lab_dir: Path, src_dir: Path):
    """추출된 JSON/JPG를 stem으로 맞춘다 (번호별 1:1 짝 전용 — 진단·테스트용).

    양쪽에 있는 파일만 반환한다. 한쪽에만 있는 추출본 파일은 삭제한다.
    원본 tar는 건드리지 않으며, 삭제 결과는 호출자가 manifest에 기록한다.

    주의: 전역 조인(판단 D-02) 파이프라인에서는 stem 의 JSON 과 JPG 가 서로
    다른 번호의 tar 에 들어 있을 수 있으므로 이 함수를 쓰지 않는다.
    """
    json_paths = {p.stem: p for p in lab_dir.rglob("*.json")}
    jpg_paths = {p.stem: p for p in src_dir.rglob("*.jpg")}
    common = set(json_paths) & set(jpg_paths)
    json_only = set(json_paths) - common
    jpg_only = set(jpg_paths) - common

    for stem in json_only:
        json_paths[stem].unlink()
    for stem in jpg_only:
        jpg_paths[stem].unlink()

    return common, len(json_only), len(jpg_only)


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
                crops_only=None, verify_images=True, report=None):
    global _REPORT
    if report is None:
        report = Reporter()
    _REPORT = report          # 기존 log() 호출이 진행 줄을 지우고 찍도록

    extract_root = data_root / "_extracted"

    # ------------------------------------------------ 단계 1: 아카이브 탐색
    report.step(1, "아카이브 탐색")
    archives, skipped = discover_archives(data_root)
    if not archives:
        raise SystemExit(f"[중단] {data_root} 에서 아카이브(*.tar)를 찾지 못했습니다.")
    n_lab = sum(1 for a in archives if a.role == "labeled")
    report.note(f"아카이브 {len(archives)}개 발견 "
                f"(labeled {n_lab} · source {len(archives) - n_lab})")
    if skipped:
        report.line(f"[경고] 명명 규칙 불일치로 제외 {len(skipped)}개: "
                    f"{', '.join(sorted(skipped)[:5])}"
                    f"{' …' if len(skipped) > 5 else ''}")
        report.line("       한쪽만 제외되면 아래에서 '짝 없음' 오류로 이어집니다.")

    # 번호 공백은 오류가 아니라 정보다 (원본 번호는 연속이 아닐 수 있다)
    gaps = number_gaps(archives)
    n_gap = 0
    for (crop, split), missing in sorted(gaps.items()):
        if not missing:
            continue
        n_gap += len(missing)
        report.info(f"{crop}/{split} — 아카이브 번호 공백 감지: "
                    f"{', '.join(str(n) for n in missing)}")
        report.line("       전역 stem 조인으로 계속 진행합니다.")

    # ------------------------- 단계 2: labeled/source 전역 매칭 (판단 D-02)
    report.step(2, "labeled/source 전역 매칭 검증")
    groups, errors = match_archives(archives, report)
    hard = [e for e in errors if not e.get("soft")]
    soft = [e for e in errors if e.get("soft")]
    report.note(f"매칭 검증 완료 — 사용 가능 그룹 {len(groups)}개 · "
                f"번호 공백 {n_gap} · 경고 {len(soft)} · 오류 {len(hard)}")
    for e in errors:
        tag = "[경고]" if e.get("soft") else "[!!]"
        num = f" #{e['number']}" if e.get("number") else ""
        log(f"{tag} {e['crop']}/{e['split']}{num} — {e['detail']}")
        if e.get("hint"):
            log(f"     ({e['hint']})")
        if e.get("compo"):
            log(f"     이 split 의 번호 구성 — {e['compo']}")

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

    # ------------------------------------------- 단계 3: 그룹별 처리
    report.step(3, "그룹별 처리 (추출 → 파싱"
                   + (" → 이미지 검사)" if verify_images else " · 이미지 검사 생략)"))
    rows = []
    excluded = Counter()
    manifest_rows = []
    todo = [g for g in groups
            if not (crops_only and FOLDER_TO_CROP[g["crop"]] not in crops_only)]
    for g_i, group in enumerate(todo, 1):
        crop_kr = FOLDER_TO_CROP[group["crop"]]
        head = f"({g_i}/{len(todo)}) {group['crop']}/{group['split']}"
        pos = f"({g_i}/{len(todo)})"
        try:
            lab_dirs = {a.stem: extract_archive(a, extract_root, report, pos)
                        for a in group["labs"]}
            src_dirs = {a.stem: extract_archive(a, extract_root, report, pos)
                        for a in group["srcs"]}
        except Exception as ex:                                # noqa: BLE001
            report.line(f"[!!] {head} / 추출 단계에서 실패: "
                        f"{type(ex).__name__}: {ex}")
            report.line(f"     진행 상황: 그룹 {g_i}/{len(todo)} · "
                        f"누적 등록 {len(rows):,}건")
            raise
        n_added = 0
        n_excluded = 0
        per_archive = Counter()   # labeled tar 별 건수 (MANIFEST 용)
        stem_list = sorted(group["stems"])
        report.phase_start = report.clock()
        report._last = None
        for s_i, stem in enumerate(stem_list, 1):
            report.progress(s_i, len(stem_list), f"{head} 처리 중",
                            extra=f"등록 {n_added:,}건 · 제외 {n_excluded:,}건")
            archive_name = group["lab_map"][stem].stem
            lab_dir = lab_dirs[archive_name]
            src_dir = src_dirs[group["src_map"][stem].stem]
            jpath = lab_dir / f"{stem}.json"
            ipath = src_dir / f"{stem}.jpg"
            if not jpath.exists():
                excluded["json_missing"] += 1
                n_excluded += 1
                continue
            if not ipath.exists():
                excluded["jpg_missing"] += 1
                n_excluded += 1
                continue
            try:
                info = parse_labeled(jpath)
            except Exception as ex:  # noqa: BLE001
                excluded["json_parse"] += 1
                n_excluded += 1
                log(f"[경고] JSON 파싱 실패 {stem}: {ex}")
                continue
            # 미지 라벨 → 중단 (라벨 체계 재확인 필요)
            where = (f"@ {archive_name} / 파싱 단계 / stem={stem}\n"
                     f"       진행 상황: 그룹 {g_i}/{len(todo)} · "
                     f"이 그룹 {s_i}/{len(stem_list)}건 · "
                     f"누적 등록 {len(rows):,}건")
            if info["crops"] not in CANON_CROPS:
                raise SystemExit(f"[중단] 미지의 작물 라벨 {info['crops']!r} {where}\n"
                                 f"       (허용: {CANON_CROPS})")
            if info["stage"] not in CANON_STAGES:
                raise SystemExit(f"[중단] 미지의 생육단계 {info['stage']!r} {where}\n"
                                 f"       (허용: {CANON_STAGES})")
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
                    n_excluded += 1
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
            n_added += 1
        report.note(f"{head} 처리 완료 — {n_added:,}건 등록"
                    + (f" · 제외 {n_excluded:,}건" if n_excluded else "")
                    + f" · 누적 {len(rows):,}건")
        for name in sorted(per_archive):
            manifest_rows.append((name, group["split"], per_archive[name]))

    if not rows:
        raise SystemExit("[중단] 조인 결과가 0건입니다. 짝·경로를 확인하세요.")

    # ------------------------------------------------ 단계 4: index.csv
    report.step(4, "index.csv 저장")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    report.note(f"index.csv 저장 완료 — {len(rows):,}행 · {out_csv}")

    # ----------------------------------- 단계 5: labels.json · MANIFEST.md
    report.step(5, "labels.json · MANIFEST.md 생성")
    write_labels(out_csv.parent / "labels.json", rows)
    report.spin("MANIFEST.md 해시 계산 중")
    write_manifest(out_csv.parent / "MANIFEST.md", out_csv, rows, manifest_rows,
                   excluded, errors)
    report.note(f"labels.json · MANIFEST.md 생성 완료")

    _summary(rows, excluded, report)
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
            num = f" #{e['number']}" if e.get("number") else ""
            lines.append(f"  - [{'경고' if e.get('soft') else '오류'}] "
                         f"{e['crop']}/{e['split']}{num}: {e['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(rows, excluded, report=None):
    n_excluded = sum(v for k, v in excluded.items()) if excluded else 0
    took = f" · 총 소요 {fmt_hms(report.elapsed())}" if report is not None else ""
    log("")
    log(f"[완료] index.csv 생성 — {len(rows):,}장 · 제외 {n_excluded:,}장{took}")
    log(f"       작물: {dict(Counter(r['crop'] for r in rows))}")
    log(f"       단계: {dict(Counter(r['stage'] for r in rows))}")
    log(f"       개체(group): {len({r['group_id'] for r in rows}):,}개")
    if excluded:
        log(f"       제외 내역: {dict(excluded)}")


def main():
    ap = argparse.ArgumentParser(description="LeafScan 데이터 인제스트")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="data/index.csv")
    ap.add_argument("--skip-broken", action="store_true",
                    help="짝이 깨진 아카이브를 건너뛰고 나머지로 index 생성")
    ap.add_argument("--crops", default="", help="쉼표구분 한글 작물명만 포함 (예: 상추)")
    ap.add_argument("--no-verify-images", action="store_true",
                    help="이미지 손상 검사 생략 (대용량에서 속도)")
    ap.add_argument("--quiet", action="store_true",
                    help="진행률 표시 끄기 (단계 헤더와 최종 요약만)")
    ap.add_argument("--progress-interval", type=float, default=0.5,
                    help="진행률 갱신 주기(초). 기본 0.5")
    ap.add_argument("--no-spinner", action="store_true",
                    help="스피너 끄기 (터미널이 아니면 자동으로 꺼짐)")
    args = ap.parse_args()
    crops_only = [c for c in args.crops.split(",") if c] or None
    report = Reporter(interval=max(0.0, args.progress_interval),
                      quiet=args.quiet, spinner=not args.no_spinner)
    build_index(Path(args.data_root), Path(args.out),
                skip_broken=args.skip_broken, crops_only=crops_only,
                verify_images=not args.no_verify_images, report=report)


if __name__ == "__main__":
    main()
