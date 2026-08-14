"""core/data_summary.py — index.csv 분포 요약 (학습 전 점검용).

**tkinter 에 의존하지 않는다.** GUI 의 `분포 보기` 버튼과 터미널이 같은 함수를 쓰고,
테스트는 이 모듈만 import 한다 (`10_인수인계.md` §7-2 의 D-7 재발 방지 규칙).

학습 전에는 `split_report` 가 없으므로 `core/figures.data_distribution()` 을 쓸 수 없다.
여기서는 `index.csv` 만 읽어 같은 판단 근거(조합별 장수·개체 수·단계 비율)를 만든다.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

# 경고 임계값 — 이 값 미만이면 "학습·평가가 불안정하다" 고 알린다.
MIN_COMBO_ROWS = 100      # 작물×단계 조합당 최소 장수
MIN_GROUPS = 30           # 최소 개체(group_id) 수
MIN_STAGE_RATIO = 0.05    # 단계 하나가 전체의 5% 미만이면 희소


def summarize_index(index_csv):
    """index.csv 를 읽어 분포 요약 dict 를 만든다.

    반환 키: n, crops, stages, combos, split_source, groups, group_sizes,
             ambiguous, archives, warnings
    파일이 없거나 비어 있으면 ValueError.
    """
    path = Path(index_csv)
    if not path.exists():
        raise ValueError(f"index.csv 를 찾을 수 없습니다 — {path}")

    crops, stages, archives = Counter(), Counter(), Counter()
    combos = Counter()                 # (crop, stage) -> 장수
    split_stage = Counter()            # (split_source, stage) -> 장수
    per_group = Counter()              # group_id -> 장수
    ambiguous = 0
    n = 0

    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"crop", "stage", "group_id"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"index.csv 에 필수 컬럼이 없습니다 — {', '.join(sorted(missing))}")
        for r in reader:
            crop, stage = r["crop"], r["stage"]
            crops[crop] += 1
            stages[stage] += 1
            combos[(crop, stage)] += 1
            per_group[r["group_id"]] += 1
            if r.get("split_source"):
                split_stage[(r["split_source"], stage)] += 1
            if r.get("archive"):
                archives[r["archive"]] += 1
            if str(r.get("ambiguous", "")).lower() == "true":
                ambiguous += 1
            n += 1

    if n == 0:
        raise ValueError(f"index.csv 에 데이터 행이 없습니다 — {path}")

    sizes = sorted(per_group.values())
    s = {
        "path": str(path),
        "n": n,
        "crops": dict(crops),
        "stages": dict(stages),
        "combos": {f"{c}|{g}": v for (c, g), v in combos.items()},
        "split_source": {f"{sp}|{g}": v for (sp, g), v in split_stage.items()},
        "groups": len(per_group),
        "group_sizes": {"min": sizes[0], "max": sizes[-1],
                        "median": sizes[len(sizes) // 2]},
        "ambiguous": ambiguous,
        "archives": len(archives),
    }
    s["warnings"] = _warnings(s, combos, crops, stages, n)
    return s


def _warnings(s, combos, crops, stages, n):
    """학습을 걸기 전에 사람이 알아야 할 사실만 모은다 (판단은 하지 않는다)."""
    out = []
    if s["groups"] < MIN_GROUPS:
        out.append(f"개체(group_id) 가 {s['groups']}개뿐입니다 "
                   f"({MIN_GROUPS}개 미만) — 그룹 분할이 불안정할 수 있습니다.")
    if len(crops) == 1:
        out.append(f"작물이 '{next(iter(crops))}' 하나뿐입니다 — 작물 분류는 "
                   f"고를 것이 하나라 정확도가 항상 100% 가 됩니다. "
                   f"생육단계 지표만 의미가 있습니다.")
    for stage, cnt in sorted(stages.items(), key=lambda x: x[1]):
        if cnt / n < MIN_STAGE_RATIO:
            out.append(f"단계 '{stage}' 가 {cnt}장({cnt / n * 100:.1f}%) 로 희소합니다 "
                       f"— macro-F1 이 이 클래스에 좌우됩니다.")
    thin = [(c, g, v) for (c, g), v in sorted(combos.items()) if v < MIN_COMBO_ROWS]
    for c, g, v in thin:
        out.append(f"조합 '{c} × {g}' 이 {v}장뿐입니다 "
                   f"({MIN_COMBO_ROWS}장 미만) — 평가가 불안정합니다.")
    return out


def _table(title, rows, cols, data):
    """rows × cols 교차표를 고정폭 문자열로 그린다. data: {(row, col): 값}"""
    if not rows or not cols:
        return ""
    w0 = max([_w(title)] + [_w(r) for r in rows]) + 2
    widths = [max(_w(c), 7) + 2 for c in cols]
    lines = [_pad(title, w0) + "".join(_rpad(c, w) for c, w in zip(cols, widths))]
    lines.append("-" * (w0 + sum(widths)))
    for r in rows:
        cells = [_rpad(f"{data.get((r, c), 0):,}", w) for c, w in zip(cols, widths)]
        lines.append(_pad(r, w0) + "".join(cells))
    if len(rows) > 1:
        tot = [_rpad(f"{sum(data.get((r, c), 0) for r in rows):,}", w)
               for c, w in zip(cols, widths)]
        lines.append("-" * (w0 + sum(widths)))
        lines.append(_pad("합계", w0) + "".join(tot))
    return "\n".join(lines)


def _w(text):
    """한글은 두 칸으로 세어 고정폭 정렬을 맞춘다."""
    return sum(2 if ord(ch) > 0x2000 else 1 for ch in text)


def _pad(text, width):
    """왼쪽 정렬 — 한글 폭을 반영해 오른쪽을 채운다."""
    return text + " " * max(0, width - _w(text))


def _rpad(text, width):
    """오른쪽 정렬 — 한글 폭을 반영해 왼쪽을 채운다."""
    return " " * max(0, width - _w(text)) + text


def format_summary(s):
    """summarize_index() 결과를 사람이 읽는 텍스트로 만든다."""
    crops = sorted(s["crops"])
    stages = sorted(s["stages"], key=lambda x: -s["stages"][x])
    gs = s["group_sizes"]

    out = [
        f"index.csv : {s['path']}",
        f"규모      : {s['n']:,}장 · 개체 {s['groups']}개 · 아카이브 {s['archives']}개",
        f"개체당    : 최소 {gs['min']}장 · 중앙 {gs['median']}장 · 최대 {gs['max']}장",
    ]
    if s["ambiguous"]:
        out.append(f"ambiguous : {s['ambiguous']:,}장 "
                   f"({s['ambiguous'] / s['n'] * 100:.1f}%)")
    out.append("")

    combos = {(k.split("|")[0], k.split("|")[1]): v for k, v in s["combos"].items()}
    out.append("[작물 × 단계]")
    out.append(_table("작물", crops, stages, combos))
    out.append("")

    ratio = "  ".join(f"{g} {s['stages'][g] / s['n'] * 100:.1f}%" for g in stages)
    out.append(f"단계 비율 : {ratio}")

    if s["split_source"]:
        sp = {(k.split("|")[0], k.split("|")[1]): v
              for k, v in s["split_source"].items()}
        out += ["", "[split_source × 단계]  (분할 정책 결정 근거)",
                _table("원본", sorted({r for r, _ in sp}), stages, sp)]

    out.append("")
    if s["warnings"]:
        out.append(f"[!] 확인할 것 {len(s['warnings'])}건")
        out += [f"  - {w}" for w in s["warnings"]]
    else:
        out.append("[OK] 눈에 띄는 분포 문제 없음")
    return "\n".join(out)
