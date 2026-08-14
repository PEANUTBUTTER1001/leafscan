"""core/data_summary.py — 학습 전 분포 요약 검증.

lab_gui 를 import 하지 않는다 (tkinter + torch 동시 로드 시 Windows DLL 접근위반).
"""
import csv

import pytest

from core.data_summary import (MIN_COMBO_ROWS, MIN_GROUPS, format_summary,
                               summarize_index)

COLS = ["path", "crop", "stage", "group_id", "archive", "split_source",
        "ambiguous"]


def _write(tmp_path, rows, cols=COLS):
    p = tmp_path / "index.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _row(crop="상추", stage="생육기", gid="g1", archive="TL_1",
         split="training", ambiguous="False", path="a.jpg"):
    return dict(path=path, crop=crop, stage=stage, group_id=gid,
                archive=archive, split_source=split, ambiguous=ambiguous)


def _many(n, **kw):
    return [_row(gid=f"g{i}", **kw) for i in range(n)]


# --------------------------------------------------------------- 집계
def test_기본_집계(tmp_path):
    p = _write(tmp_path, [_row(gid="g1"), _row(gid="g1"), _row(gid="g2")])
    s = summarize_index(p)
    assert s["n"] == 3
    assert s["groups"] == 2
    assert s["crops"] == {"상추": 3}
    assert s["stages"] == {"생육기": 3}


def test_조합_교차표(tmp_path):
    p = _write(tmp_path, [_row(crop="상추", stage="정식기"),
                          _row(crop="케일", stage="수확기"),
                          _row(crop="케일", stage="수확기")])
    s = summarize_index(p)
    assert s["combos"]["상추|정식기"] == 1
    assert s["combos"]["케일|수확기"] == 2


def test_split_source별_단계(tmp_path):
    p = _write(tmp_path, [_row(split="training"), _row(split="validation"),
                          _row(split="validation")])
    s = summarize_index(p)
    assert s["split_source"]["training|생육기"] == 1
    assert s["split_source"]["validation|생육기"] == 2


def test_개체당_장수(tmp_path):
    rows = [_row(gid="g1")] * 5 + [_row(gid="g2")] * 3 + [_row(gid="g3")]
    s = summarize_index(_write(tmp_path, rows))
    assert s["group_sizes"]["min"] == 1
    assert s["group_sizes"]["max"] == 5


def test_ambiguous_집계(tmp_path):
    p = _write(tmp_path, [_row(ambiguous="True"), _row(ambiguous="true"),
                          _row(ambiguous="False")])
    assert summarize_index(p)["ambiguous"] == 2


def test_아카이브_수는_고유값(tmp_path):
    p = _write(tmp_path, [_row(archive="A"), _row(archive="A"),
                          _row(archive="B")])
    assert summarize_index(p)["archives"] == 2


# --------------------------------------------------------------- 경고
def test_개체수_부족_경고(tmp_path):
    s = summarize_index(_write(tmp_path, _many(MIN_GROUPS - 1)))
    assert any("개체" in w for w in s["warnings"])


def test_개체수_충분하면_경고없음(tmp_path):
    rows = _many(MIN_GROUPS + 1, crop="상추", stage="생육기")
    rows += [_row(crop="케일", stage="정식기", gid=f"k{i}")
             for i in range(MIN_COMBO_ROWS)]
    s = summarize_index(_write(tmp_path, rows))
    assert not any("개체(group_id)" in w for w in s["warnings"])


def test_단일_작물_경고(tmp_path):
    s = summarize_index(_write(tmp_path, _many(MIN_GROUPS + 1)))
    assert any("작물이" in w for w in s["warnings"])


def test_희소_조합_경고(tmp_path):
    rows = _many(MIN_COMBO_ROWS + 10, stage="생육기")
    rows += [_row(stage="수확기", gid="rare")]
    s = summarize_index(_write(tmp_path, rows))
    assert any("수확기" in w for w in s["warnings"])


# --------------------------------------------------------------- 오류
def test_파일_없음(tmp_path):
    with pytest.raises(ValueError, match="찾을 수 없습니다"):
        summarize_index(tmp_path / "없다.csv")


def test_빈_파일(tmp_path):
    with pytest.raises(ValueError, match="데이터 행이 없습니다"):
        summarize_index(_write(tmp_path, []))


def test_필수_컬럼_누락(tmp_path):
    p = _write(tmp_path, [{"path": "a.jpg", "crop": "상추"}],
               cols=["path", "crop"])
    with pytest.raises(ValueError, match="필수 컬럼"):
        summarize_index(p)


def test_split_source_없어도_동작(tmp_path):
    p = _write(tmp_path, [{"path": "a.jpg", "crop": "상추", "stage": "생육기",
                           "group_id": "g1"}],
               cols=["path", "crop", "stage", "group_id"])
    s = summarize_index(p)
    assert s["n"] == 1 and s["split_source"] == {}


# --------------------------------------------------------------- 출력
def test_출력_문자열(tmp_path):
    rows = [_row(crop="상추", stage="생육기", gid=f"g{i}") for i in range(3)]
    rows += [_row(crop="케일", stage="정식기", gid="k1", split="validation")]
    text = format_summary(summarize_index(_write(tmp_path, rows)))
    for token in ("규모", "작물 × 단계", "상추", "케일",
                  "단계 비율", "split_source"):
        assert token in text


def test_split_source_없으면_해당_표_생략(tmp_path):
    p = _write(tmp_path, [{"path": "a.jpg", "crop": "상추", "stage": "생육기",
                           "group_id": "g1"}],
               cols=["path", "crop", "stage", "group_id"])
    # 경로 줄에 tmp 디렉터리명이 섞이므로 표 제목으로 판정한다
    assert "분할 정책 결정 근거" not in format_summary(summarize_index(p))
