"""데이터 인제스트 — 추출본 짝 정리 · 번호 공백 판정 · 진행 표시 검증."""
import importlib.util
import io
import tarfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "prepare_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_dataset", SCRIPT)
prepare_dataset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_dataset)

Archive = prepare_dataset.Archive
Reporter = prepare_dataset.Reporter


class FakeClock:
    """테스트에서 시간을 직접 움직인다."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, sec):
        self.t += sec


def make_reporter(tty, clock=None, **kw):
    stream = io.StringIO()
    rep = Reporter(stream=stream, clock=clock or FakeClock(), tty=tty, **kw)
    return rep, stream


def archive(name, crop_folder, split, role, number, root=None, stems=()):
    """Archive 객체. root 를 주면 실제 tar 파일까지 만든다.

    validate_pairs 는 짝이 모두 있으면 tar 를 열어 stem 교집합을 보므로,
    그 경로를 타는 테스트에는 실제 파일이 필요하다.
    """
    path = Path(f"/data/{crop_folder}/{split}/{name}.tar")
    if root is not None:
        path = Path(root) / crop_folder / split / role / f"{name}.tar"
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = ".json" if role == "labeled" else ".jpg"
        payload = b"{}" if role == "labeled" else b"jpg"
        with tarfile.open(path, "w") as t:
            for s in stems:
                info = tarfile.TarInfo(f"{s}{ext}")
                info.size = len(payload)
                t.addfile(info, io.BytesIO(payload))
    return Archive(path, crop_folder, split, role, number)


def paired(root, number, stems=("a", "b", "c")):
    """같은 stem 을 가진 TL/TS 한 쌍."""
    return [
        archive(f"TL_3.상추{number}", "lettuce", "training", "labeled", number,
                root=root, stems=stems),
        archive(f"TS_3.상추{number}", "lettuce", "training", "source", number,
                root=root, stems=stems),
    ]


def test_reconcile_extracted_pair_deletes_unpaired_files(tmp_path):
    labeled = tmp_path / "TL"
    source = tmp_path / "TS"
    labeled.mkdir()
    source.mkdir()
    (labeled / "paired.json").write_text("{}", encoding="utf-8")
    (source / "paired.jpg").write_bytes(b"jpg")
    (labeled / "label_only.json").write_text("{}", encoding="utf-8")
    (source / "image_only.jpg").write_bytes(b"jpg")

    stems, json_only, jpg_only = prepare_dataset.reconcile_extracted_pair(labeled, source)

    assert stems == {"paired"}
    assert (json_only, jpg_only) == (1, 1)
    assert (labeled / "paired.json").exists()
    assert (source / "paired.jpg").exists()
    assert not (labeled / "label_only.json").exists()
    assert not (source / "image_only.jpg").exists()


# ======================================================================
# 아카이브 번호 공백 — 연속성은 무결성 기준이 아니다
# ======================================================================
def test_number_gaps_detects_missing_middle_numbers(tmp_path):
    """1,2,3,6,7,8 → 4,5 는 공백. 양쪽 다 없으므로 오류가 아니다."""
    archives = []
    for n in ("1", "2", "3", "6", "7", "8"):
        archives += paired(tmp_path, n)

    gaps = prepare_dataset.number_gaps(archives)
    assert gaps[("lettuce", "training")] == [4, 5]

    pairs, errors = prepare_dataset.validate_pairs(archives)
    assert errors == []                            # 번호 공백은 오류가 아니다
    assert len(pairs) == 6                         # 존재하는 번호만 짝으로


def test_number_gaps_ignores_one_sided_numbers(tmp_path):
    """한쪽만 있는 번호는 '공백'이 아니라 결손이다."""
    archives = paired(tmp_path, "1") + [
        archive("TL_3.상추2", "lettuce", "training", "labeled", "2",
                root=tmp_path, stems=("a",)),      # source 없음
    ]
    # 2 는 존재하는 번호이므로 공백 목록에 없다
    assert prepare_dataset.number_gaps(archives)[("lettuce", "training")] == []

    _pairs, errors = prepare_dataset.validate_pairs(archives)
    assert len(errors) == 1
    e = errors[0]
    assert e["kind"] == "missing_source"
    assert e["number"] == "2"
    assert not e.get("soft")                       # 하드 오류
    assert "번호 공백이 아니라" in e["hint"]        # 오해 방지 문구
    assert "labeled: [1, 2]" in e["compo"]
    assert "source: [1]" in e["compo"]


def test_number_gaps_separates_gap_and_missing(tmp_path):
    """공백과 결손이 함께 있어도 서로 섞이지 않는다."""
    archives = paired(tmp_path, "1") + [
        archive("TL_3.상추5", "lettuce", "training", "labeled", "5",
                root=tmp_path, stems=("a",)),      # source 없음
    ]
    assert prepare_dataset.number_gaps(archives)[("lettuce", "training")] == [2, 3, 4]
    _pairs, errors = prepare_dataset.validate_pairs(archives)
    assert [e["kind"] for e in errors] == ["missing_source"]


def test_zero_intersection_is_still_an_error(tmp_path):
    """번호는 맞지만 내부 stem 이 전혀 겹치지 않으면 오류다."""
    archives = [
        archive("TL_3.상추1", "lettuce", "training", "labeled", "1",
                root=tmp_path, stems=("a", "b")),
        archive("TS_3.상추1", "lettuce", "training", "source", "1",
                root=tmp_path, stems=("x", "y")),
    ]
    _pairs, errors = prepare_dataset.validate_pairs(archives)
    assert [e["kind"] for e in errors] == ["zero_intersection"]
    assert not errors[0].get("soft")


def test_number_gaps_single_archive_has_no_gap():
    archives = [archive("TL_3.상추1", "lettuce", "training", "labeled", "1")]
    assert prepare_dataset.number_gaps(archives)[("lettuce", "training")] == []


# ======================================================================
# 진행 표시
# ======================================================================
def test_fmt_hms():
    assert prepare_dataset.fmt_hms(0) == "00:00:00"
    assert prepare_dataset.fmt_hms(59) == "00:00:59"
    assert prepare_dataset.fmt_hms(3661) == "01:01:01"
    assert prepare_dataset.fmt_hms(-5) == "00:00:00"       # 음수 방어


def test_fmt_eta():
    # 10초에 25% 처리 → 남은 75% 는 30초
    assert prepare_dataset.fmt_eta(10, 25, 100) == "00:00:30"
    # 추정 불가한 경우들
    assert prepare_dataset.fmt_eta(10, 0, 100) == ""       # 0 나눗셈 방어
    assert prepare_dataset.fmt_eta(10, 100, 100) == ""     # 이미 완료
    assert prepare_dataset.fmt_eta(10, 5, 0) == ""         # 전체를 모름
    assert prepare_dataset.fmt_eta(0, 5, 100) == ""        # 경과 0


def test_progress_is_throttled_by_time():
    clock = FakeClock()
    rep, out = make_reporter(tty=True, clock=clock, interval=0.5, spinner=False)
    rep.progress(1, 100, "처리")          # 첫 호출은 항상
    rep.progress(2, 100, "처리")          # 0초 경과 → 억제
    rep.progress(3, 100, "처리")          # 억제
    clock.advance(0.6)
    rep.progress(4, 100, "처리")          # 주기 경과 → 출력
    rep.progress(100, 100, "처리")        # 마지막은 항상
    shown = [n for n in (1, 2, 3, 4, 100) if f"{n} / 100" in out.getvalue()]
    assert shown == [1, 4, 100]


def test_progress_non_tty_uses_newlines_not_carriage_return():
    """파일·파이프에서는 \\r 를 쓰지 않는다. 안 그러면 로그가 한 줄로 뭉친다."""
    clock = FakeClock()
    rep, out = make_reporter(tty=False, clock=clock, interval=0.5)
    rep.progress(1, 100, "처리")
    clock.advance(30)
    rep.progress(50, 100, "처리")
    text = out.getvalue()
    assert "\r" not in text
    assert text.count("\n") == 2


def test_progress_tty_uses_carriage_return_and_spinner():
    clock = FakeClock()
    rep, out = make_reporter(tty=True, clock=clock, interval=0.5, spinner=True)
    rep.progress(1, 100, "처리")
    text = out.getvalue()
    assert text.startswith("\r")
    assert "\n" not in text                       # 제자리 갱신이라 줄바꿈 없음
    assert text.rstrip()[-1] in prepare_dataset.SPINNER


def test_spinner_off_when_not_tty():
    rep, _out = make_reporter(tty=False, spinner=True)
    assert rep.spinner is False                   # 비TTY 면 자동으로 꺼진다


def test_quiet_suppresses_progress_but_not_lines():
    rep, out = make_reporter(tty=True, quiet=True)
    rep.progress(1, 100, "처리")
    rep.spin("무언가")
    assert out.getvalue() == ""
    rep.note("중요한 소식")
    assert "[진행] 중요한 소식" in out.getvalue()


def test_line_clears_live_progress_first():
    """경고가 진행 줄 위에 겹쳐 찍히면 화면이 깨진다."""
    clock = FakeClock()
    rep, out = make_reporter(tty=True, clock=clock, interval=0.5, spinner=False)
    rep.progress(1, 100, "처리")
    live_len = len(rep._live)
    out.truncate(0), out.seek(0)
    rep.line("[경고] 문제 발생")
    text = out.getvalue()
    assert text.startswith("\r" + " " * live_len + "\r")   # 지우고
    assert text.endswith("[경고] 문제 발생\n")              # 찍는다
    assert rep._live == ""


def test_log_routes_through_active_reporter():
    """기존 log() 호출도 진행 줄을 지우고 찍어야 한다."""
    clock = FakeClock()
    rep, out = make_reporter(tty=True, clock=clock, spinner=False)
    rep.progress(1, 100, "처리")
    prev = prepare_dataset._REPORT
    prepare_dataset._REPORT = rep
    try:
        prepare_dataset.log("[경고] 기존 문구")
    finally:
        prepare_dataset._REPORT = prev
    assert "[경고] 기존 문구\n" in out.getvalue()
    assert rep._live == ""


def test_step_header_resets_phase_timing():
    clock = FakeClock()
    rep, out = make_reporter(tty=False, clock=clock)
    clock.advance(100)
    rep.step(2, "짝 검증")
    assert rep.phase_start == clock.t          # 단계마다 경과 시간 재시작
    assert "[단계 2/5] 짝 검증" in out.getvalue()
