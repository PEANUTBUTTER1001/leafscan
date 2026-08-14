"""GUI Study 실행 모드의 순수 로직 검증 — 창을 띄우지 않는다.

lab_gui.py 는 run_study.py 의 표준 출력을 그대로 읽어 큐 상태를 만든다.
따라서 run_study.py 의 로그 문구가 바뀌면 여기서 먼저 깨져야 한다.

core.study_link 만 import 한다 (tkinter 비의존) — 같은 pytest 프로세스에서
tkinter 와 torch 를 함께 로드하면 Windows 에서 DLL 충돌이 난다.

    python tests/test_gui_study.py   또는   pytest tests/test_gui_study.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.study_link import (build_study_base_config, collect_study_members,
                             fmt_hms, parse_study_line, study_arch_from_run_name,
                             study_config_diff, validate_study_name)


def test_fmt_hms():
    assert fmt_hms(0) == "00:00:00"
    assert fmt_hms(59) == "00:00:59"
    assert fmt_hms(3661) == "01:01:01"
    assert fmt_hms(-5) == "00:00:00"       # 음수 방어


# --------------------------------------------------------------- run 이름
def test_arch_from_run_name():
    assert study_arch_from_run_name("study_01__arch-resnet18") == "resnet18"
    assert study_arch_from_run_name(
        "study_01__arch-mobilenet_v3_small") == "mobilenet_v3_small"
    # --lr-search 를 쓰면 run 이름에 lr 이 덧붙는다
    assert study_arch_from_run_name(
        "study_01__arch-resnet18__lr-0.001") == "resnet18"
    assert study_arch_from_run_name(
        "study_01__arch-efficientnet_b0__lr-0.003") == "efficientnet_b0"
    assert study_arch_from_run_name("study_01__base") is None
    assert study_arch_from_run_name("") is None


# ------------------------------------------------------------- 로그 파싱
def test_parse_study_line():
    """run_study.py 의 실제 출력 문구 그대로."""
    assert parse_study_line(
        "[실행] study_01__arch-resnet18 · {'arch': 'resnet18'}") == (
            "running", "resnet18")
    assert parse_study_line(
        "[건너뜀] study_01__arch-resnet50 — 이미 완료됨") == ("skipped", "resnet50")
    assert parse_study_line(
        "[실패] study_01__arch-convnext_tiny") == ("failed", "convnext_tiny")
    assert parse_study_line("[완료] 3개 run · runs/study_01") == ("finished", None)
    # lr 탐색은 두 가지 문구가 나온다 (둘 다 앞에 공백 2칸)
    assert parse_study_line(
        "  [lr탐색] resnet18: (0.001, 0.003, 0.01) · 6epoch") == (
            "lr_search", "resnet18")
    assert parse_study_line(
        "  [lr탐색] resnet18 최적 lr=0.003 (val_loss 0.5120)") == (
            "lr_search", "resnet18")


def test_parse_study_line_ignores_worker_output():
    """train_worker 로그나 무관한 줄은 상태를 바꾸지 않는다."""
    for line in ("[split] runs/study_01/split.json 생성 (전 run 공유)",
                 "  epoch   3/ 20 | loss 0.4120 val_loss 0.5510",
                 "[STUDY.md] runs/study_01/STUDY.md 생성",
                 "공정성 경고: 'img_size' 는 모델별 허용 항목이 아닙니다",
                 "", "   "):
        assert parse_study_line(line) == (None, None), line


# ----------------------------------------------------------- 이름 검증
def test_validate_study_name():
    assert validate_study_name("study_01_backbone") == (True, "study_01_backbone")
    assert validate_study_name("  study_01  ")[1] == "study_01"
    for bad in ("", "   ", "a/b", "a\\b", "a:b", "a*b", "a?b", 'a"b',
                "a<b", "a>b", "a|b", ".hidden", "_lrsearch"):
        ok, msg = validate_study_name(bad)
        assert not ok, bad
        assert msg, bad


# ------------------------------------------------------- base config
def test_build_study_base_config_drops_simple_cnn_keys():
    cfg = {"arch": "simple_cnn", "epochs": 20, "batch_size": 16, "lr": 1e-3,
           "img_size": 224, "seed": 42, "hypothesis": "단일 실행용 가설",
           "parent_run": "exp_003", "study": None,
           "conv_channels": 32, "kernel_size": 5, "conv_blocks": 3,
           "stride": 2, "padding": 1, "pool_kernel": 3, "pool_stride": 2,
           "use_relu": True}
    out = build_study_base_config(cfg, "study_01", ["resnet18", "resnet50"])

    for k in ("conv_channels", "kernel_size", "conv_blocks", "stride",
              "padding", "pool_kernel", "pool_stride", "use_relu"):
        assert k not in out, k
    assert out["arch"] == "resnet18"        # --vary 가 run 마다 덮어쓴다
    assert out["study"] == "study_01"
    assert out["hypothesis"] == ""
    assert out["parent_run"] is None
    # 공통 학습 설정은 그대로 전달돼야 한다 (공정 비교)
    for k in ("epochs", "batch_size", "lr", "img_size", "seed"):
        assert out[k] == cfg[k], k
    assert cfg["conv_channels"] == 32       # 원본 불변


# --------------------------------------------------------- 멤버 수집
def _member(study_dir, name, arch):
    d = study_dir / name
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(
        json.dumps({"config": {"arch": arch}}), encoding="utf-8")
    return d


def test_collect_study_members(tmp_path):
    study = tmp_path / "study_01"
    _member(study, "study_01__arch-resnet18", "resnet18")
    _member(study, "study_01__arch-resnet50__lr-0.001", "resnet50")
    # 보조 run 과 미완 run 은 제외
    _member(study, "_lrsearch__resnet18__lr-0.001", "resnet18")
    (study / "study_01__arch-convnext_tiny").mkdir()          # metrics.json 없음
    (study / "_gui_base.json").write_text("{}", encoding="utf-8")

    found = collect_study_members(study)
    assert set(found) == {"resnet18", "resnet50"}
    assert found["resnet50"].name == "study_01__arch-resnet50__lr-0.001"
    assert collect_study_members(tmp_path / "없는폴더") == {}


# ------------------------------------------- 재개 시 설정 변경 감지
def test_study_config_diff_flags_condition_changes(tmp_path):
    """재개는 metrics.json 유무만 보므로, 조건이 바뀌면 비교표가 오염된다."""
    base = tmp_path / "_gui_base.json"
    old = {"dataset": "fake", "img_size": 224, "epochs": 20, "seed": 42,
           "arch": "simple_cnn", "study": "s1", "hypothesis": "옛날 가설"}
    base.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")

    new = dict(old, dataset="index_csv", img_size=192, arch="resnet18",
               hypothesis="", study="s1")
    diff = study_config_diff(base, new)

    keys = [k for k, _o, _n in diff]
    assert keys == ["dataset", "img_size"]          # 조건이 바뀐 것만
    assert ("dataset", "fake", "index_csv") in diff
    # arch·hypothesis·study 는 run 마다 달라지는 값이라 경고 대상이 아니다
    assert "arch" not in keys and "hypothesis" not in keys and "study" not in keys


def test_study_config_diff_no_change(tmp_path):
    base = tmp_path / "_gui_base.json"
    cfg = {"dataset": "index_csv", "img_size": 192, "epochs": 2}
    base.write_text(json.dumps(cfg), encoding="utf-8")
    assert study_config_diff(base, dict(cfg, arch="resnet50")) == []


def test_study_config_diff_missing_or_broken_file(tmp_path):
    """비교할 수 없으면 조용히 넘어간다 (새 study·손상 파일)."""
    assert study_config_diff(tmp_path / "없음.json", {"a": 1}) == []
    broken = tmp_path / "_gui_base.json"
    broken.write_text("{ 깨진 json", encoding="utf-8")
    assert study_config_diff(broken, {"a": 1}) == []


def test_study_config_diff_detects_added_key(tmp_path):
    base = tmp_path / "_gui_base.json"
    base.write_text(json.dumps({"epochs": 2}), encoding="utf-8")
    diff = study_config_diff(base, {"epochs": 2, "batch_size": 64})
    assert diff == [("batch_size", None, 64)]


if __name__ == "__main__":
    import tempfile

    for fn in (test_arch_from_run_name, test_parse_study_line,
               test_parse_study_line_ignores_worker_output,
               test_validate_study_name,
               test_build_study_base_config_drops_simple_cnn_keys):
        fn(); print(f"  ✓ {fn.__name__}")
    with tempfile.TemporaryDirectory() as td:
        test_collect_study_members(Path(td)); print("  ✓ test_collect_study_members")
    print("[GUI Study] 전체 통과")
