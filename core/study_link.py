"""core/study_link.py — GUI ↔ run_study.py 브리지 (순수 로직).

lab_gui.py 의 Study 실행 모드가 쓰는 문자열·config 변환만 담는다.
**tkinter 를 import 하지 않는다** — 그래야 테스트가 GUI 없이 돈다.

담는 것:
  · run_study.py 표준 출력 → 큐 상태 파싱 (run_study.py 는 수정하지 않는다)
  · 단일 실행 config → Study base config
  · study 폴더의 완료 member run 수집

run_study.py 의 로그 문구가 바뀌면 tests/test_gui_study.py 가 먼저 깨진다.
"""
import json
import re
from pathlib import Path

# Study base config 에서 빼는 키 — simple_cnn 전용 구조값.
# 특정 모델 전용 설정은 공정 비교에 넣지 않는다 (SRS §3.4).
SIMPLE_CNN_KEYS = ("conv_channels", "kernel_size", "stride", "padding",
                   "pool_kernel", "pool_stride", "conv_blocks", "use_relu")


def fmt_hms(sec):
    """초 → 00:12:34. 음수는 0 으로 본다."""
    sec = max(0, int(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def study_arch_from_run_name(text):
    """'study__arch-resnet18__lr-0.001' → 'resnet18'. 없으면 None.

    --lr-search 를 쓰면 run 이름에 lr 이 덧붙으므로 '__' 앞까지만 취한다.
    """
    m = re.search(r"arch-([^\s·]+)", text or "")
    return m.group(1).split("__")[0] if m else None


def parse_study_line(line):
    """run_study.py 로그 한 줄 → (상태, arch). 해당 없으면 (None, None).

    대응 문구 (run_study.py 원문):
      [실행] {run} · {override}     [건너뜀] {run} — 이미 완료됨
      [실패] {run}                  [lr탐색] {arch}: ...  /  {arch} 최적 lr=...
      [완료] N개 run · {dir}
    """
    s = (line or "").strip()
    if s.startswith("[실행]"):
        return "running", study_arch_from_run_name(s)
    if s.startswith("[건너뜀]"):
        return "skipped", study_arch_from_run_name(s)
    if s.startswith("[실패]"):
        return "failed", study_arch_from_run_name(s)
    if s.startswith("[lr탐색]"):
        rest = s[len("[lr탐색]"):].strip()
        head = rest.split(":", 1)[0].split()
        return "lr_search", (head[0] if head else None)
    if s.startswith("[완료]"):
        return "finished", None
    return None, None


def validate_study_name(name):
    """(ok, 결과) — ok 면 결과는 정규화된 이름, 아니면 안내 문구."""
    n = (name or "").strip()
    if not n:
        return False, "Study 이름을 입력하세요."
    if any(c in n for c in '\\/:*?"<>|'):
        return False, 'Study 이름에 \\ / : * ? " < > | 는 쓸 수 없습니다.'
    if n.startswith("."):
        return False, "Study 이름은 '.' 으로 시작할 수 없습니다."
    if n.startswith("_"):
        # '_' 로 시작하는 폴더는 보조 run(_lrsearch)으로 취급돼 비교에서 빠진다
        return False, "Study 이름은 '_' 로 시작할 수 없습니다."
    return True, n


def build_study_base_config(cfg, study_name, archs):
    """단일 실행용 config → Study base config.

    arch 는 --vary 가 run 마다 덮어쓰지만, base 에도 유효한 값을 남긴다.
    simple_cnn 전용 키는 제거한다 (core/models.py 가 모두 기본값을 갖는다).
    """
    out = {k: v for k, v in cfg.items() if k not in SIMPLE_CNN_KEYS}
    if archs:
        out["arch"] = archs[0]
    out["study"] = study_name
    # Study 모드는 가설 입력칸을 숨기므로 사람이 쓸 기회가 없다.
    # 리포트 §1 을 비워 두면 멤버마다 '⚠️ 미작성' 이 뜨므로 여기서 채운다.
    # 해석(§7 결론)은 그대로 사람 몫이다.
    out["hypothesis"] = study_hypothesis(study_name, archs)
    out["parent_run"] = None
    return out


def study_hypothesis(study_name, archs):
    """멤버 리포트 §1 에 들어갈 자동 가설 문구."""
    if not archs:
        return f"{study_name} — 같은 조건에서 모델을 비교한다."
    return (f"{study_name} — 같은 조건에서 {len(archs)}개 모델을 비교한다 "
            f"({', '.join(archs)}).")


# Study 폴더 이름의 자동 번호 — 'study_01', 'study_02_backbone' 둘 다 인식한다.
_STUDY_NUM = re.compile(r"^study_(\d+)")


def next_study_name(runs_dir):
    """runs/ 를 훑어 다음 Study 이름을 만든다 — 'study_03' 형식.

    단일 실행의 exp_001·exp_002 자동 증가와 같은 규칙이다. 다만 exp 는 빈 번호를
    찾아 채우는 반면, study 는 **최대 번호 + 1** 을 쓴다. study 는 재개를 위해
    이름을 다시 입력하는 일이 있어, 지웠던 번호를 재사용하면 옛 폴더와 헷갈린다.

    주제 접미사(`study_02_backbone`)는 사용자가 덧붙인다 — 번호만 읽는다.
    """
    runs_dir = Path(runs_dir)
    used = set()
    if runs_dir.exists():
        for d in runs_dir.iterdir():
            if not d.is_dir():
                continue
            m = _STUDY_NUM.match(d.name)
            if m:
                used.add(int(m.group(1)))
    return f"study_{(max(used) + 1) if used else 1:02d}"


def members_needing_report(study_dir, done=()):
    """리포트를 만들어야 할 멤버 [(arch, run_dir)…] — 이미 처리한 것은 뺀다.

    done 에는 모델 완료 시점(A)에 이미 후처리한 arch 를 넘긴다. Study 종료(B)에서
    이 함수를 다시 불러 **재개로 건너뛴 멤버**처럼 A 가 놓친 것만 보충한다.
    """
    done = set(done)
    return [(arch, run) for arch, run in sorted(collect_study_members(study_dir).items())
            if arch not in done]


# 비교 조건에 영향을 주지 않는 키 — 달라도 경고하지 않는다.
# arch 는 --vary 가 run 마다 바꾸는 값이므로 base 끼리 비교할 의미가 없다.
_DIFF_IGNORE = {"arch", "hypothesis", "tags", "parent_run", "study",
                "schema_version", "num_workers", "device"}


def study_config_diff(base_path, new_cfg):
    """이전 base config 와 지금 설정의 차이 [(키, 이전, 지금)…].

    파일이 없거나 읽을 수 없으면 빈 목록(=비교 불가, 경고하지 않음).
    """
    path = Path(base_path)
    if not path.exists():
        return []
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    keys = (set(old) | set(new_cfg)) - _DIFF_IGNORE
    return sorted((k, old.get(k), new_cfg.get(k))
                  for k in keys if old.get(k) != new_cfg.get(k))


def collect_study_members(study_dir):
    """{arch: run_dir} — metrics.json 이 있는 완료 member run 만. 보조 run 제외."""
    found = {}
    study_dir = Path(study_dir)
    if not study_dir.exists():
        return found
    for run in sorted(study_dir.iterdir()):
        if not run.is_dir() or run.name.startswith("_"):
            continue
        if not (run / "metrics.json").exists():
            continue
        arch = study_arch_from_run_name(run.name)
        if arch:
            found[arch] = run
    return found
