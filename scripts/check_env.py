"""scripts/check_env.py — 설치 환경 확인 (README.md §2단계 5).

학습을 돌리기 전에 **조용히 잘못될 수 있는 것들**을 잡는다.

  · torch 가 CPU 빌드 → GPU 가 있어도 안 쓰고 10배 느려지는데 에러가 안 난다
  · Tkinter 누락 (Microsoft Store 판 Python) → lab_gui.py 만 안 뜬다
  · matplotlib 한글 폰트 없음 → 그림의 한글이 전부 □ 로 깨진다

사용:
    python scripts/check_env.py

종료 코드: 정상 0 · 필수 항목 실패 1 (경고만 있으면 0)
"""
import importlib
import subprocess
import sys

# 출력 UTF-8 고정 — Windows 에서 `> log.txt` 로 리다이렉트하면 cp949 로 떨어져
# '—' 같은 문자에서 UnicodeEncodeError 가 난다 (prepare_dataset.py 와 같은 이유).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PY_MIN, PY_MAX = (3, 10), (3, 12)

# requirements.txt 의 실행 의존성. (import 이름, 표시 이름)
REQUIRED = [("torch", "torch"), ("torchvision", "torchvision"),
            ("numpy", "numpy"), ("sklearn", "scikit-learn"),
            ("matplotlib", "matplotlib"), ("PIL", "pillow")]

# core/figures.py 와 같은 후보 순서 — 여기서 통과하면 그림도 안 깨진다.
KOREAN_FONTS = ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR")

fails, warns = [], []


def ok(msg):
    print(f"[OK] {msg}")


def bad(msg, *hints, fatal=True):
    print(f"[!!] {msg}")
    for h in hints:
        print(f"     {h}")
    (fails if fatal else warns).append(msg)


def check_python():
    v = sys.version_info
    text = f"Python {v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < PY_MIN or (v.major, v.minor) > PY_MAX:
        bad(f"{text} — 지원 범위 밖입니다",
            f"{PY_MIN[0]}.{PY_MIN[1]} ~ {PY_MAX[0]}.{PY_MAX[1]} 를 쓰세요 "
            f"(권장 3.11). README.md §1.1 참조.", fatal=False)
    else:
        ok(text)


def check_torch():
    """torch 를 import 하고 버전·CUDA 빌드·GPU 인식을 본다."""
    try:
        import torch
    except ImportError:
        bad("torch 없음 — 학습을 실행할 수 없습니다",
            "README.md §2단계 3 을 따라 설치하세요.")
        return None
    ok(f"torch {torch.__version__}")

    try:
        import torchvision
        ok(f"torchvision {torchvision.__version__}")
    except ImportError:
        bad("torchvision 없음 — 사전학습 백본을 만들 수 없습니다",
            "torch 와 같은 인덱스에서 함께 설치하세요 (README.md §2단계 3).")

    cpu_build = "+cpu" in torch.__version__ or "+" not in torch.__version__
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        ok(f"CUDA 사용 가능 — {name} ({vram:.1f}GB)")
        if vram < 4:
            print(f"     VRAM 이 {vram:.1f}GB 입니다. batch_size 를 권장치보다 "
                  f"낮춰야 할 수 있습니다 (콘솔이 경고합니다).")
    else:
        hints = []
        if cpu_build:
            hints.append(f"torch 가 CPU 빌드입니다 ({torch.__version__}).")
            hints.append("GPU 를 쓰려면 README.md §2단계 3 을 다시 하세요.")
        else:
            hints.append("GPU 빌드인데 장치를 못 찾았습니다. "
                         "`nvidia-smi` 로 드라이버를 확인하세요.")
        # GPU 는 필수가 아니다 — CPU 로도 학습된다 (느릴 뿐).
        bad("CUDA 사용 불가 — CPU로 학습됩니다", *hints, fatal=False)
    return torch


def check_tkinter():
    """**별도 프로세스**로 확인한다.

    Windows 에서 tkinter 와 torch 를 한 프로세스에 함께 로드하면 DLL 접근위반이
    난 사례가 있다. 여기서는 torch 를 이미 import 한 뒤이므로 분리해서 띄운다.
    """
    code = "import tkinter; r = tkinter.Tk(); r.destroy(); print(tkinter.TkVersion)"
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        bad(f"Tkinter 확인 실패 — {e}", fatal=False)
        return
    if p.returncode == 0:
        ok(f"Tkinter 사용 가능 (Tk {p.stdout.strip()})")
    else:
        bad("Tkinter 없음 — 콘솔 GUI를 실행할 수 없습니다",
            "Microsoft Store 버전 Python일 가능성이 높습니다. README.md §1.2 참조.",
            "학습 자체는 train_worker.py 로 계속 쓸 수 있습니다.")


def check_font():
    try:
        from matplotlib import font_manager
    except ImportError:
        return          # matplotlib 누락은 아래 패키지 점검에서 잡는다
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in KOREAN_FONTS:
        if cand in names:
            ok(f"matplotlib 한글 폰트 — {cand}")
            return
    bad("matplotlib 한글 폰트 없음 — 그림의 한글이 □ 로 깨집니다",
        f"후보: {' · '.join(KOREAN_FONTS)}",
        "학습·지표에는 영향이 없습니다. 그림만 다시 만들면 됩니다.", fatal=False)


def check_packages():
    missing = []
    for mod, shown in REQUIRED:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(shown)
    if missing:
        bad(f"필수 패키지 누락 {len(missing)}개 — {', '.join(missing)}",
            "pip install -r requirements.txt")
    else:
        ok(f"필수 패키지 {len(REQUIRED)}종 확인")


def main():
    check_python()
    check_torch()
    check_tkinter()
    check_font()
    check_packages()

    print()
    if fails:
        print(f"→ 문제 {len(fails)}건 — 위 안내를 따라 해결한 뒤 다시 실행하세요.")
        return 1
    if warns:
        print(f"→ 경고 {len(warns)}건 있으나 학습은 가능합니다. "
              f"`python lab_gui.py` 를 실행하세요.")
        return 0
    print("→ 준비 완료. `python lab_gui.py` 를 실행하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
