"""
lab_gui.py — LeafScan Lab 실험 콘솔 (Tkinter 단독 창)

GUI는 편집기, 실행 주체는 train_worker.py (구조도 §2).
  · 화면 값을 runs/<run_id>/config.json 으로 쓰고
  · train_worker.py 를 별도 프로세스로 띄우고 (학습)
  · 종료 후 make_figures.py / make_report.py 를 띄우고 (그림·리포트)
  · stdout JSON 라인과 결과 파일을 읽어 보여준다.

핵심 기능:
  · arch 세그먼트 (레지스트리 6종). simple_cnn 파라미터는 arch=simple_cnn 일 때만 노출
  · 멀티헤드 지표 카드 (stage macro-F1 · 정식기 recall · crop acc · 미판정률)
  · 혼동행렬 head 탭
  · 가설(실행 전)·결론(실행 후) 입력 → report.md 반영
  · 초기 run(exp_001~009, 단일 head/logits.npy)도 그대로 표시 (하위호환)

실행 모드 (좌측 최상단에서 선택):
  · 단일 실행 — 기존 흐름. train_worker.py 1개 프로세스
  · Study 실행 — run_study.py 1개 프로세스로 선택 모델을 순차 비교 학습
                 (분할 고정·중단 후 재개·STUDY.md 는 run_study.py 의 기존 규칙 그대로)

실행:  python lab_gui.py
"""
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

try:
    import numpy as np
except ImportError:                                            # noqa: BLE001
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.study_link import (build_study_base_config, collect_study_members,
                             parse_study_line, study_config_diff,
                             validate_study_name)

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
WORKER = BASE / "train_worker.py"
FIGURES = BASE / "make_figures.py"
REPORT = BASE / "make_report.py"
STUDY_RUNNER = BASE / "run_study.py"
STUDY_FIGURES = BASE / "scripts" / "make_study.py"


def _console_python():
    """자식 프로세스를 띄울 인터프리터 — pythonw.exe 는 피한다.

    'LeafScan Lab.bat' 은 창을 숨기려고 pythonw.exe 로 GUI 를 띄운다.
    pythonw 로 실행된 프로세스가 표준 핸들을 **상속만** 시켜 손자 프로세스를
    만들면 그 손자의 sys.stdout/sys.stderr 가 None 이 된다.
      GUI(pythonw) → run_study.py(PIPE 명시, 정상) → train_worker.py(상속) = None
    그러면 train_worker 의 print 는 조용히 사라지고(로그·진행률 유실),
    torchvision 이 사전학습 가중치를 내려받을 때 tqdm 이 sys.stderr 에 쓰다가
    'NoneType' object has no attribute 'write' 로 학습이 죽는다.
    CREATE_NO_WINDOW 를 함께 주므로 python.exe 를 써도 콘솔 창은 뜨지 않는다.
    """
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        cand = exe.with_name("python.exe")
        if cand.exists():
            return str(cand)
    return sys.executable


PYTHON = _console_python()

BG, CARD, LINE, INK, MUTED = "#f4f5f7", "#ffffff", "#dcdfe4", "#1f2328", "#6b7280"
ACCENT, WARN, OK = "#2f6f4f", "#b4432f", "#2f6f4f"

ARCHS = ["simple_cnn", "resnet18", "mobilenet_v3_small",
         "efficientnet_b0", "resnet50", "convnext_tiny"]
# arch별 상대 학습시간 계수 (러프 추정용 — conv_blocks 대신)
ARCH_COST = {"simple_cnn": 0.2, "resnet18": 1.0, "mobilenet_v3_small": 0.6,
             "efficientnet_b0": 1.3, "resnet50": 2.4, "convnext_tiny": 2.6}

# GUI 기본 설정 — 시작 상태이자 '기본값 초기화' 의 복원 대상.
# {키: (tk 변수 타입, 기본값)} — 여기 한 곳만 고치면 둘 다 따라온다.
DEFAULTS = {
    "dataset":          (tk.StringVar,  "index_csv"),
    "data_root":        (tk.StringVar,  str(BASE / "data")),
    "index_csv":        (tk.StringVar,  str(BASE / "data" / "index.csv")),
    "subset":           (tk.IntVar,     0),
    "val_ratio":        (tk.DoubleVar,  0.3),
    "split_policy":     (tk.StringVar,  "resplit_all"),
    "ambiguous_policy": (tk.StringVar,  "include"),
    "crop_mode":        (tk.StringVar,  "full"),
    "augment":          (tk.BooleanVar, True),
    "arch":             (tk.StringVar,  "resnet18"),
    "img_size":         (tk.IntVar,     224),
    "pretrained":       (tk.BooleanVar, True),
    "freeze_epochs":    (tk.IntVar,     3),
    # simple_cnn 전용
    "conv_channels":    (tk.IntVar,     16),
    "kernel_size":      (tk.IntVar,     3),
    "stride":           (tk.IntVar,     2),
    "pool_kernel":      (tk.IntVar,     3),
    "pool_stride":      (tk.IntVar,     2),
    "conv_blocks":      (tk.IntVar,     1),
    "use_relu":         (tk.BooleanVar, True),
    "epochs":           (tk.IntVar,     20),
    "batch_size":       (tk.IntVar,     16),
    "lr_log":           (tk.DoubleVar,  -3.0),
    "lr_finetune_log":  (tk.DoubleVar,  -4.0),
    "optimizer":        (tk.StringVar,  "AdamW"),
    "class_weight":     (tk.StringVar,  "auto"),
    "w_crop":           (tk.DoubleVar,  0.4),
    "w_stage":          (tk.DoubleVar,  0.6),
    "patience":         (tk.IntVar,     5),
    "seed":             (tk.IntVar,     42),
}
# Study 모드 기본값 (초기화 대상)
DEFAULT_MODE = "단일 실행"
DEFAULT_STUDY_NAME = "study_01_backbone"

# 큐 상태 표기 — (기호, 문구, 색)
STUDY_STATES = {
    "pending":   ("○", "대기", MUTED),
    "lr_search": ("◌", "LR 탐색 중", ACCENT),
    "running":   ("▶", "실행 중", ACCENT),
    "done":      ("✔", "완료", OK),
    "skipped":   ("⏭", "건너뜀", MUTED),
    "failed":    ("✖", "실패", WARN),
}


# ==========================================================================
# 하위호환 요약 — 구버전(단일 head)·신버전(멀티헤드) run 을 통일해 읽는다
# ==========================================================================
def normalize_run_summary(metrics):
    """run metrics.json 을 카드용 요약으로 정규화. 구/신 버전 모두 처리."""
    per_head = metrics.get("per_head") or {}
    out = {"stage_macro_f1": None, "primary_recall": None, "crop_acc": None,
           "abstain_rate": None, "val_acc": metrics.get("best_val_acc"),
           "heads": list((metrics.get("heads") or {}).keys()) or None}
    if "stage" in per_head:
        st = per_head["stage"]
        out["stage_macro_f1"] = st.get("macro_f1")
        out["primary_recall"] = st.get("primary_recall")
        out["abstain_rate"] = st.get("abstain_rate")
    if "crop" in per_head:
        out["crop_acc"] = per_head["crop"].get("accuracy")
    if not per_head:
        # 구버전 — 단일 head accuracy 만 존재
        out["legacy_acc"] = metrics.get("best_val_acc")
        out["heads"] = ["label"]
    return out


def available_heads(run: Path, metrics):
    """혼동행렬 탭용 head 목록과 logits/labels 파일 경로."""
    heads = list((metrics.get("heads") or {}).keys())
    result = []
    for h in heads:
        lg, lb = run / f"logits_{h}.npy", run / f"labels_{h}.npy"
        if lg.exists() and lb.exists():
            names = (metrics.get("label_names") or {}).get(h) or metrics.get("classes")
            result.append((h, lg, lb, names))
    if not result and (run / "logits.npy").exists():   # 구버전 하위호환
        result.append(("label", run / "logits.npy", run / "labels.npy",
                       metrics.get("classes")))
    return result


def pick_font():
    import tkinter.font as tkfont
    fams = set(tkfont.families())
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in fams:
            return name
    return tkfont.nametofont("TkDefaultFont").actual("family")


# ==========================================================================
# 위젯
# ==========================================================================
class MetricCard(ttk.Frame):
    def __init__(self, master, title, fmt="{:.3f}", higher_is_better=True, **kw):
        super().__init__(master, style="Card.TFrame", padding=(10, 8), **kw)
        self.fmt, self.hib = fmt, higher_is_better
        ttk.Label(self, text=title, style="CardTitle.TLabel").pack(anchor="w")
        self.value = ttk.Label(self, text="—", style="CardValue.TLabel")
        self.value.pack(anchor="w")
        self.delta = ttk.Label(self, text="기준 run 없음", style="CardDelta.TLabel")
        self.delta.pack(anchor="w")

    def update_value(self, v, base=None):
        if v is None:
            self.value.config(text="—")
            self.delta.config(text="—", foreground=MUTED)
            return
        self.value.config(text=self.fmt.format(v))
        if base is None:
            self.delta.config(text="기준 run 없음", foreground=MUTED)
            return
        d = v - base
        good = (d > 0) if self.hib else (d < 0)
        arrow = "▲" if d > 0 else ("▼" if d < 0 else "＝")
        self.delta.config(text=f"{arrow} {d:+.3f} (기준 {self.fmt.format(base)})",
                          foreground=OK if good else (WARN if d else MUTED))


class LossChart(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, bg=CARD, highlightthickness=1,
                         highlightbackground=LINE, **kw)
        self.series = {"train": [], "val": []}
        self.marks = []
        self.bind("<Configure>", lambda e: self.redraw())

    def reset(self):
        self.series = {"train": [], "val": []}
        self.marks = []
        self.redraw()

    def add(self, epoch, tr, va):
        self.series["train"].append((epoch, tr))
        self.series["val"].append((epoch, va))
        self.redraw()

    def mark(self, epoch, label):
        self.marks.append((epoch, label))
        self.redraw()

    def load(self, history, marks=None):
        self.series = {"train": [(h["epoch"], h["loss"]) for h in history],
                       "val": [(h["epoch"], h["val_loss"]) for h in history]}
        self.marks = [(m["epoch"], m.get("name", "")) for m in (marks or [])]
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 60 or h < 50:
            return
        pad_l, pad_r, pad_t, pad_b = 46, 12, 22, 26
        pts = self.series["train"] + self.series["val"]
        if not pts:
            self.create_text(w // 2, h // 2, fill=MUTED,
                             text="학습을 실행하면 손실 곡선이 그려집니다")
            return
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(max(xs), min(xs) + 1)
        y0, y1 = min(ys), max(ys)
        if y1 - y0 < 1e-9:
            y0, y1 = y0 - 0.1, y1 + 0.1
        m = (y1 - y0) * 0.12
        y0, y1 = y0 - m, y1 + m
        sx = lambda x: pad_l + (x - x0) / (x1 - x0) * (w - pad_l - pad_r)
        sy = lambda y: pad_t + (1 - (y - y0) / (y1 - y0)) * (h - pad_t - pad_b)
        for i in range(5):
            yv = y0 + (y1 - y0) * i / 4; yy = sy(yv)
            self.create_line(pad_l, yy, w - pad_r, yy, fill="#eef0f3")
            self.create_text(pad_l - 6, yy, text=f"{yv:.2f}", anchor="e",
                             fill=MUTED, font=("TkDefaultFont", 7))
        self.create_line(pad_l, pad_t, pad_l, h - pad_b, fill=LINE)
        self.create_line(pad_l, h - pad_b, w - pad_r, h - pad_b, fill=LINE)
        for ep, lbl in self.marks:
            xx = sx(ep)
            self.create_line(xx, pad_t, xx, h - pad_b, fill=WARN, dash=(3, 2))
            self.create_text(xx, pad_t, text=lbl, anchor="n", fill=WARN,
                             font=("TkDefaultFont", 7))
        for key, color, dash in (("train", "#4b7bec", ()), ("val", ACCENT, (4, 3))):
            data = self.series[key]
            if len(data) >= 2:
                flat = []
                for x, y in data:
                    flat += [sx(x), sy(y)]
                self.create_line(*flat, fill=color, width=2, dash=dash or None)
            for x, y in data:
                self.create_oval(sx(x) - 2, sy(y) - 2, sx(x) + 2, sy(y) + 2,
                                 fill=color, outline="")


class ConfusionMatrix(tk.Canvas):
    def __init__(self, master, on_select=None, **kw):
        super().__init__(master, bg=CARD, highlightthickness=1,
                         highlightbackground=LINE, **kw)
        self.matrix, self.classes, self.on_select, self.cells = None, [], on_select, {}
        self.bind("<Button-1>", self._click)
        self.bind("<Configure>", lambda e: self.redraw())

    def load_from_arrays(self, logits, labels, classes):
        if np is None or logits is None:
            return
        preds = logits.argmax(axis=1)
        n = len(classes)
        m = np.zeros((n, n), dtype=int)
        for t, p in zip(labels, preds):
            if int(t) < n and int(p) < n:
                m[int(t), int(p)] += 1
        self.matrix, self.classes = m, classes
        self.redraw()

    def clear(self):
        self.matrix = None
        self.redraw()

    def _click(self, ev):
        for (r, c), (a, b, cc, d) in self.cells.items():
            if a <= ev.x <= cc and b <= ev.y <= d and self.on_select:
                row = self.matrix[r]
                self.on_select(self.classes[r], self.classes[c],
                               int(self.matrix[r, c]), int(row.sum()))
                return

    def redraw(self):
        self.delete("all"); self.cells = {}
        w, h = self.winfo_width(), self.winfo_height()
        if self.matrix is None or w < 60 or h < 60:
            if w > 60 and h > 60:
                self.create_text(w // 2, h // 2, fill=MUTED,
                                 text="학습이 끝나면 혼동행렬이 표시됩니다")
            return
        n = len(self.classes); pad = 62
        size = min((w - pad - 8) / n, (h - pad - 8) / n)
        maxv = max(1, self.matrix.max())
        for r in range(n):
            for c in range(n):
                x0, y0 = pad + c * size, pad + r * size
                v = self.matrix[r, c]; t = (v / maxv) ** 0.6
                shade = int(255 - t * 175)
                color = (f"#{shade:02x}{max(shade-10,40):02x}{shade:02x}" if r != c
                         else f"#{shade:02x}{255-int(t*90):02x}{shade:02x}")
                self.create_rectangle(x0, y0, x0 + size, y0 + size, fill=color,
                                      outline="#ffffff")
                if size > 22 and v:
                    self.create_text(x0 + size / 2, y0 + size / 2, text=str(v),
                                     fill=INK if t < 0.55 else "#ffffff",
                                     font=("TkDefaultFont", 7))
                self.cells[(r, c)] = (x0, y0, x0 + size, y0 + size)
            self.create_text(pad - 5, pad + r * size + size / 2, anchor="e",
                             text=str(self.classes[r])[:6], fill=MUTED,
                             font=("TkDefaultFont", 7))
            self.create_text(pad + r * size + size / 2, pad - 6, anchor="s",
                             text=str(self.classes[r])[:5], fill=MUTED, angle=45,
                             font=("TkDefaultFont", 7))
        self.create_text(6, pad / 2, anchor="w", text="↓실제 / →예측", fill=MUTED,
                         font=("TkDefaultFont", 7))


# ==========================================================================
# 메인 앱
# ==========================================================================
class LabApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LeafScan Lab 1.0 — 멀티헤드 실험 콘솔 (PyTorch)")
        self.geometry("1320x880")
        self.minsize(1120, 720)
        self.configure(bg=BG)
        self.font = pick_font()
        self._style()
        self.proc = None
        self.q = queue.Queue()
        self.baseline = None
        self.current_run = None
        self.log_fp = None
        self.head_data = []   # [(head, logits, labels, names)]
        self.current_metrics = None
        # Study 실행 상태
        self.is_study = False        # 현재 돌고 있는 프로세스가 run_study.py 인가
        self.study_dir = None
        self.study_order = []        # 실행 순서대로의 arch 목록
        self.study_state = {}        # arch → STUDY_STATES 키
        self.study_current = None
        self.study_started = None    # 소요 시간 계산용
        self._study_result = None    # (상태, 폴더, 요약) — 최종 완료 줄에 쓴다
        RUNS.mkdir(exist_ok=True)
        self._build_vars()
        self._build_ui()
        self._refresh_runs()
        self.after(80, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _style(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        f = self.font
        s.configure(".", background=BG, foreground=INK, font=(f, 9))
        s.configure("TFrame", background=BG)
        s.configure("Card.TFrame", background=CARD, relief="solid", borderwidth=1)
        s.configure("TLabel", background=BG, foreground=INK, font=(f, 9))
        s.configure("H1.TLabel", font=(f, 13, "bold"))
        s.configure("Sec.TLabel", font=(f, 9, "bold"), foreground=ACCENT)
        s.configure("Hint.TLabel", foreground=MUTED, font=(f, 8))
        s.configure("CardTitle.TLabel", background=CARD, foreground=MUTED, font=(f, 8))
        s.configure("CardValue.TLabel", background=CARD, foreground=INK, font=(f, 16, "bold"))
        s.configure("CardDelta.TLabel", background=CARD, foreground=MUTED, font=(f, 8))
        s.configure("TButton", font=(f, 9), padding=5)
        s.configure("Run.TButton", font=(f, 10, "bold"), padding=8)
        s.configure("Toolbutton", font=(f, 8), padding=3)
        s.configure("TCheckbutton", background=BG, font=(f, 9))
        s.configure("TLabelframe", background=BG)
        s.configure("TLabelframe.Label", background=BG, foreground=ACCENT, font=(f, 9, "bold"))
        s.configure("TNotebook", background=BG)

    def _build_vars(self):
        # DEFAULTS 하나만 읽는다 → "시작 상태 == 초기화 결과" 가 코드로 보장된다
        self.v = {k: kind(value=val) for k, (kind, val) in DEFAULTS.items()}
        self.hypothesis = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="유휴")
        self.lr_text = tk.StringVar(value="1.0e-03")
        self.est_text = tk.StringVar(value="")
        self.cell_text = tk.StringVar(value="혼동행렬의 칸을 클릭하면 상세가 표시됩니다")
        self.baseline_var = tk.StringVar(value="(없음)")
        # 실행 모드 — 기본은 단일 실행 (기존 동작)
        self.mode = tk.StringVar(value=DEFAULT_MODE)
        self.study_name = tk.StringVar(value=DEFAULT_STUDY_NAME)
        self.study_lr_search = tk.BooleanVar(value=False)
        self.study_pick = {a: tk.BooleanVar(value=False) for a in ARCHS}
        self.study_pick_text = tk.StringVar(value="선택 0개 — 최소 2개")
        self.v["lr_log"].trace_add("write", lambda *_: self._on_lr())
        self.v["arch"].trace_add("write", lambda *_: self._on_arch())
        for k in ("epochs", "batch_size", "subset", "dataset", "arch", "img_size"):
            self.v[k].trace_add("write", lambda *_: self._estimate())
        self.mode.trace_add("write", lambda *_: self._on_mode())
        for var in self.study_pick.values():
            var.trace_add("write", lambda *_: self._on_pick())

    def lr(self):
        return 10 ** float(self.v["lr_log"].get())

    def lr_finetune(self):
        return 10 ** float(self.v["lr_finetune_log"].get())

    def _on_lr(self):
        self.lr_text.set(f"{self.lr():.1e}")

    def _on_arch(self):
        # Study 모드에서는 특정 모델 전용 설정을 아예 노출하지 않는다
        is_simple = (self.v["arch"].get() == "simple_cnn"
                     and not self.study_mode())
        if hasattr(self, "simple_frame"):
            if is_simple:
                self.simple_frame.pack(fill="x", pady=(4, 0))
            else:
                self.simple_frame.pack_forget()
        self._estimate()

    def study_mode(self):
        return self.mode.get() == "Study 실행"

    def picked_archs(self):
        """체크된 모델을 ARCHS 표시 순서대로 반환 (= 실행 순서)."""
        return [a for a in ARCHS if self.study_pick[a].get()]

    def _estimate(self):
        try:
            ds = self.v["dataset"].get()
            n = self.v["subset"].get() or (177 if ds == "index_csv"
                                           else 50000 if ds == "cifar10" else 1024)
            steps = max(1, int(n * (1 - self.v["val_ratio"].get())
                               / max(1, self.v["batch_size"].get())))
            size_factor = (self.v["img_size"].get() / 32) ** 2
            if self.study_mode():
                picked = self.picked_archs()
                cost = sum(ARCH_COST.get(a, 1.0) for a in picked)
                total = steps * 0.010 * cost * size_factor * self.v["epochs"].get()
                if self.study_lr_search.get():
                    total *= 1.9        # 3점 × 예산 30% ≒ +90%
                self.est_text.set(
                    f"예상 소요 약 {total/60:.1f}분 · 모델 {len(picked)}개 순차"
                    f"{' · LR 탐색 포함' if self.study_lr_search.get() else ''}"
                    " (CPU 러프 추정)")
                return
            per = steps * 0.010 * ARCH_COST.get(self.v["arch"].get(), 1.0) * size_factor
            total = per * self.v["epochs"].get()
            self.est_text.set(f"예상 소요 약 {total/60:.1f}분 (CPU 러프 추정)")
        except Exception:                                      # noqa: BLE001
            self.est_text.set("")

    # --------------------------------------------------------- 모드 전환
    def _on_pick(self):
        n = len(self.picked_archs())
        self.study_pick_text.set(
            f"선택 {n}개" + (" — 최소 2개" if n < 2 else ""))
        self._estimate()

    def _on_mode(self):
        """단일 ↔ Study 전환.

        쓸 수 없는 항목은 회색 비활성이 아니라 **숨기고**, 그 자리에 이유만 남긴다.
        (회색 처리는 readonly 콤보·일반 입력과 구분이 안 된다는 피드백)
        공통 학습 설정값은 건드리지 않는다 (SRS §3.1.4).
        """
        if not hasattr(self, "study_frame"):
            return
        study = self.study_mode()
        if study:
            # ② 모델 구조 — 단일 arch 선택을 숨기고 안내로 대체
            self.arch_row.pack_forget()
            self.arch_hint.pack(fill="x", pady=(0, 2), before=self.imgsize_row)
            # ④ 실행 — Study 설정을 띄우고 가설 입력을 숨긴다
            self.study_frame.pack(fill="x", pady=(0, 6), before=self.hypo_label)
            self.hypo_label.pack_forget()
            self.hypo_text.pack_forget()
            self.hypo_hint.pack(fill="x", pady=(0, 6), before=self.run_btn)
            self.run_btn.config(text="▶  Study 큐 시작")
        else:
            self.arch_hint.pack_forget()
            self.arch_row.pack(fill="x", before=self.imgsize_row)
            self.study_frame.pack_forget()
            self.hypo_hint.pack_forget()
            self.hypo_label.pack(anchor="w", before=self.run_btn)
            self.hypo_text.pack(fill="x", pady=(2, 6), before=self.run_btn)
            self.run_btn.config(text="▶  학습 실행")
        self._sync_result_buttons(study)
        self._on_arch()      # simple_cnn 프레임 표시 규칙 재적용 + 예상시간 갱신

    def _sync_result_buttons(self, study):
        """결과 영역 — 모드에서 쓰지 않는 버튼·입력은 숨긴다.

        side='right' 라 패킹 순서가 곧 오른쪽부터의 배치 순서다.
        Study 모드에서는 결론 저장이 없으므로 결론 입력칸도 함께 숨기고,
        그만큼 로그 영역이 넓어진다.
        """
        for btn in (self.study_md_btn, self.report_btn, self.concl_btn):
            btn.pack_forget()
        if study:
            self.study_md_btn.pack(side="right")
            self.concl_text.pack_forget()
            self.concl_label.config(text="Study 결과 (결론은 STUDY.md 에 작성)")
        else:
            self.report_btn.pack(side="right")
            self.concl_btn.pack(side="right", padx=(0, 6))
            self.concl_text.pack(fill="x", pady=(2, 6), before=self.log_label)
            self.concl_label.config(text="결론 (결과 해석 — report.md §7 에 반영)")

    # ------------------------------------------------------------------- UI
    def _build_ui(self):
        top = ttk.Frame(self, padding=(12, 10, 12, 6))
        top.pack(fill="x")
        ttk.Label(top, text="LeafScan Lab", style="H1.TLabel").pack(side="left")
        ttk.Label(top, text="1.0", style="Hint.TLabel").pack(side="left", padx=(4, 0))
        ttk.Label(top, text="  멀티헤드 · config 한 줄로 모델 교체",
                  style="Hint.TLabel").pack(side="left", padx=(4, 0))
        ttk.Label(top, textvariable=self.status_text, style="Hint.TLabel").pack(side="right")
        ttk.Label(top, text="기준 run:", style="Hint.TLabel").pack(side="right", padx=(12, 4))
        self.run_combo = ttk.Combobox(top, textvariable=self.baseline_var, width=22,
                                      state="readonly")
        self.run_combo.pack(side="right")
        self.run_combo.bind("<<ComboboxSelected>>", lambda e: self._set_baseline())
        ttk.Button(top, text="이 run 보기", command=self._view_selected).pack(
            side="right", padx=(0, 6))

        body = ttk.Frame(self, padding=(12, 0, 12, 12))
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, width=380)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        self._build_left(self._scrollable(left))
        self._build_right(right)

    def _scrollable(self, parent):
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0, width=360)
        bar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        win = canvas.create_window((0, 0), window=inner, anchor="nw", width=356)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width - 2))

        # 휠 이벤트는 커서 아래 위젯으로만 가고 부모로 전파되지 않는다.
        # 사이드바는 자식 위젯으로 덮여 있으므로 전역(all 태그)으로 받은 뒤
        # 커서가 이 canvas 안에 있을 때만 처리한다.
        def wheel(ev):
            if not self._pointer_in(canvas, ev):
                return
            if getattr(ev, "num", 0) in (4, 5):        # X11
                step = -1 if ev.num == 4 else 1
            else:                                      # Windows / macOS
                step = -1 if ev.delta > 0 else 1
            canvas.yview_scroll(step, "units")
        canvas.bind_all("<MouseWheel>", wheel, add="+")
        canvas.bind_all("<Button-4>", wheel, add="+")
        canvas.bind_all("<Button-5>", wheel, add="+")
        return inner

    def _pointer_in(self, target, ev):
        """커서(ev)가 target 위젯 또는 그 자식 위에 있는가."""
        try:
            w = self.winfo_containing(ev.x_root, ev.y_root)
        except KeyError:                               # Tk가 모르는 창 위
            return False
        while w is not None:
            if w is target:
                return True
            w = getattr(w, "master", None)
        return False

    def _seg(self, parent, var, values, label=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(4, 0))
        if label:
            ttk.Label(row, text=label, width=13).pack(side="left")
        for val in values:
            ttk.Radiobutton(row, text=str(val), value=val, variable=var,
                            style="Toolbutton").pack(side="left", padx=1)
        return row

    def _build_left(self, p):
        # 실행 모드 — 사이드바 최상단
        self.mode_row = self._seg(p, self.mode, ["단일 실행", "Study 실행"], "실행 모드")
        self.mode_row.pack(fill="x", pady=(0, 6))
        ttk.Separator(p, orient="horizontal").pack(fill="x", pady=(0, 4))

        # ① 데이터
        d = ttk.Labelframe(p, text="① 데이터", padding=8)
        d.pack(fill="x", pady=4)
        self._seg(d, self.v["dataset"],
                  ["index_csv", "multilabel_fake", "cifar10", "fake"], "데이터셋")
        self._slider(d, "사용 샘플 수", self.v["subset"], 0, 20000, 500,
                     lambda x: "전체" if x == 0 else f"{int(x):,}장")
        self._slider(d, "검증 비율", self.v["val_ratio"], 0.1, 0.5, 0.05,
                     lambda x: f"val {x*100:.0f}%")
        adv_d = ttk.Labelframe(d, text="고급", padding=6)
        adv_d.pack(fill="x", pady=(6, 0))
        self._seg(adv_d, self.v["split_policy"],
                  ["respect_provided", "resplit_all"], "분할정책")
        self._seg(adv_d, self.v["crop_mode"], ["full", "bbox", "bbox_expand"], "crop")
        self._seg(adv_d, self.v["ambiguous_policy"],
                  ["exclude", "downweight", "include"], "ambiguous")
        ttk.Checkbutton(adv_d, text="Augmentation", variable=self.v["augment"]).pack(
            anchor="w", pady=(4, 0))

        # ② 모델
        m = ttk.Labelframe(p, text="② 모델 구조", padding=8)
        m.pack(fill="x", pady=4)
        arow = ttk.Frame(m)
        arow.pack(fill="x")
        self.arch_row = arow
        ttk.Label(arow, text="arch", width=13).pack(side="left")
        self.arch_combo = ttk.Combobox(arow, textvariable=self.v["arch"], values=ARCHS,
                                       state="readonly", width=20)
        self.arch_combo.pack(side="left")
        self.arch_hint = ttk.Label(
            m, text="arch — Study 모드에서는 비교 모델 목록으로 모델을 선택합니다. "
                    "(④ 실행 › Study 설정)",
            style="Hint.TLabel", wraplength=330)
        self.imgsize_row = self._seg(m, self.v["img_size"],
                                     [32, 64, 128, 192, 224], "입력 해상도")
        ttk.Checkbutton(m, text="pretrained (ImageNet 가중치)",
                        variable=self.v["pretrained"]).pack(anchor="w", pady=(4, 0))
        self._slider(m, "freeze epoch", self.v["freeze_epochs"], 0, 10, 1,
                     lambda x: "동결 없음" if x == 0 else f"{int(x)}epoch 백본 동결")
        ttk.Label(m, text="헤드 구성은 데이터에서 자동 (crop·stage)",
                  style="Hint.TLabel").pack(anchor="w")
        # simple_cnn 전용 (arch=simple_cnn 일 때만 노출)
        self.simple_frame = ttk.Labelframe(m, text="simple_cnn 파라미터", padding=6)
        self._seg(self.simple_frame, self.v["conv_channels"], [8, 16, 32, 64], "채널")
        self._seg(self.simple_frame, self.v["conv_blocks"], [1, 2, 3], "블록 수")
        self._seg(self.simple_frame, self.v["kernel_size"], [3, 5], "커널")

        # ③ 학습
        t = ttk.Labelframe(p, text="③ 학습 설정", padding=8)
        t.pack(fill="x", pady=4)
        self._slider(t, "Epoch", self.v["epochs"], 1, 100, 1, lambda x: f"{int(x)} epoch")
        self._seg(t, self.v["batch_size"], [8, 16, 32, 64, 128], "배치")
        row = ttk.Frame(t)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text="Learning rate", width=13).pack(side="left")
        ttk.Label(row, textvariable=self.lr_text, foreground=ACCENT).pack(side="left")
        ttk.Scale(t, from_=-5, to=-1, variable=self.v["lr_log"],
                  orient="horizontal").pack(fill="x")
        self._seg(t, self.v["optimizer"], ["SGD", "Adam", "AdamW"], "Optimizer")
        self._seg(t, self.v["class_weight"], ["none", "auto", "manual"], "class weight")
        wrow = ttk.Frame(t)
        wrow.pack(fill="x", pady=(6, 0))
        ttk.Label(wrow, text="head 가중치", width=13).pack(side="left")
        ttk.Label(wrow, text="crop").pack(side="left")
        ttk.Spinbox(wrow, from_=0, to=1, increment=0.1, width=4,
                    textvariable=self.v["w_crop"]).pack(side="left", padx=(2, 8))
        ttk.Label(wrow, text="stage").pack(side="left")
        ttk.Spinbox(wrow, from_=0, to=1, increment=0.1, width=4,
                    textvariable=self.v["w_stage"]).pack(side="left", padx=2)
        adv_t = ttk.Labelframe(t, text="고급", padding=6)
        adv_t.pack(fill="x", pady=(6, 0))
        self._slider(adv_t, "Early stop", self.v["patience"], 0, 20, 1,
                     lambda x: "없음" if x == 0 else f"{int(x)}회 미개선 중단")
        srow = ttk.Frame(adv_t)
        srow.pack(fill="x", pady=(4, 0))
        ttk.Label(srow, text="seed", width=13).pack(side="left")
        ttk.Spinbox(srow, from_=0, to=9999, width=6,
                    textvariable=self.v["seed"]).pack(side="left")

        # ④ 실행 — 가설 입력(실행 전 강제)
        r = ttk.Labelframe(p, text="④ 실행", padding=8)
        r.pack(fill="x", pady=(8, 0))
        self._build_study_frame(r)
        self.hypo_label = ttk.Label(r, text="가설 (왜 이 실험을 하는가 — 실행 전 작성)",
                                    style="Hint.TLabel")
        self.hypo_label.pack(anchor="w")
        self.hypo_hint = ttk.Label(
            r, text="가설 — Study 이름과 결과표(STUDY.md)가 실험 묶음을 식별합니다.",
            style="Hint.TLabel", wraplength=330)
        self.hypo_text = tk.Text(r, height=3, font=(self.font, 9), wrap="word")
        self.hypo_text.pack(fill="x", pady=(2, 6))
        self.run_btn = ttk.Button(r, text="▶  학습 실행", style="Run.TButton",
                                  command=self.start_run)
        self.run_btn.pack(fill="x")
        self.stop_btn = ttk.Button(r, text="■  중단", command=self.stop_training,
                                   state="disabled")
        self.stop_btn.pack(fill="x", pady=(4, 0))
        self.reset_btn = ttk.Button(r, text="↺  기본값 초기화",
                                    command=self.reset_to_defaults)
        self.reset_btn.pack(fill="x", pady=(4, 0))
        ttk.Label(r, textvariable=self.est_text, style="Hint.TLabel").pack(
            anchor="w", pady=(4, 0))
        self._estimate(); self._on_lr(); self._on_arch(); self._on_pick()

    def _build_study_frame(self, parent):
        """Study 전용 설정 — 기존 simple_frame 과 같은 조건부 pack 방식."""
        f = ttk.Labelframe(parent, text="Study 설정", padding=6)
        self.study_frame = f

        nrow = ttk.Frame(f)
        nrow.pack(fill="x")
        ttk.Label(nrow, text="Study 이름", width=11).pack(side="left")
        ttk.Entry(nrow, textvariable=self.study_name).pack(
            side="left", fill="x", expand=True)
        ttk.Label(f, text="같은 이름으로 다시 실행하면 완료된 모델은 건너뜁니다.",
                  style="Hint.TLabel", wraplength=320).pack(anchor="w", pady=(2, 0))

        prow = ttk.Frame(f)
        prow.pack(fill="x", pady=(6, 0))
        ttk.Label(prow, text="비교 모델", style="Sec.TLabel").pack(side="left")
        ttk.Label(prow, textvariable=self.study_pick_text,
                  style="Hint.TLabel").pack(side="left", padx=(6, 0))
        grid = ttk.Frame(f)
        grid.pack(fill="x", pady=(2, 0))
        self.study_checks = {}
        for i, arch in enumerate(ARCHS):
            cb = ttk.Checkbutton(grid, text=arch, variable=self.study_pick[arch])
            cb.grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 6))
            self.study_checks[arch] = cb
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        ttk.Label(f, text="simple_cnn 은 하네스 검증용 fixture 입니다 "
                          "(실험 대상 아님).",
                  style="Hint.TLabel", wraplength=320).pack(anchor="w", pady=(2, 0))

        self.lrsearch_check = ttk.Checkbutton(
            f, text="모델별 LR 자동 탐색", variable=self.study_lr_search,
            command=self._estimate)
        self.lrsearch_check.pack(anchor="w", pady=(6, 0))
        ttk.Label(f, text="모델마다 후보 3점을 짧게 학습해 lr 을 고릅니다. "
                          "실행 시간이 크게 늘어납니다.",
                  style="Hint.TLabel", wraplength=320).pack(anchor="w")

        ttk.Label(f, text="큐 상태", style="Sec.TLabel").pack(anchor="w", pady=(8, 0))
        self.queue_box = ttk.Frame(f)
        self.queue_box.pack(fill="x", pady=(2, 0))
        self.queue_labels = {}
        self._rebuild_queue([])
        ttk.Button(f, text="Study 폴더 열기", command=self._open_study_dir).pack(
            fill="x", pady=(6, 0))

    def _rebuild_queue(self, archs):
        for w in self.queue_box.winfo_children():
            w.destroy()
        self.queue_labels = {}
        if not archs:
            ttk.Label(self.queue_box, text="시작하면 모델별 진행 상태가 표시됩니다",
                      style="Hint.TLabel").pack(anchor="w")
            return
        for arch in archs:
            lab = ttk.Label(self.queue_box, text=f"○  {arch} · 대기",
                            foreground=MUTED)
            lab.pack(anchor="w")
            self.queue_labels[arch] = lab

    def _set_member_state(self, arch, state):
        if not arch or arch not in self.study_state:
            return
        self.study_state[arch] = state
        mark, text, color = STUDY_STATES.get(state, STUDY_STATES["pending"])
        lab = self.queue_labels.get(arch)
        if lab is not None:
            lab.config(text=f"{mark}  {arch} · {text}", foreground=color)

    def _slider(self, parent, label, var, lo, hi, step, fmt):
        box = ttk.Frame(parent)
        box.pack(fill="x", pady=(6, 0))
        head = ttk.Frame(box)
        head.pack(fill="x")
        ttk.Label(head, text=label, width=13).pack(side="left")
        val_lbl = ttk.Label(head, text="", foreground=ACCENT)
        val_lbl.pack(side="left")
        sc = ttk.Scale(box, from_=lo, to=hi, orient="horizontal")
        guard = {"busy": False}

        def snap(raw):
            v = max(lo, min(hi, round(raw / step) * step))
            return int(round(v)) if isinstance(var, tk.IntVar) else round(v, 4)

        def on_move(_=None):
            if guard["busy"]:
                return
            guard["busy"] = True
            v = snap(sc.get()); var.set(v); val_lbl.config(text=fmt(v))
            guard["busy"] = False

        def on_var(*_):
            if guard["busy"]:
                return
            guard["busy"] = True
            try:
                v = var.get(); sc.set(v); val_lbl.config(text=fmt(v))
            finally:
                guard["busy"] = False
        sc.set(var.get()); sc.config(command=on_move); sc.pack(fill="x")
        val_lbl.config(text=fmt(var.get())); var.trace_add("write", on_var)
        return sc

    # -- 우: 눈 --------------------------------------------------------
    def _build_right(self, p):
        cards = ttk.Frame(p)
        cards.pack(fill="x")
        self.cards = {
            "stage_macro_f1": MetricCard(cards, "stage macro-F1", "{:.3f}", True),
            "primary_recall": MetricCard(cards, "정식기 recall", "{:.3f}", True),
            "crop_acc": MetricCard(cards, "crop accuracy", "{:.3f}", True),
            "abstain_rate": MetricCard(cards, "미판정률", "{:.3f}", False),
        }
        for c in self.cards.values():
            c.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.prog = ttk.Progressbar(p, mode="determinate")
        self.prog.pack(fill="x", pady=(10, 0))

        mid = ttk.Frame(p)
        mid.pack(fill="both", expand=True, pady=(10, 0))
        lf = ttk.Frame(mid)
        lf.pack(side="left", fill="both", expand=True)
        ttk.Label(lf, text="손실 곡선", style="Sec.TLabel").pack(anchor="w")
        self.chart = LossChart(lf, height=210)
        self.chart.pack(fill="both", expand=True, pady=(2, 0))

        rf = ttk.Frame(mid, width=340)
        rf.pack(side="left", fill="both", padx=(10, 0))
        rf.pack_propagate(False)
        ttk.Label(rf, text="혼동행렬 (head 탭)", style="Sec.TLabel").pack(anchor="w")
        # 혼동행렬은 탭 **안**에 그린다 (head 마다 캔버스 하나)
        self.head_tabs = ttk.Notebook(rf)
        self.head_tabs.pack(fill="both", expand=True, pady=(2, 0))
        self.head_tabs.bind("<<NotebookTabChanged>>", lambda e: self._on_head_tab())
        self.cms = {}
        self.cm = None
        self._reset_head_tabs()
        ttk.Label(rf, textvariable=self.cell_text, style="Hint.TLabel",
                  wraplength=330).pack(anchor="w", pady=(4, 0))

        # 결론 + 리포트 열기
        cf = ttk.Frame(p)
        cf.pack(fill="x", pady=(8, 0))
        self.concl_label = ttk.Label(cf, text="결론 (결과 해석 — report.md §7 에 반영)",
                                     style="Sec.TLabel")
        self.concl_label.pack(side="left")
        # Study 모드 전용 — 단일 모드에서는 숨긴다 (_sync_result_buttons)
        self.study_md_btn = ttk.Button(cf, text="STUDY.md 열기",
                                       command=self._open_study_md)
        self.report_btn = ttk.Button(cf, text="report.md 열기", command=self._open_report)
        self.concl_btn = ttk.Button(cf, text="결론 저장", command=self._save_conclusion)
        self.concl_text = tk.Text(p, height=3, font=(self.font, 9), wrap="word")
        self.concl_text.pack(fill="x", pady=(2, 6))

        self.log_label = ttk.Label(p, text="로그", style="Sec.TLabel")
        self.log_label.pack(anchor="w")
        wrap = ttk.Frame(p, style="Card.TFrame")
        wrap.pack(fill="both", expand=True)
        self.log = tk.Text(wrap, height=8, bg="#101418", fg="#d7dde3", relief="flat",
                           font=("Consolas" if os.name == "nt" else "TkFixedFont", 9),
                           wrap="none")
        sb = ttk.Scrollbar(wrap, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set, state="disabled")
        sb.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)
        self._sync_result_buttons(self.study_mode())

    def _on_cell(self, true_c, pred_c, v, row_total):
        pct = 100 * v / row_total if row_total else 0
        kind = "정답" if true_c == pred_c else "오분류"
        self.cell_text.set(f"실제 [{true_c}] → 예측 [{pred_c}] : {v}장 "
                           f"({kind}, 실제 클래스의 {pct:.1f}%)")

    def _add_head_tab(self, text):
        """탭 하나 = 프레임 + 그 안을 채우는 혼동행렬 캔버스."""
        frame = ttk.Frame(self.head_tabs, padding=2)
        self.head_tabs.add(frame, text=text)
        cm = ConfusionMatrix(frame, on_select=self._on_cell)
        cm.pack(fill="both", expand=True)
        return cm

    def _reset_head_tabs(self):
        """탭을 비우고 안내용 빈 혼동행렬 하나만 남긴다."""
        for tab in self.head_tabs.tabs():
            self.head_tabs.forget(tab)
        self.cms = {}
        self.head_data = []
        self.cm = self._add_head_tab("결과")

    def _on_head_tab(self):
        if not self.head_data:
            return
        idx = self.head_tabs.index("current")
        if idx < len(self.head_data):
            head, lg, lb, names = self.head_data[idx]
            cm = self.cms.get(head)
            if cm is None:
                return
            self.cm = cm
            if np is not None and cm.matrix is None:   # 처음 볼 때만 로드
                names = names or [f"{head}_{i}" for i in range(np.load(lg).shape[1])]
                cm.load_from_arrays(np.load(lg), np.load(lb), names)

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        if self.log_fp:
            self.log_fp.write(text.rstrip() + "\n"); self.log_fp.flush()

    # ----------------------------------------------------------- run 관리
    def _refresh_runs(self):
        runs = sorted([p.name for p in RUNS.glob("*")
                       if (p / "metrics.json").exists()])
        self.run_combo["values"] = ["(없음)"] + runs
        if self.baseline_var.get() not in self.run_combo["values"]:
            self.baseline_var.set("(없음)")

    def _set_baseline(self):
        name = self.baseline_var.get()
        if name == "(없음)":
            self.baseline = None
        else:
            try:
                self.baseline = json.loads(
                    (RUNS / name / "metrics.json").read_text(encoding="utf-8"))
                self._log(f"[기준 run] {name} 선택")
            except Exception as e:                             # noqa: BLE001
                self.baseline = None
                self._log(f"[경고] 기준 run 로드 실패: {e}")
        self._refresh_cards(self.current_metrics)

    def _view_selected(self):
        """선택한 run(구·신 버전 모두)을 메인 뷰로 불러온다."""
        name = self.baseline_var.get()
        if name == "(없음)":
            return
        run = RUNS / name
        if (run / "metrics.json").exists():
            self.current_run = run
            try:
                self._load_run(run)
                self._log(f"[보기] {name} 로드")
            except Exception as e:                             # noqa: BLE001
                self._log(f"[경고] {name} 로드 실패: {e}")

    def _next_run_dir(self):
        i = 1
        while (RUNS / f"exp_{i:03d}").exists():
            i += 1
        d = RUNS / f"exp_{i:03d}"
        d.mkdir(parents=True)
        return d

    def build_config(self):
        cfg = dict(
            schema_version=2,
            dataset=self.v["dataset"].get(),
            data_root=self.v["data_root"].get(),
            index_csv=self.v["index_csv"].get(),
            subset=int(self.v["subset"].get()),
            val_ratio=float(self.v["val_ratio"].get()),
            split_policy=self.v["split_policy"].get(),
            ambiguous_policy=self.v["ambiguous_policy"].get(),
            crop_mode=self.v["crop_mode"].get(),
            augment=bool(self.v["augment"].get()),
            arch=self.v["arch"].get(),
            img_size=int(self.v["img_size"].get()),
            pretrained=bool(self.v["pretrained"].get()),
            freeze_epochs=int(self.v["freeze_epochs"].get()),
            epochs=int(self.v["epochs"].get()),
            batch_size=int(self.v["batch_size"].get()),
            lr=self.lr(),
            lr_finetune=self.lr_finetune(),
            optimizer=self.v["optimizer"].get(),
            class_weight=self.v["class_weight"].get(),
            class_weight_head="stage",
            head_weights={"crop": float(self.v["w_crop"].get()),
                          "stage": float(self.v["w_stage"].get())},
            patience=int(self.v["patience"].get()),
            seed=int(self.v["seed"].get()),
            num_workers=0,
            device="auto",
            hypothesis=self.hypo_text.get("1.0", "end").strip(),
            tags=[],
            study=None,
            parent_run=(self.baseline_var.get()
                        if self.baseline_var.get() != "(없음)" else None),
        )
        if self.v["arch"].get() == "simple_cnn":
            cfg.update(conv_channels=int(self.v["conv_channels"].get()),
                       kernel_size=int(self.v["kernel_size"].get()),
                       stride=int(self.v["stride"].get()), padding=1,
                       pool_kernel=int(self.v["pool_kernel"].get()),
                       pool_stride=int(self.v["pool_stride"].get()),
                       conv_blocks=int(self.v["conv_blocks"].get()),
                       use_relu=bool(self.v["use_relu"].get()))
        return cfg

    # ------------------------------------------------- 기본값 초기화
    def reset_to_defaults(self, confirm=True):
        """편집 중인 설정만 기본값으로 되돌린다. 결과 파일은 건드리지 않는다."""
        if self.proc and self.proc.poll() is None:
            return                      # 실행 중에는 버튼도 비활성이지만 이중 방어
        if confirm and not messagebox.askokcancel(
                "기본값 초기화",
                "현재 편집 중인 설정을 기본값으로 되돌릴까요?\n\n"
                "생성된 실험 결과와 파일은 삭제되지 않습니다."):
            return
        for key, (_kind, val) in DEFAULTS.items():
            self.v[key].set(val)
        # 실행 모드와 Study 설정도 '편집 중인 설정'이다
        self.mode.set(DEFAULT_MODE)
        self.study_name.set(DEFAULT_STUDY_NAME)
        self.study_lr_search.set(False)
        for var in self.study_pick.values():
            var.set(False)
        # 기준 run 은 config.json 의 parent_run 으로 들어가므로 선택도 되돌린다.
        # (run 폴더와 결과 파일은 그대로 둔다)
        self.baseline_var.set("(없음)")
        self.baseline = None
        for text in (self.hypo_text, self.concl_text):
            state = str(text["state"])
            text.config(state="normal")
            text.delete("1.0", "end")
            text.config(state=state)
        # 표시 갱신
        self._on_lr(); self._on_arch(); self._on_pick(); self._on_mode()
        self._refresh_cards(self.current_metrics)
        self._estimate()
        self._log("[완료] GUI 설정을 기본값으로 초기화했습니다.")

    def start_run(self):
        """실행 버튼 — 선택된 모드로 분기한다."""
        if self.study_mode():
            self.start_study()
        else:
            self.start_training()

    def start_training(self):
        if self.proc and self.proc.poll() is None:
            return
        if not self.hypo_text.get("1.0", "end").strip():
            if not messagebox.askokcancel(
                    "가설 미작성", "가설이 비어 있습니다. 연구 기록을 위해 작성을 권장합니다.\n"
                    "그래도 실행할까요?"):
                return
        run_dir = self._next_run_dir()
        cfg = self.build_config()
        (run_dir / "config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        self.current_run = run_dir
        self.log_fp = open(run_dir / "train.log", "w", encoding="utf-8")
        self.chart.reset(); self._reset_head_tabs()
        self.prog.config(value=0, maximum=cfg["epochs"])
        self._log(f"[{run_dir.name}] 시작 · arch={cfg['arch']} · config.json 저장")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   PYTHONUTF8="1")
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            [PYTHON, "-u", str(WORKER), "--config",
             str(run_dir / "config.json"), "--out", str(run_dir)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1, env=env, **kw)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.reset_btn.config(state="disabled")   # 실행 중에는 초기화 금지
        self.status_text.set(f"학습 중 · {run_dir.name} · 0%")

    # ------------------------------------------------------------- Study
    def start_study(self):
        """선택 모델을 run_study.py 하나로 순차 실행한다 (병렬 아님)."""
        if self.proc and self.proc.poll() is None:
            return
        ok, name = validate_study_name(self.study_name.get())
        if not ok:
            messagebox.showinfo("Study 이름", name)
            return
        archs = self.picked_archs()
        if len(archs) < 2:
            messagebox.showinfo("비교 모델", "Study 실행에는 모델을 2개 이상 선택하세요.")
            return

        study_dir = RUNS / name
        study_dir.mkdir(parents=True, exist_ok=True)
        base_cfg = build_study_base_config(self.build_config(), name, archs)
        base_path = study_dir / "_gui_base.json"

        # 재개 — 이미 metrics.json 이 있는 모델은 run_study.py 가 건너뛴다.
        # 그런데 건너뛰는 판단은 metrics.json 유무만 보므로, 그 사이 설정이 바뀌면
        # 서로 다른 조건으로 학습된 run 이 한 STUDY.md 표에 섞인다. 미리 막는다.
        done = collect_study_members(study_dir)
        changed = study_config_diff(base_path, base_cfg) if done else []
        if changed:
            lines = "\n".join(f"  · {k}: {old!r} → {new!r}" for k, old, new in changed[:8])
            more = f"\n  … 외 {len(changed) - 8}개" if len(changed) > 8 else ""
            if not messagebox.askokcancel(
                    "설정이 달라졌습니다",
                    f"'{name}' 에는 이미 완료된 run 이 {len(done)}개 있습니다.\n"
                    f"완료된 run 은 건너뛰는데, 지금 설정이 그때와 다릅니다.\n\n"
                    f"{lines}{more}\n\n"
                    "이대로 진행하면 서로 다른 조건으로 학습된 모델이\n"
                    "하나의 STUDY.md 비교표에 섞여 공정 비교가 깨집니다.\n\n"
                    "그래도 계속할까요?\n"
                    "(취소 후 다른 Study 이름을 쓰는 것을 권장합니다)"):
                return
            self._log(f"[경고] {name} — 완료된 run 과 설정이 다릅니다. 비교 해석에 주의:")
            for k, old, new in changed:
                self._log(f"         {k}: {old!r} → {new!r}")

        base_path.write_text(json.dumps(base_cfg, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        self.is_study = True
        self.study_dir = study_dir
        self.study_order = archs
        self.study_current = None
        self.study_started = time.time()
        self.study_state = {a: ("done" if a in done else "pending") for a in archs}
        self._rebuild_queue(archs)
        for a in archs:
            self._set_member_state(a, self.study_state[a])

        self.current_run = None
        self.current_metrics = None
        self.log_fp = open(study_dir / "_gui_study.log", "a", encoding="utf-8")
        self.chart.reset(); self._reset_head_tabs()
        self.prog.config(value=0, maximum=int(self.v["epochs"].get()))

        cmd = [PYTHON, "-u", str(STUDY_RUNNER), "--study", name,
               "--base", str(base_path), "--vary", "arch=" + ",".join(archs)]
        if self.study_lr_search.get():
            cmd.append("--lr-search")
        self._log(f"[Study] {name} 시작 · 모델 {len(archs)}개 순차 "
                  f"({' → '.join(archs)})")
        if done:
            self._log(f"  이미 완료된 run {len(done)}개는 건너뜁니다: "
                      f"{', '.join(sorted(done))}")
        self._log("  " + " ".join(cmd[1:]))

        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   PYTHONUTF8="1")
        kw = {}
        if os.name == "nt":
            # 중단 시 자식 train_worker 까지 함께 종료하기 위한 별도 프로세스 그룹
            kw["creationflags"] = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                                   | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        else:
            kw["start_new_session"] = True
        self.proc = subprocess.Popen(
            cmd, cwd=str(BASE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env, **kw)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.reset_btn.config(state="disabled")   # 실행 중에는 초기화 금지
        self._lock_study_controls(True)
        self.status_text.set(f"Study 0/{len(archs)} · {name}")

    def _lock_study_controls(self, locked):
        """실행 중 설정 변경 차단 — 기존 실행 보호 정책과 동일 (SRS §5)."""
        state = "disabled" if locked else "normal"
        for child in self.mode_row.winfo_children():
            try:
                child.config(state=state)
            except tk.TclError:
                pass
        for cb in self.study_checks.values():
            cb.config(state=state)
        self.lrsearch_check.config(state=state)
        for child in self.study_frame.winfo_children():
            if isinstance(child, ttk.Frame):
                for sub in child.winfo_children():
                    if isinstance(sub, ttk.Entry):
                        sub.config(state=state)

    def _study_line(self, line):
        """run_study.py 로그로 큐 상태를 갱신한다 (Study 모드에서만 호출)."""
        kind, arch = parse_study_line(line)
        if kind is None:
            return
        if kind == "running":
            self._finish_current()
            self.study_current = arch
            self._set_member_state(arch, "running")
            self.chart.reset()          # 모델이 바뀌면 곡선을 새로 그린다
            self.prog.config(value=0)
        elif kind == "lr_search":
            if arch and self.study_state.get(arch) != "running":
                self._set_member_state(arch, "lr_search")
        elif kind == "skipped":
            self._set_member_state(arch, "skipped")
        elif kind == "failed":
            self._set_member_state(arch, "failed")
            if arch == self.study_current:
                self.study_current = None
        elif kind == "finished":
            self._finish_current()

    def _finish_current(self):
        """직전까지 실행 중이던 모델을 완료 처리하고 결과 화면을 갱신한다."""
        arch = self.study_current
        self.study_current = None
        if not arch or self.study_state.get(arch) != "running":
            return
        member = collect_study_members(self.study_dir).get(arch)
        if member is None:
            self._set_member_state(arch, "failed")
            return
        self._set_member_state(arch, "done")
        self.current_run = member
        try:
            self._load_run(member)      # 카드·혼동행렬·곡선을 완료 모델 기준으로
            # run_study.py 의 '[완료] N개 run'(study 전체)과 헷갈리지 않게 구분
            self._log(f"[모델 완료] {arch} · {member.name}")
        except Exception as e:                                 # noqa: BLE001
            self._log(f"  [경고] {arch} 결과 로드 실패: {e}")

    def _study_progress_label(self):
        finished = sum(1 for s in self.study_state.values()
                       if s in ("done", "skipped", "failed"))
        return f"Study {finished}/{len(self.study_order)}"

    def _reader(self, proc):
        for line in proc.stdout:
            self.q.put(line)
        self.q.put({"__exit__": proc.wait()})

    def _kill_tree(self, proc):
        """자식 train_worker 까지 함께 종료. 성공하면 True."""
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True, check=False,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            return True
        except Exception:                                      # noqa: BLE001
            return False

    def stop_training(self):
        if self.proc and self.proc.poll() is None:
            if self.is_study:
                # run_study.py 만 죽이면 자식 train_worker 가 고아로 남는다
                if not self._kill_tree(self.proc):
                    self.proc.terminate()
                self._log("[중단] Study 프로세스 트리에 종료 신호를 보냈습니다")
                return
            self.proc.terminate()
            self._log("[중단] 종료 신호를 보냈습니다")

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, dict):
                    if "__exit__" in item:
                        self._on_exit(item["__exit__"])
                    elif "__study_done__" in item:
                        self._finish_study_log()
                    # "__loaded__" 등 기타 신호는 무시 (이미 처리됨)
                    continue
                line = item.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # run_study.py 자체 로그 — JSON 이 아니다.
                    # 파서를 먼저 돌려야 직전 모델의 [모델 완료] 가
                    # 다음 모델의 [실행] 보다 앞에 찍힌다.
                    if self.is_study:
                        self._study_line(line)
                    self._log("  " + line)
                    continue
                self._dispatch(msg)
        except queue.Empty:
            pass
        self.after(80, self._poll)

    def _dispatch(self, msg):
        ev = msg.get("event")
        if ev == "start":
            self._log(f"  device={msg.get('device')} · arch={msg.get('arch')} "
                      f"· heads={msg.get('heads')}")
        elif ev == "log":
            self._log("  " + msg["message"])
        elif ev == "warn":
            self._log("  ⚠ " + msg.get("message", ""))
        elif ev == "stage":
            self._log(f"  [단계] {msg.get('name')} @epoch {msg.get('epoch')} "
                      f"· 학습가능 {msg.get('trainable_params'):,}")
            self.chart.mark(msg.get("epoch"), msg.get("name", ""))
        elif ev == "epoch":
            e, tot = msg["epoch"], msg["total_epochs"]
            self.prog.config(value=e, maximum=tot)
            self.chart.add(e, msg["loss"], msg["val_loss"])
            ph = msg.get("per_head_acc") or {}
            phs = " ".join(f"{k} {v*100:.0f}%" for k, v in ph.items())
            self._log(f"  epoch {e:>3}/{tot} | loss {msg['loss']:.4f} "
                      f"val_loss {msg['val_loss']:.4f} | {phs}"
                      f"{' ★' if msg.get('improved') else ''}")
            pct = int(e / tot * 100) if tot else 0
            if self.is_study:
                self.status_text.set(f"{self._study_progress_label()} · "
                                     f"{self.study_current or '-'} · {pct}%")
            else:
                self.status_text.set(f"학습 중 · {self.current_run.name} · {pct}%")
        elif ev == "error":
            self._log("[실패] " + msg.get("message", "")
                      + f" (유형: {msg.get('kind','?')})")
            for l in msg.get("traceback", "").splitlines()[-10:]:
                self._log("    " + l)

    def _base_summary(self):
        return normalize_run_summary(self.baseline) if self.baseline else {}

    def _refresh_cards(self, metrics=None):
        s = normalize_run_summary(metrics) if metrics else {}
        b = self._base_summary()
        for key in ("stage_macro_f1", "primary_recall", "crop_acc", "abstain_rate"):
            self.cards[key].update_value(s.get(key), b.get(key))

    def _on_exit(self, code):
        if self.is_study:
            self._on_study_exit(code)
            return
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.reset_btn.config(state="normal")
        run = self.current_run
        if self.log_fp:
            self.log_fp.close(); self.log_fp = None
        if run and (run / "metrics.json").exists():
            self._log(f"[{run.name}] 학습 종료 (exit={code}) · 그림·리포트 생성 중…")
            self._post_process(run)
        else:
            state = "중단됨" if code and code < 0 else "실패"
            self.status_text.set(f"{state} · {run.name if run else '-'}")
            self._log(f"[{state}] exit={code}")
        self._refresh_runs()
        self.proc = None

    def _on_study_exit(self, code):
        """Study 종료 — 멤버 폴더를 실제로 스캔해 최종 상태를 확정한다."""
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.reset_btn.config(state="normal")
        self._lock_study_controls(False)
        study_dir = self.study_dir
        done = collect_study_members(study_dir)
        # 로그 파싱이 놓친 항목을 파일 기준으로 보정
        for arch in self.study_order:
            if arch in done:
                if self.study_state.get(arch) not in ("done", "skipped"):
                    self._set_member_state(arch, "done")
            elif self.study_state.get(arch) in ("running", "lr_search"):
                self._set_member_state(arch, "failed")
        counts = {k: 0 for k in ("done", "skipped", "failed", "pending")}
        for st in self.study_state.values():
            counts[st] = counts.get(st, 0) + 1
        self.is_study = False
        self.study_current = None

        n_done = counts["done"] + counts["skipped"]
        summary = (f"완료 {counts['done']} · 건너뜀 {counts['skipped']} · "
                   f"실패 {counts['failed']} · 대기 {counts['pending']}")
        if counts["pending"] or counts["failed"]:
            state = "중단됨" if counts["pending"] else "일부 실패"
        else:
            state = "완료"
        self._study_result = (state, study_dir, summary)
        self.status_text.set(f"Study {state} · {study_dir.name} · {summary}")
        self._log(f"[Study {state}] exit={code} · {summary}")
        if counts["pending"]:
            self._log("  같은 Study 이름으로 다시 시작하면 남은 모델만 실행됩니다")

        # 마지막 완료 모델을 화면에 남긴다
        last = next((done[a] for a in reversed(self.study_order) if a in done), None)
        if last is not None:
            self.current_run = last
            try:
                self._load_run(last)
                self.status_text.set(f"Study {state} · {study_dir.name} · {summary}")
            except Exception as e:                             # noqa: BLE001
                self._log(f"  [경고] 결과 로드 실패: {e}")
        self._refresh_runs()
        self.proc = None
        # 비교 그림까지 끝난 뒤에 최종 완료 줄을 찍고 로그 파일을 닫는다.
        # (여기서 log_fp 를 먼저 닫으면 요약이 파일에 남지 않는다)
        if n_done >= 2:
            self._make_study_figures(study_dir)
        else:
            if n_done:
                self._log("  비교 그림은 완료 run 이 2개 이상일 때 생성됩니다")
            self._finish_study_log()

    def _finish_study_log(self):
        """Study 로그의 마지막 줄 — 여기서 끝났음을 한눈에 보이게 찍는다."""
        state, study_dir, summary = getattr(
            self, "_study_result", ("종료", self.study_dir, ""))
        elapsed = ""
        if self.study_started is not None:
            sec = int(time.time() - self.study_started)
            elapsed = f" · 소요 {sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"
        self._log("─" * 60)
        self._log(f"[Study {state}] {study_dir.name if study_dir else '-'} · "
                  f"{summary}{elapsed}")
        self._log(f"  결과: {study_dir}" if study_dir else "")
        if study_dir and (study_dir / "STUDY.md").exists():
            self._log("  STUDY.md · figures/loss_overlay.png 확인 가능 "
                      "(우측 'STUDY.md 열기' / 'Study 폴더 열기')")
        self._log("─" * 60)
        if self.log_fp:
            self.log_fp.close(); self.log_fp = None
        self.study_started = None

    def _make_study_figures(self, study_dir):
        """모델별 val_loss 곡선 겹치기 등 비교 그림 4종 (scripts/make_study.py)."""
        def worker():
            try:
                env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
                r = subprocess.run(
                    [PYTHON, str(STUDY_FIGURES), "--study", str(study_dir)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    cwd=str(BASE), env=env)
                for ln in (r.stdout or "").strip().splitlines():
                    self.q.put(json.dumps({"event": "log", "message": ln.strip()}))
                if r.returncode == 0:
                    self.q.put(json.dumps({"event": "log", "message":
                                           f"[비교 그림] {study_dir/'figures'/'loss_overlay.png'}"
                                           " — 모델별 val_loss 곡선"}))
                else:
                    self.q.put(json.dumps({"event": "log", "message":
                                           f"[비교 그림] 실패(무시): {(r.stderr or '').strip()[-200:]}"}))
            except Exception as e:                             # noqa: BLE001
                self.q.put(json.dumps({"event": "log",
                                       "message": f"[비교 그림] 실패(무시): {e}"}))
            self.q.put({"__study_done__": True})   # 최종 완료 줄 트리거
        self._log("[비교 그림] 생성 중…")
        threading.Thread(target=worker, daemon=True).start()

    def _post_process(self, run):
        """학습과 분리된 별도 프로세스로 그림·리포트 생성 (실패해도 결과 보존)."""
        def worker():
            for script, label in ((FIGURES, "그림"), (REPORT, "리포트")):
                try:
                    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
                    r = subprocess.run(
                        [PYTHON, str(script), "--run", str(run)],
                        capture_output=True, text=True, encoding="utf-8", env=env)
                    self.q.put(json.dumps({"event": "log",
                                           "message": f"[{label}] "
                                           + (r.stdout or "").strip().splitlines()[-1]
                                           if r.stdout else f"[{label}] 완료"}))
                except Exception as e:                         # noqa: BLE001
                    self.q.put(json.dumps({"event": "log",
                                           "message": f"[{label}] 실패(무시): {e}"}))
            self.q.put({"__loaded__": str(run)})
        threading.Thread(target=worker, daemon=True).start()
        # 즉시 metrics 로드 (그림·리포트를 기다리지 않는다)
        self._load_run(run)

    def _load_run(self, run):
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        self.current_metrics = metrics
        self._refresh_cards(metrics)
        self.chart.load(metrics.get("history", []), metrics.get("stage_events"))
        # head 탭 구성 — 탭 하나마다 혼동행렬 캔버스를 그 안에 넣는다
        head_data = available_heads(run, metrics)
        self._reset_head_tabs()
        if head_data:
            for tab in self.head_tabs.tabs():
                self.head_tabs.forget(tab)          # 안내용 '결과' 탭 제거
            self.cms = {}
            self.head_data = head_data
            for head, _lg, _lb, _names in head_data:
                self.cms[head] = self._add_head_tab(head)
            self._on_head_tab()
        s = normalize_run_summary(metrics)
        self.status_text.set(
            f"완료 · {run.name} · macro-F1 "
            f"{s.get('stage_macro_f1') or s.get('val_acc') or '—'}")

    def _save_conclusion(self):
        run = self.current_run
        if not run or not (run / "report.md").exists():
            messagebox.showinfo("결론", "먼저 학습을 완료해 report.md 가 생성돼야 합니다.")
            return
        import re
        text = self.concl_text.get("1.0", "end").strip() or "⚠️ 미작성"
        rp = run / "report.md"
        t = rp.read_text(encoding="utf-8")
        t = re.sub(r"(<!--fill:conclusion-start-->\n).*?(\n<!--fill:conclusion-end-->)",
                   lambda m: m.group(1) + text + m.group(2), t, flags=re.DOTALL)
        rp.write_text(t, encoding="utf-8")
        self._log(f"[결론] {run.name}/report.md 에 저장")

    def _open_report(self):
        run = self.current_run
        if run and (run / "report.md").exists():
            try:
                os.startfile(run / "report.md")             # noqa: — Windows
            except Exception:                                # noqa: BLE001
                self._log(f"report.md 경로: {run/'report.md'}")
        else:
            messagebox.showinfo("리포트", "report.md 가 아직 없습니다.")

    def _study_target_dir(self):
        if self.study_dir is not None:
            return self.study_dir
        ok, name = validate_study_name(self.study_name.get())
        return RUNS / name if ok else None

    def _open_path(self, path, what):
        if path is not None and Path(path).exists():
            try:
                os.startfile(path)                             # noqa: — Windows
            except Exception:                                  # noqa: BLE001
                self._log(f"{what} 경로: {path}")
            return
        messagebox.showinfo(what, f"{what} 가 아직 없습니다. Study 를 먼저 실행하세요.")

    def _open_study_md(self):
        d = self._study_target_dir()
        self._open_path(d / "STUDY.md" if d else None, "STUDY.md")

    def _open_study_dir(self):
        self._open_path(self._study_target_dir(), "Study 폴더")

    def _on_close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askokcancel("종료", "학습이 진행 중입니다. 중단하고 종료할까요?"):
                return
            if self.is_study and self._kill_tree(self.proc):
                pass
            else:
                self.proc.terminate()
        self.destroy()


if __name__ == "__main__":
    LabApp().mainloop()
