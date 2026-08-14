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

실행:  python lab_gui.py
"""
import json
import math
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

try:
    import numpy as np
except ImportError:                                            # noqa: BLE001
    np = None

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
WORKER = BASE / "train_worker.py"
FIGURES = BASE / "make_figures.py"
REPORT = BASE / "make_report.py"

BG, CARD, LINE, INK, MUTED = "#f4f5f7", "#ffffff", "#dcdfe4", "#1f2328", "#6b7280"
ACCENT, WARN, OK = "#2f6f4f", "#b4432f", "#2f6f4f"

ARCHS = ["simple_cnn", "resnet18", "mobilenet_v3_small",
         "efficientnet_b0", "resnet50", "convnext_tiny"]
# arch별 상대 학습시간 계수 (러프 추정용 — conv_blocks 대신)
ARCH_COST = {"simple_cnn": 0.2, "resnet18": 1.0, "mobilenet_v3_small": 0.6,
             "efficientnet_b0": 1.3, "resnet50": 2.4, "convnext_tiny": 2.6}

PRESETS = {
    "빠른확인(더미)": dict(dataset="multilabel_fake", subset=256, img_size=32,
                       arch="resnet18", epochs=3, batch_size=32, lr=1e-3,
                       optimizer="Adam", pretrained=False, freeze_epochs=0,
                       class_weight="none"),
    "상추 베이스라인": dict(dataset="index_csv", arch="resnet18", img_size=224,
                       epochs=20, batch_size=16, lr=1e-3, optimizer="AdamW",
                       pretrained=True, freeze_epochs=3, class_weight="auto",
                       split_policy="resplit_all", crop_mode="full"),
    "경량(모바일)": dict(dataset="index_csv", arch="mobilenet_v3_small", img_size=224,
                     epochs=20, batch_size=32, lr=3e-3, optimizer="AdamW",
                     pretrained=True, freeze_epochs=3, class_weight="auto",
                     split_policy="resplit_all"),
    "CIFAR 회귀": dict(dataset="cifar10", subset=10000, img_size=32,
                     arch="simple_cnn", epochs=10, batch_size=64, lr=1e-3,
                     optimizer="Adam", pretrained=False, freeze_epochs=0),
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
        self.v = dict(
            dataset=tk.StringVar(value="index_csv"),
            data_root=tk.StringVar(value=str(BASE / "data")),
            index_csv=tk.StringVar(value=str(BASE / "data" / "index.csv")),
            subset=tk.IntVar(value=0),
            val_ratio=tk.DoubleVar(value=0.3),
            split_policy=tk.StringVar(value="resplit_all"),
            ambiguous_policy=tk.StringVar(value="include"),
            crop_mode=tk.StringVar(value="full"),
            augment=tk.BooleanVar(value=True),
            arch=tk.StringVar(value="resnet18"),
            img_size=tk.IntVar(value=224),
            pretrained=tk.BooleanVar(value=True),
            freeze_epochs=tk.IntVar(value=3),
            # simple_cnn 전용
            conv_channels=tk.IntVar(value=16),
            kernel_size=tk.IntVar(value=3),
            stride=tk.IntVar(value=2),
            pool_kernel=tk.IntVar(value=3),
            pool_stride=tk.IntVar(value=2),
            conv_blocks=tk.IntVar(value=1),
            use_relu=tk.BooleanVar(value=True),
            epochs=tk.IntVar(value=20),
            batch_size=tk.IntVar(value=16),
            lr_log=tk.DoubleVar(value=-3.0),
            lr_finetune_log=tk.DoubleVar(value=-4.0),
            optimizer=tk.StringVar(value="AdamW"),
            class_weight=tk.StringVar(value="auto"),
            w_crop=tk.DoubleVar(value=0.4),
            w_stage=tk.DoubleVar(value=0.6),
            patience=tk.IntVar(value=5),
            seed=tk.IntVar(value=42),
        )
        self.hypothesis = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="유휴")
        self.lr_text = tk.StringVar(value="1.0e-03")
        self.est_text = tk.StringVar(value="")
        self.cell_text = tk.StringVar(value="혼동행렬의 칸을 클릭하면 상세가 표시됩니다")
        self.baseline_var = tk.StringVar(value="(없음)")
        self.v["lr_log"].trace_add("write", lambda *_: self._on_lr())
        self.v["arch"].trace_add("write", lambda *_: self._on_arch())
        for k in ("epochs", "batch_size", "subset", "dataset", "arch", "img_size"):
            self.v[k].trace_add("write", lambda *_: self._estimate())

    def lr(self):
        return 10 ** float(self.v["lr_log"].get())

    def lr_finetune(self):
        return 10 ** float(self.v["lr_finetune_log"].get())

    def _on_lr(self):
        self.lr_text.set(f"{self.lr():.1e}")

    def _on_arch(self):
        is_simple = self.v["arch"].get() == "simple_cnn"
        if hasattr(self, "simple_frame"):
            if is_simple:
                self.simple_frame.pack(fill="x", pady=(4, 0))
            else:
                self.simple_frame.pack_forget()
        self._estimate()

    def _estimate(self):
        try:
            ds = self.v["dataset"].get()
            n = self.v["subset"].get() or (177 if ds == "index_csv"
                                           else 50000 if ds == "cifar10" else 1024)
            steps = max(1, int(n * (1 - self.v["val_ratio"].get())
                               / max(1, self.v["batch_size"].get())))
            size_factor = (self.v["img_size"].get() / 32) ** 2
            per = steps * 0.010 * ARCH_COST.get(self.v["arch"].get(), 1.0) * size_factor
            total = per * self.v["epochs"].get()
            self.est_text.set(f"예상 소요 약 {total/60:.1f}분 (CPU 러프 추정)")
        except Exception:                                      # noqa: BLE001
            self.est_text.set("")

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
        pre = ttk.Frame(p)
        pre.pack(fill="x")
        ttk.Label(pre, text="프리셋", style="Sec.TLabel").pack(anchor="w")
        prow = ttk.Frame(pre)
        prow.pack(fill="x", pady=(2, 8))
        for name in PRESETS:
            ttk.Button(prow, text=name, width=13,
                       command=lambda n=name: self._apply_preset(n)).pack(
                           side="left", padx=1)

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
        ttk.Label(arow, text="arch", width=13).pack(side="left")
        ttk.Combobox(arow, textvariable=self.v["arch"], values=ARCHS,
                     state="readonly", width=20).pack(side="left")
        self._seg(m, self.v["img_size"], [32, 64, 128, 192, 224], "입력 해상도")
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
        ttk.Label(r, text="가설 (왜 이 실험을 하는가 — 실행 전 작성)",
                  style="Hint.TLabel").pack(anchor="w")
        self.hypo_text = tk.Text(r, height=3, font=(self.font, 9), wrap="word")
        self.hypo_text.pack(fill="x", pady=(2, 6))
        self.run_btn = ttk.Button(r, text="▶  학습 실행", style="Run.TButton",
                                  command=self.start_training)
        self.run_btn.pack(fill="x")
        self.stop_btn = ttk.Button(r, text="■  중단", command=self.stop_training,
                                   state="disabled")
        self.stop_btn.pack(fill="x", pady=(4, 0))
        ttk.Label(r, textvariable=self.est_text, style="Hint.TLabel").pack(
            anchor="w", pady=(4, 0))
        self._estimate(); self._on_lr(); self._on_arch()

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
        self.head_tabs = ttk.Notebook(rf, height=24)
        self.head_tabs.pack(fill="x")
        self.head_tabs.bind("<<NotebookTabChanged>>", lambda e: self._on_head_tab())
        self.cm = ConfusionMatrix(rf, on_select=self._on_cell, height=240)
        self.cm.pack(fill="both", expand=True, pady=(2, 0))
        ttk.Label(rf, textvariable=self.cell_text, style="Hint.TLabel",
                  wraplength=330).pack(anchor="w", pady=(4, 0))

        # 결론 + 리포트 열기
        cf = ttk.Frame(p)
        cf.pack(fill="x", pady=(8, 0))
        ttk.Label(cf, text="결론 (결과 해석 — report.md §7 에 반영)",
                  style="Sec.TLabel").pack(side="left")
        ttk.Button(cf, text="report.md 열기", command=self._open_report).pack(side="right")
        ttk.Button(cf, text="결론 저장", command=self._save_conclusion).pack(
            side="right", padx=(0, 6))
        self.concl_text = tk.Text(p, height=3, font=(self.font, 9), wrap="word")
        self.concl_text.pack(fill="x", pady=(2, 6))

        ttk.Label(p, text="로그", style="Sec.TLabel").pack(anchor="w")
        wrap = ttk.Frame(p, style="Card.TFrame")
        wrap.pack(fill="both", expand=True)
        self.log = tk.Text(wrap, height=8, bg="#101418", fg="#d7dde3", relief="flat",
                           font=("Consolas" if os.name == "nt" else "TkFixedFont", 9),
                           wrap="none")
        sb = ttk.Scrollbar(wrap, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set, state="disabled")
        sb.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)

    def _on_cell(self, true_c, pred_c, v, row_total):
        pct = 100 * v / row_total if row_total else 0
        kind = "정답" if true_c == pred_c else "오분류"
        self.cell_text.set(f"실제 [{true_c}] → 예측 [{pred_c}] : {v}장 "
                           f"({kind}, 실제 클래스의 {pct:.1f}%)")

    def _on_head_tab(self):
        if not self.head_data:
            return
        idx = self.head_tabs.index("current")
        if idx < len(self.head_data):
            head, lg, lb, names = self.head_data[idx]
            if np is not None:
                names = names or [f"{head}_{i}" for i in range(np.load(lg).shape[1])]
                self.cm.load_from_arrays(np.load(lg), np.load(lb), names)

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        if self.log_fp:
            self.log_fp.write(text.rstrip() + "\n"); self.log_fp.flush()

    def _apply_preset(self, name):
        cfg = PRESETS[name]
        for k, val in cfg.items():
            if k == "lr":
                self.v["lr_log"].set(math.log10(val))
            elif k in self.v:
                self.v[k].set(val)
        self._log(f"[프리셋] '{name}' 적용")
        self._estimate()

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
            num_workers="auto",   # index_csv 에서 병렬 로딩 (워커가 CPU 수로 결정)
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
        self.chart.reset(); self.cm.clear()
        for tab in self.head_tabs.tabs():
            self.head_tabs.forget(tab)
        self.head_data = []
        self.prog.config(value=0, maximum=cfg["epochs"])
        self._log(f"[{run_dir.name}] 시작 · arch={cfg['arch']} · config.json 저장")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   PYTHONUTF8="1")
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(WORKER), "--config",
             str(run_dir / "config.json"), "--out", str(run_dir)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1, env=env, **kw)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_text.set(f"학습 중 · {run_dir.name} · 0%")

    def _reader(self, proc):
        for line in proc.stdout:
            self.q.put(line)
        self.q.put({"__exit__": proc.wait()})

    def stop_training(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self._log("[중단] 종료 신호를 보냈습니다")

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, dict):
                    if "__exit__" in item:
                        self._on_exit(item["__exit__"])
                    # "__loaded__" 등 기타 신호는 무시 (이미 처리됨)
                    continue
                line = item.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    self._log("  " + line); continue
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
            self.status_text.set(f"학습 중 · {self.current_run.name} · "
                                 f"{int(e/tot*100)}%")
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
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
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

    def _post_process(self, run):
        """학습과 분리된 별도 프로세스로 그림·리포트 생성 (실패해도 결과 보존)."""
        def worker():
            for script, label in ((FIGURES, "그림"), (REPORT, "리포트")):
                try:
                    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
                    r = subprocess.run(
                        [sys.executable, str(script), "--run", str(run)],
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
        # head 탭 구성
        for tab in self.head_tabs.tabs():
            self.head_tabs.forget(tab)
        self.head_data = available_heads(run, metrics)
        for head, _lg, _lb, _names in self.head_data:
            self.head_tabs.add(ttk.Frame(self.head_tabs), text=head)
        if self.head_data:
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

    def _on_close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askokcancel("종료", "학습이 진행 중입니다. 중단하고 종료할까요?"):
                return
            self.proc.terminate()
        self.destroy()


if __name__ == "__main__":
    LabApp().mainloop()
