"""core/figures_advanced.py — 심화 그림 (P10b).

오분류 그리드 · Grad-CAM · t-SNE. 실제 이미지·모델이 필요하므로,
run 의 config 로 데이터/모델을 재구성하고 best.pt 를 로드한다.
fake 데이터·이미지 없음·best.pt 없음이면 (False, 사유) 로 건너뛴다 (축 C7).

Grad-CAM 대상 레이어는 레지스트리의 CamSpec 에서 온다 → 그림 코드는 arch 를 모른다.
"""
import json
from pathlib import Path

import numpy as np
import torch

from core.figures import PALETTE, _ensure_style, _save


def _rebuild(run: Path, metrics):
    """(model, val_ds, heads, label_names, cfg) 재구성. 불가하면 None."""
    cfg = metrics.get("config", {})
    if cfg.get("dataset") != "index_csv":
        return None
    from core.data import build_multihead_data
    from core.models import build_multihead_model
    train_ds, val_ds, test_ds, meta = build_multihead_data(cfg)
    model = build_multihead_model(cfg, meta["heads"])
    bp = run / "best.pt"
    if bp.exists():
        model.load_state_dict(torch.load(bp, map_location="cpu"))
    model.eval()
    return model, val_ds, meta["heads"], {
        "crop": meta["crop_names"], "stage": meta["stage_names"]}, cfg


def _denorm(tensor, norm):
    (mean, std) = norm
    t = tensor.clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    return t.clamp(0, 1).permute(1, 2, 0).numpy()


# --------------------------------------------------------------------------
# 오분류 그리드
# --------------------------------------------------------------------------
def misclassified_grid(run, metrics, out_path, head="stage", max_n=12):
    rb = _rebuild(Path(run), metrics)
    if rb is None:
        return False, "index_csv 아님 — 건너뜀"
    model, val_ds, heads, names, cfg = rb
    if head not in heads:
        head = next(iter(heads))
    from core.metrics import softmax_np
    from tasks.models_registry import get_spec
    norm = get_spec(cfg.get("arch", "resnet18")).norm
    _ensure_style()
    import matplotlib.pyplot as plt

    wrong = []
    with torch.no_grad():
        for i in range(len(val_ds)):
            x, y = val_ds[i]
            out = model(x.unsqueeze(0))
            logit = out[head][0].numpy()
            p = int(logit.argmax())
            conf = float(softmax_np(logit[None])[0].max())
            if p != y[head]:
                wrong.append((i, x, y[head], p, conf))
            if len(wrong) >= max_n:
                break
    if not wrong:
        return False, "오분류 표본 없음 (완벽 분류이거나 표본 부족)"
    cn = names.get(head, [])
    cols = min(4, len(wrong)); rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (i, x, t, p, conf) in zip(axes, wrong):
        ax.imshow(_denorm(x, norm))
        ax.set_title(f"정답 {cn[t] if t < len(cn) else t}\n"
                     f"예측 {cn[p] if p < len(cn) else p} ({conf:.2f})",
                     fontsize=8, color=PALETTE[3])
    fig.suptitle(f"오분류 상위 {len(wrong)}장 · {head}")
    return _save(fig, out_path)


# --------------------------------------------------------------------------
# Grad-CAM — 계열별 CamSpec 으로 대상 레이어 자동 선택
# --------------------------------------------------------------------------
def resolve_layer(module, path):
    """'layer4.-1' / 'features.-1' 같은 경로를 실제 모듈로 해석."""
    obj = module
    for tok in path.split("."):
        if tok.lstrip("-").isdigit():
            obj = obj[int(tok)]
        else:
            obj = getattr(obj, tok)
    return obj


def _gradcam_one(model, backbone, target_layer, x, head, target_class):
    acts, grads = {}, {}

    def fwd_hook(_m, _i, o):
        acts["v"] = o.detach()

    def bwd_hook(_m, gi, go):
        grads["v"] = go[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad()
        out = model(x.unsqueeze(0))
        score = out[head][0, target_class]
        score.backward()
        a = acts["v"][0]          # C,H,W
        g = grads["v"][0]         # C,H,W
        if a.dim() != 3:          # convnext 등 채널 순서 대비
            return None
        weights = g.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * a).sum(0))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.numpy()
    finally:
        h1.remove(); h2.remove()


def gradcam_samples(run, metrics, out_path, head="stage", n=4):
    rb = _rebuild(Path(run), metrics)
    if rb is None:
        return False, "index_csv 아님 — 건너뜀"
    model, val_ds, heads, names, cfg = rb
    if head not in heads:
        head = next(iter(heads))
    from tasks.models_registry import get_spec
    spec = get_spec(cfg.get("arch", "resnet18"))
    backbone = getattr(model, "backbone", model)
    try:
        target_layer = resolve_layer(backbone, spec.cam.layer)
    except Exception as e:  # noqa: BLE001
        return False, f"CAM 레이어 해석 실패({spec.cam.layer}): {e}"
    _ensure_style()
    import matplotlib.pyplot as plt
    norm = spec.norm
    k = min(n, len(val_ds))
    fig, axes = plt.subplots(2, k, figsize=(k * 2.4, 5))
    axes = np.array(axes).reshape(2, k)
    for j in range(k):
        x, y = val_ds[j]
        with torch.no_grad():
            pred = int(model(x.unsqueeze(0))[head][0].argmax())
        cam = _gradcam_one(model, backbone, target_layer, x, head, pred)
        img = _denorm(x, norm)
        axes[0, j].imshow(img); axes[0, j].axis("off")
        cn = names.get(head, [])
        axes[0, j].set_title(cn[pred] if pred < len(cn) else pred, fontsize=8)
        axes[1, j].imshow(img)
        if cam is not None:
            import torch.nn.functional as F
            cam_r = F.interpolate(torch.tensor(cam)[None, None],
                                  size=img.shape[:2], mode="bilinear",
                                  align_corners=False)[0, 0].numpy()
            axes[1, j].imshow(cam_r, cmap="jet", alpha=0.45)
        axes[1, j].axis("off")
    fig.suptitle(f"Grad-CAM · {cfg.get('arch')} · {head}")
    return _save(fig, out_path)


# --------------------------------------------------------------------------
# t-SNE — 특징 공간 클래스 분리도
# --------------------------------------------------------------------------
def embedding_tsne(run, metrics, out_path, head="stage", max_n=300):
    rb = _rebuild(Path(run), metrics)
    if rb is None:
        return False, "index_csv 아님 — 건너뜀"
    model, val_ds, heads, names, cfg = rb
    if head not in heads:
        head = next(iter(heads))
    backbone = getattr(model, "backbone", model)
    feats, labels = [], []
    with torch.no_grad():
        for i in range(min(max_n, len(val_ds))):
            x, y = val_ds[i]
            f = backbone(x.unsqueeze(0)).flatten(1)[0].numpy()
            feats.append(f); labels.append(y[head])
    feats = np.array(feats); labels = np.array(labels)
    if len(feats) < 5:
        return False, "표본 부족"
    from sklearn.manifold import TSNE
    perp = min(30, max(2, len(feats) // 4))
    emb = TSNE(n_components=2, perplexity=perp, init="pca",
               random_state=42).fit_transform(feats)
    _ensure_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    cn = names.get(head, [])
    for c in sorted(set(labels)):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=14, alpha=0.7,
                   color=PALETTE[c % len(PALETTE)],
                   label=cn[c] if c < len(cn) else str(c))
    ax.legend(); ax.set_title(f"t-SNE 특징 공간 · {head}")
    ax.set_xticks([]); ax.set_yticks([])
    return _save(fig, out_path)
