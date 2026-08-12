"""core/schedule.py — 전이학습 freeze 스케줄 (P5).

가장 흔한 실수: requires_grad 만 바꾸고 optimizer 를 그대로 두면 파라미터 그룹이
갱신되지 않아 **아무 일도 일어나지 않는다** (에러도 없다). 그래서 unfreeze 시
optimizer 를 반드시 재생성하고, 학습 가능 파라미터 수를 로그로 남겨 확인한다.
"""
import torch.optim as optim


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_backbone_frozen(model, frozen: bool):
    """model.backbone 이 있으면 requires_grad 를 설정. 없으면(simple_cnn 단일head) 무시."""
    bb = getattr(model, "backbone", None)
    if bb is None:
        return False
    for p in bb.parameters():
        p.requires_grad = not frozen
    return True


def build_optimizer(cfg, model, lr):
    """학습 가능 파라미터만으로 optimizer 생성 (freeze 반영)."""
    params = [p for p in model.parameters() if p.requires_grad]
    name = cfg.get("optimizer", "Adam")
    if name == "SGD":
        return optim.SGD(params, lr=lr, momentum=0.9)
    if name == "Adam":
        return optim.Adam(params, lr=lr)
    return optim.AdamW(params, lr=lr)
