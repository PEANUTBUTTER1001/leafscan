"""core/models.py — 모델 팩토리.

P1: SimpleCNN 을 train_worker.py 에서 **변경 없이** 이동. build_model(cfg) 도입.
    이 단계에서는 기능을 추가하지 않는다 (골든 회귀 동일성 유지).

이후 P2 에서 ModelSpec 레지스트리와 백본 6종, P3 에서 MultiHeadNet 이 붙는다.
"""
import torch
import torch.nn as nn

from tasks.models_registry import get_spec


class SimpleCNN(nn.Module):
    """Conv → (ReLU) → MaxPool 블록을 N개 쌓고 FC 하나로 분류.

    FC 입력 크기를 더미 텐서 forward 로 자동 산출한다.
    (train_worker.py 원본에서 그대로 이동 — 동작 불변.)
    """

    def __init__(self, num_classes=10, conv_channels=16, kernel_size=3,
                 stride=2, padding=1, pool_kernel=3, pool_stride=2,
                 conv_blocks=1, use_relu=True, in_ch=3, img_size=32):
        super().__init__()
        mods = []
        with torch.no_grad():
            x = torch.zeros(1, in_ch, img_size, img_size)
            c_in = in_ch
            for i in range(conv_blocks):
                c_out = conv_channels * (2 ** i)
                conv = nn.Conv2d(c_in, c_out, kernel_size,
                                 stride=(stride if i == 0 else 1), padding=padding)
                x = conv(x)
                mods.append(conv)
                if use_relu:
                    mods.append(nn.ReLU(inplace=True))
                    x = torch.relu(x)
                pool = nn.MaxPool2d(pool_kernel, stride=pool_stride)
                if x.shape[-1] >= pool_kernel and x.shape[-2] >= pool_kernel:
                    x = pool(x)
                    mods.append(pool)
                c_in = c_out
            self.flat_features = int(x.numel())
        self.features = nn.Sequential(*mods)
        self.fc = nn.Linear(self.flat_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)          # FC 뒤에는 활성화 없음 (CrossEntropyLoss가 처리)


def build_simple_cnn(cfg, num_classes, in_ch=3, img_size=32):
    # 기본값 — 골든 config 는 이 값들을 명시하므로 수치 불변, 미지정 시에도 동작.
    return SimpleCNN(
        num_classes=num_classes,
        conv_channels=int(cfg.get("conv_channels", 16)),
        kernel_size=int(cfg.get("kernel_size", 3)),
        stride=int(cfg.get("stride", 2)),
        padding=int(cfg.get("padding", 1)),
        pool_kernel=int(cfg.get("pool_kernel", 3)),
        pool_stride=int(cfg.get("pool_stride", 2)),
        conv_blocks=int(cfg.get("conv_blocks", 1)),
        use_relu=bool(cfg.get("use_relu", True)),
        in_ch=in_ch, img_size=img_size)


# --------------------------------------------------------------------------
# P2: torchvision 백본 — classifier/fc 를 nn.Identity 로 치환하고 feature 만 뽑는다
# --------------------------------------------------------------------------
def strip_classifier(backbone: nn.Module) -> nn.Module:
    """백본 마지막 분류 레이어를 Identity 로 교체. 계열별 속성명 차이를 흡수한다."""
    if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Linear):
        backbone.fc = nn.Identity()                 # resnet 계열
    elif hasattr(backbone, "classifier"):
        backbone.classifier = nn.Identity()         # efficientnet/mobilenet/convnext
    else:
        raise RuntimeError(f"분류 레이어를 찾지 못함: {type(backbone).__name__}")
    return backbone


def infer_feature_dim(backbone: nn.Module, img_size: int, in_ch=3) -> int:
    """더미 텐서를 통과시켜 flatten 후 feature 차원을 구한다 (해상도 무관 = 채널수)."""
    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        x = torch.zeros(1, in_ch, img_size, img_size)
        feat = backbone(x).flatten(1)
    if was_training:
        backbone.train()
    return int(feat.shape[1])


class SingleHeadNet(nn.Module):
    """백본 + 단일 Linear head. feature 를 flatten 해 head 로 보낸다.

    (P3 에서 MultiHeadNet 으로 대체·확장된다. simple_cnn 은 이 경로를 쓰지 않는다.)
    """

    def __init__(self, backbone, feature_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.head(feat)


def build_backbone(arch, pretrained, img_size=None):
    """(backbone, feature_dim) 반환. simple_cnn 은 여기서 처리하지 않는다."""
    spec = get_spec(arch)
    if spec.is_fixture:
        raise ValueError("simple_cnn 은 build_simple_cnn 으로 만든다")
    size = img_size or spec.default_size
    if pretrained and spec.min_size and size < spec.min_size:
        raise ValueError(
            f"{arch}: pretrained 백본에 입력 {size}px 는 너무 작다 "
            f"(최소 {spec.min_size}px — feature map 소실). img_size 를 키우세요.")
    backbone = spec.build(pretrained)
    backbone = strip_classifier(backbone)
    fd = spec.feature_dim or infer_feature_dim(backbone, size)
    return backbone, fd


def build_model(cfg, num_classes, in_ch=3, img_size=None):
    """cfg["arch"] 에 따라 모델을 만든다. 미지정 시 simple_cnn (v1 하위호환).

    P2: simple_cnn 은 기존 경로(동작 불변), 그 외 arch 는 백본+단일 head.
    """
    arch = cfg.get("arch", "simple_cnn")
    if arch == "simple_cnn":
        return build_simple_cnn(cfg, num_classes, in_ch=in_ch,
                                img_size=img_size or 32)
    spec = get_spec(arch)
    size = int(cfg.get("img_size") or img_size or spec.default_size)
    pretrained = bool(cfg.get("pretrained", False))
    backbone, fd = build_backbone(arch, pretrained, img_size=size)
    return SingleHeadNet(backbone, fd, num_classes)


# --------------------------------------------------------------------------
# P3: 멀티헤드 — 백본 공유, head N개를 nn.ModuleDict 로. forward → dict
# --------------------------------------------------------------------------
class MultiHeadNet(nn.Module):
    """백본 하나를 공유하고 head 만 분리. forward 는 {head: logit} dict 를 반환.

    · head 는 logit 만 반환 (CrossEntropyLoss 가 log-softmax 수행 — 중복 금지)
    · 단일 head 도 dict 로 처리 (분기를 두 벌 만들면 한쪽만 고치는 버그 발생)
    """

    def __init__(self, backbone, in_features, heads: dict):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = in_features
        self.heads = nn.ModuleDict({k: nn.Linear(in_features, n)
                                    for k, n in heads.items()})

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return {k: h(feat) for k, h in self.heads.items()}


def build_simple_cnn_features(cfg, in_ch=3, img_size=32):
    """simple_cnn 을 feature 추출기로 (fc → Identity). 멀티헤드 결합용."""
    m = build_simple_cnn(cfg, num_classes=1, in_ch=in_ch, img_size=img_size)
    fd = m.flat_features
    m.fc = nn.Identity()
    return m, fd


def build_multihead_model(cfg, heads: dict, in_ch=3, img_size=None):
    """heads(={name: n_classes}) 에 맞는 멀티헤드 모델을 만든다.

    simple_cnn 단일 head 는 build_model 의 기존 경로(동작 불변)를 그대로 쓴다.
    그 외에는 MultiHeadNet 으로 백본을 공유한다.
    """
    arch = cfg.get("arch", "simple_cnn")
    if arch == "simple_cnn":
        size = int(cfg.get("img_size") or img_size or 32)
        backbone, fd = build_simple_cnn_features(cfg, in_ch=in_ch, img_size=size)
        return MultiHeadNet(backbone, fd, heads)
    spec = get_spec(arch)
    size = int(cfg.get("img_size") or img_size or spec.default_size)
    pretrained = bool(cfg.get("pretrained", False))
    backbone, fd = build_backbone(arch, pretrained, img_size=size)
    return MultiHeadNet(backbone, fd, heads)
