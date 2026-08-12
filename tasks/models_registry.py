"""tasks/models_registry.py — 모델 레지스트리 (P2).

레지스트리 항목은 **생성자 함수가 아니라 명세 객체(ModelSpec)** 다 (구조도 §4).
학습·CAM·그림·리포트가 모두 이 객체만 읽으므로, 새 모델 추가는
**이 파일에 한 항목을 더하는 것으로 끝난다** (다른 파일 수정 없음 = 축 A2).

    새 모델 추가 예:
    "efficientnet_b2": ModelSpec(
        build=lambda pre: tv.efficientnet_b2(weights="IMAGENET1K_V1" if pre else None),
        feature_dim=None, default_size=260, norm=IMAGENET_NORM,
        cam=CamSpec(layer="features.-1"), params_m=9.1,
        suggested_lr=(3e-4, 1e-3, 3e-3), suggested_batch={4: 16, 8: 32},
        notes="B0 대비 정확도 우위"),
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch.nn as nn
import torchvision.models as tv

IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF_NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


@dataclass(frozen=True)
class CamSpec:
    """Grad-CAM 대상 레이어와 후처리. 계열마다 다르므로 레지스트리에 담는다.

    layer: 백본 내부 대상 모듈 경로. 예 "layer4.-1", "features.-1".
           -N 은 Sequential 인덱스의 뒤에서 N 번째를 뜻한다.
    channels_last_norm: ConvNeXt 처럼 채널 순서 처리가 필요한 경우 True.
    """
    layer: str
    channels_last_norm: bool = False


@dataclass(frozen=True)
class ModelSpec:
    build: Callable[[bool], nn.Module]   # pretrained 여부 → 백본 모듈
    default_size: int                    # 권장 입력 해상도
    norm: tuple                          # (mean, std) 정규화 상수
    cam: CamSpec                         # Grad-CAM 명세
    params_m: float                      # 파라미터 수 (백만)
    suggested_lr: tuple                  # lr 탐색 시작 3점
    suggested_batch: dict                # VRAM(GB)별 권장 배치
    notes: str = ""
    feature_dim: Optional[int] = None    # None → 더미 forward 로 자동 산출
    is_fixture: bool = False             # simple_cnn 처럼 실험 대상 아님
    min_size: int = 0                    # pretrained 사용 시 최소 입력 (feature map 소실 차단)


# --------------------------------------------------------------------------
# torchvision 백본 빌더 — pretrained 여부만 받는다
# --------------------------------------------------------------------------
def _w(name, pretrained):
    """weights enum 문자열. 미설치/구버전 대비 try."""
    return name if pretrained else None


REGISTRY: dict[str, ModelSpec] = {
    "resnet50": ModelSpec(
        build=lambda pre: tv.resnet50(weights=_w("IMAGENET1K_V2", pre)),
        default_size=224, norm=IMAGENET_NORM, cam=CamSpec(layer="layer4.-1"),
        params_m=25.6, suggested_lr=(3e-4, 1e-3, 3e-3),
        suggested_batch={4: 16, 8: 32, 12: 64}, min_size=64,
        notes="CNN 표준 베이스라인"),
    "efficientnet_b0": ModelSpec(
        build=lambda pre: tv.efficientnet_b0(weights=_w("IMAGENET1K_V1", pre)),
        default_size=224, norm=IMAGENET_NORM, cam=CamSpec(layer="features.-1"),
        params_m=5.3, suggested_lr=(3e-4, 1e-3, 3e-3),
        suggested_batch={4: 32, 8: 64, 12: 128}, min_size=64,
        notes="연산 대비 성능"),
    "convnext_tiny": ModelSpec(
        build=lambda pre: tv.convnext_tiny(weights=_w("IMAGENET1K_V1", pre)),
        default_size=224, norm=IMAGENET_NORM,
        cam=CamSpec(layer="features.-1", channels_last_norm=True),
        params_m=28.6, suggested_lr=(1e-4, 3e-4, 1e-3),
        suggested_batch={4: 16, 8: 32, 12: 64}, min_size=64,
        notes="Transformer 아이디어를 CNN 으로 흡수. 큰 lr 에서 발산 주의"),
    "resnet18": ModelSpec(
        build=lambda pre: tv.resnet18(weights=_w("IMAGENET1K_V1", pre)),
        default_size=224, norm=IMAGENET_NORM, cam=CamSpec(layer="layer4.-1"),
        params_m=11.7, suggested_lr=(1e-3, 3e-3, 1e-2),
        suggested_batch={4: 32, 8: 64, 12: 128}, min_size=32,
        notes="경량 · 빠른 반복 · 비교 하한"),
    "mobilenet_v3_small": ModelSpec(
        build=lambda pre: tv.mobilenet_v3_small(weights=_w("IMAGENET1K_V1", pre)),
        default_size=224, norm=IMAGENET_NORM, cam=CamSpec(layer="features.-1"),
        params_m=2.5, suggested_lr=(1e-3, 3e-3, 1e-2),
        suggested_batch={4: 64, 8: 128, 12: 256}, min_size=32,
        notes="초경량 · 속도/크기 하한"),
    "simple_cnn": ModelSpec(
        build=lambda pre: None,   # 특수 처리 (core/models.build_simple_cnn)
        default_size=32, norm=HALF_NORM, cam=CamSpec(layer="features.-1"),
        params_m=0.1, suggested_lr=(1e-3,),
        suggested_batch={4: 128, 8: 256, 12: 512}, is_fixture=True,
        notes="테스트 픽스처 — 하네스 회귀 검증용, 실험 대상 아님"),
}


def get_spec(arch: str) -> ModelSpec:
    if arch not in REGISTRY:
        raise ValueError(f"미등록 arch: {arch!r} (등록: {list(REGISTRY)})")
    return REGISTRY[arch]


def list_archs(include_fixture=True):
    return [k for k, v in REGISTRY.items() if include_fixture or not v.is_fixture]


def suggested_batch_for(spec: ModelSpec, vram_gb: int) -> int:
    """VRAM(GB) 이하 키 중 가장 가까운 권장 배치."""
    keys = sorted(spec.suggested_batch)
    chosen = keys[0]
    for k in keys:
        if k <= vram_gb:
            chosen = k
    return spec.suggested_batch[chosen]
