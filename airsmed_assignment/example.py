"""
모델 레지스트리 데모 스크립트
registry.py의 기능을 시연하는 예제
"""

import torch
import torchvision.models as models
from registry import ModelRegistry


def download_and_save_models():
    """사전 학습된 모델 2개를 다운로드하고 저장"""
    print("\n" + "="*60)
    print("1단계: 사전 학습된 모델 다운로드")
    print("="*60)

    # ResNet-18 다운로드 (~45MB)
    print("\n📥 ResNet-18 다운로드 중...")
    resnet18 = models.resnet18(pretrained=True)
    torch.save(resnet18.state_dict(), "resnet18_pretrained.pth")
    print("✓ ResNet-18 저장 완료: resnet18_pretrained.pth")

    # MobileNetV2 다운로드 (~14MB)
    print("\n📥 MobileNetV2 다운로드 중...")
    mobilenet = models.mobilenet_v2(pretrained=True)
    torch.save(mobilenet.state_dict(), "mobilenetv2_pretrained.pth")
    print("✓ MobileNetV2 저장 완료: mobilenetv2_pretrained.pth")


def demo_registry():
    """레지스트리 기능 시연"""

    # 모델 다운로드
    download_and_save_models()

    # 레지스트리 초기화
    print("\n" + "="*60)
    print("2단계: 레지스트리 초기화")
    print("="*60)
    registry = ModelRegistry(storage_path="./models", metadata_file="./registry.yaml")
    print("✓ 레지스트리 초기화 완료")

    # 모델 등록
    print("\n" + "="*60)
    print("3단계: 모델 등록")
    print("="*60)

    # ResNet-18 등록 (v1)
    print("\n[1] ResNet-18 등록...")
    registry.register(
        name="resnet18",
        model_path="resnet18_pretrained.pth",
        framework="pytorch",
        architecture="ResNet-18",
        input_shape=(3, 224, 224),
        metrics={"top1_accuracy": 0.697, "top5_accuracy": 0.891},
        dataset="ImageNet",
        description="사전 학습된 ResNet-18 모델 (일반 이미지 분류)"
    )

    # MobileNetV2 등록 (v1)
    print("\n[2] MobileNetV2 등록...")
    registry.register(
        name="mobilenetv2",
        model_path="mobilenetv2_pretrained.pth",
        framework="pytorch",
        architecture="MobileNetV2",
        input_shape=(3, 224, 224),
        metrics={"top1_accuracy": 0.718, "top5_accuracy": 0.901},
        dataset="ImageNet",
        description="경량화된 MobileNetV2 모델 (모바일 환경용)"
    )

    # ResNet-18 v2 등록 (버전 자동 증가 테스트)
    print("\n[3] ResNet-18 v2 등록 (동일 모델, 버전 증가 테스트)...")
    registry.register(
        name="resnet18",
        model_path="resnet18_pretrained.pth",
        metrics={"top1_accuracy": 0.710, "top5_accuracy": 0.895},
        description="Fine-tuned ResNet-18 v2"
    )

    # 모델 조회
    print("\n" + "="*60)
    print("4단계: 모델 조회")
    print("="*60)

    # latest 버전 조회
    print("\n[1] ResNet-18 latest 버전 조회:")
    model_info = registry.get("resnet18", "latest")
    if model_info:
        print(f"   - 버전: {model_info['version']}")
        print(f"   - 파일 경로: {model_info['file_path']}")
        print(f"   - 메트릭: {model_info['metrics']}")

    # 특정 버전 조회
    print("\n[2] ResNet-18 v1 버전 조회:")
    model_info = registry.get("resnet18", "v1")
    if model_info:
        print(f"   - 등록 일시: {model_info['registered_at']}")
        print(f"   - 설명: {model_info['description']}")

    # 모델 목록 조회
    print("\n" + "="*60)
    print("5단계: 모델 목록 조회")
    print("="*60)

    # 전체 모델 목록
    print("\n[1] 전체 등록된 모델:")
    all_models = registry.list()
    for model in all_models:
        print(f"   - {model}")

    # 특정 모델의 모든 버전
    print("\n[2] ResNet-18의 모든 버전:")
    resnet_versions = registry.list("resnet18")
    for model in resnet_versions:
        print(f"   - {model}")

    # 모델 패밀리 목록
    print("\n[3] 모델 패밀리:")
    families = registry.list_families()
    for family in families:
        print(f"   - {family}")

    # 레지스트리 요약
    registry.print_summary()

    print("\n" + "="*60)
    print("✓ 데모 완료!")
    print("="*60)
    print("\n💡 생성된 파일:")
    print("   - ./models/          : 모델 파일 저장소")
    print("   - ./registry.yaml    : 메타데이터 파일")
    print("\n메타데이터 파일(registry.yaml)을 열어서 구조를 확인해보세요!")


if __name__ == "__main__":
    demo_registry()
