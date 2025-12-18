"""
모델 레지스트리 데모 스크립트
registry.py의 기능을 시연하는 예제
"""

import os
import torch
import torchvision.models as models
from registry import ModelRegistry
from pathlib import Path

# 원본 모델 다운로드 경로
PRETRAINED_DIR = Path("pretrained_models")

def download_and_save_models():
    """사전 학습된 모델 2개를 다운로드하고 저장 (이미 있으면 스킵)"""
    # 다운로드 디렉토리 생성
    PRETRAINED_DIR.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("1단계: 사전 학습된 모델 다운로드")
    print(f"저장 위치: {PRETRAINED_DIR}")
    print("="*60)

    # 1. ResNet-18 (General Vision)
    r18_path = PRETRAINED_DIR / "resnet18_pretrained.pth"
    if r18_path.exists():
         print("\n✓ ResNet-18 파일이 이미 존재하여 다운로드를 건너뜁니다.")
    else:
        print("\n📥 ResNet-18 다운로드 중...")
        resnet18 = models.resnet18(pretrained=True)
        torch.save(resnet18.state_dict(), r18_path) 
        print("✓ ResNet-18 저장 완료")

    # 2. DenseNet-121 (Medical AI - Chest X-ray Standard)
    dense_path = PRETRAINED_DIR / "densenet121_pretrained.pth"
    if dense_path.exists():
        print("\n✓ DenseNet-121 파일이 이미 존재하여 다운로드를 건너뜁니다.")
    else:
        print("\n📥 DenseNet-121 (Medical AI용) 다운로드 중...")
        # DenseNet121은 흉부 X-ray 분류 등 의료 영상에서 표준적으로 사용됨
        densenet = models.densenet121(pretrained=True)
        torch.save(densenet.state_dict(), dense_path)
        print("✓ DenseNet-121 저장 완료")
    
    return r18_path, dense_path


def demo_registry():
    """레지스트리 기능 시연"""

    # 모델 다운로드
    r18_path, dense_path = download_and_save_models()

    # 레지스트리 초기화
    print("\n" + "="*60)
    print("2단계: 레지스트리 초기화")
    print("="*60)
    registry = ModelRegistry(storage_path="./models", metadata_file="./registry.yaml")
    print("✓ 레지스트리 초기화 완료")

    #############
    # 모델 등록
    #############
    print("\n" + "="*60)
    print("3단계: 모델 등록 (시뮬레이션)")
    print("="*60)

    # [1] ResNet-18 등록 (General Vision)
    print("\n[1] ResNet-18 등록 (v1)...")
    registry.register(
        name="resnet18",
        model_path=str(r18_path),
        framework="pytorch",
        architecture="ResNet-18",
        input_shape=(3, 224, 224),
        metrics={"top1_accuracy": 0.697},
        dataset="ImageNet",
        description="일반적인 이미지 분류 용도"
    )

    # ResNet-18 v2 등록 (성능 개선 시뮬레이션)
    print("\n[2] ResNet-18 등록 (v2 - 성능 개선 시뮬레이션)...")
    registry.register(
        name="resnet18",
        model_path=str(r18_path), 
        metrics={"top1_accuracy": 0.725},
        description="Hyperparameter Tuned v2"
    )

    # [3] DenseNet-121 등록 (Medical AI)
    print("\n[3] DenseNet-121 (Medical) 등록...")
    registry.register(
        name="chest_xray_densenet",
        model_path=str(dense_path),
        framework="pytorch",
        architecture="DenseNet-121",
        input_shape=(1, 224, 224), # X-ray는 보통 Grayscale(1ch)이지만 전처리에 따라 3ch도 씀
        metrics={"auc_roc": 0.845}, # 의료 AI는 보통 AUC를 많이 봄
        dataset="ChestX-ray14",
        description="흉부 X-ray 진단을 위한 의료용 AI Model"
    )

    ##########
    # 모델 조회
    ##########
    print("\n" + "="*60)
    print("4단계: 모델 조회")
    print("="*60)

    # latest 버전 조회
    print("\n[1] ResNet-18 'latest' 버전 조회:")
    latest_info = registry.get("resnet18", "latest")
    if latest_info:
        print(f"   - 버전: {latest_info['version']}")
        print(f"   - 메트릭: {latest_info['metrics']}")

    # best 버전 조회 (새로 추가된 기능)
    print("\n[2] ResNet-18 'best' 버전 조회 (성능 최고점):")
    best_info = registry.get("resnet18", "best")
    if best_info:
        print(f"   - 버전: {best_info['version']}")
        print(f"   - 메트릭: {best_info['metrics']}")
        print(f"   -> v1보다 v2가 선택됨!")

    #############
    # 모델 목록 조회
    #############
    print("\n" + "="*60)
    print("5단계: 모델 목록 조회")
    print("="*60)

    # 전체 모델 목록
    print("\n[1] 전체 등록된 모델:")
    all_models = registry.list()
    for model in all_models:
        print(f"   - {model}")

    # 레지스트리 요약
    registry.print_summary()

    print("\n" + "="*60)
    print("✓ 데모 완료!")
    print("="*60)
    print("\n💡 폴더 구조:")
    print("   - ./pretrained_models/ : 원본 다운로드 파일")
    print("   - ./models/            : 레지스트리 저장소 (버전 관리)")
    print("   - ./registry.yaml      : 메타데이터 파일")


if __name__ == "__main__":
    demo_registry()
