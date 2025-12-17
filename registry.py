"""
모델 레지스트리 시스템
ML 모델을 체계적으로 관리하기 위한 레지스트리 클래스 구현
"""

import os
import shutil
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class ModelRegistry:
    """
    모델 파일과 메타데이터를 관리하는 레지스트리 클래스

    Features:
    - 모델 등록 (버전 관리)
    - 모델 조회 (이름/버전, latest 지원)
    - 메타데이터 관리 (YAML 기반)
    - 시맨틱 버저닝
    """

    def __init__(self, storage_path: str = "./models", metadata_file: str = "./registry.yaml"):
        """
        레지스트리 초기화

        Args:
            storage_path: 모델 파일이 저장될 디렉토리
            metadata_file: 메타데이터 YAML 파일 경로
        """
        self.storage_path = Path(storage_path)
        self.metadata_file = Path(metadata_file)

        # 저장소 디렉토리 생성
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 메타데이터 로드 (없으면 빈 딕셔너리)
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """메타데이터 파일 로드"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save_metadata(self):
        """메타데이터 파일 저장"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.metadata, f, allow_unicode=True, default_flow_style=False)

    def register(
        self,
        name: str,
        model_path: str,
        version: Optional[str] = None,
        framework: str = "pytorch",
        metrics: Optional[Dict[str, float]] = None,
        architecture: Optional[str] = None,
        input_shape: Optional[tuple] = None,
        dataset: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """
        모델을 레지스트리에 등록

        Args:
            name: 모델 이름 (패밀리명, 예: resnet18, mobilenetv2)
            model_path: 등록할 모델 파일 경로
            version: 버전 (미지정시 자동 증가, 예: v1, v2, v3)
            framework: 프레임워크 (pytorch, tensorflow 등)
            metrics: 성능 메트릭 딕셔너리 (예: {"accuracy": 0.92, "f1": 0.88})
            architecture: 아키텍처 정보
            input_shape: 입력 shape (예: (3, 224, 224))
            dataset: 학습 데이터셋
            description: 설명

        Returns:
            등록된 모델의 전체 이름 (name/version)
        """
        # 모델 파일 존재 확인
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        # 버전 자동 증가
        if version is None:
            version = self._get_next_version(name)

        # 모델 저장 경로 생성
        model_dir = self.storage_path / name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # 모델 파일 복사
        model_filename = os.path.basename(model_path)
        dest_path = model_dir / model_filename
        shutil.copy2(model_path, dest_path)

        # 메타데이터 구성
        model_key = f"{name}/{version}"

        if name not in self.metadata:
            self.metadata[name] = {}

        self.metadata[name][version] = {
            "name": name,
            "version": version,
            "file_path": str(dest_path.relative_to(self.storage_path.parent)),
            "framework": framework,
            "architecture": architecture or name,
            "input_shape": list(input_shape) if input_shape else None,
            "metrics": metrics or {},
            "dataset": dataset,
            "description": description,
            "registered_at": datetime.now().isoformat(),
        }

        # 메타데이터 저장
        self._save_metadata()

        print(f"✓ 모델 등록 완료: {model_key}")
        return model_key

    def _get_next_version(self, name: str) -> str:
        """모델 패밀리의 다음 버전 번호 반환"""
        if name not in self.metadata or not self.metadata[name]:
            return "v1"

        # 기존 버전들 중 최대값 찾기
        versions = [v for v in self.metadata[name].keys()]
        version_numbers = []

        for v in versions:
            if v.startswith('v') and v[1:].isdigit():
                version_numbers.append(int(v[1:]))

        if version_numbers:
            next_num = max(version_numbers) + 1
        else:
            next_num = 1

        return f"v{next_num}"

    def get(self, name: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        """
        모델 조회

        Args:
            name: 모델 이름
            version: 버전 (기본값: "latest")

        Returns:
            모델 메타데이터 딕셔너리 (없으면 None)
        """
        if name not in self.metadata:
            print(f"✗ 모델을 찾을 수 없습니다: {name}")
            return None

        # latest 버전 처리
        if version == "latest":
            version = self._get_latest_version(name)
            if version is None:
                print(f"✗ {name}의 버전을 찾을 수 없습니다")
                return None

        # 버전 확인
        if version not in self.metadata[name]:
            print(f"✗ 버전을 찾을 수 없습니다: {name}/{version}")
            available = list(self.metadata[name].keys())
            print(f"  사용 가능한 버전: {available}")
            return None

        model_info = self.metadata[name][version].copy()

        # 파일 경로 존재 확인
        file_path = Path(model_info["file_path"])
        if not file_path.exists():
            print(f"⚠ 경고: 메타데이터는 존재하지만 파일을 찾을 수 없습니다: {file_path}")

        return model_info

    def _get_latest_version(self, name: str) -> Optional[str]:
        """모델 패밀리의 최신 버전 반환"""
        if name not in self.metadata or not self.metadata[name]:
            return None

        versions = list(self.metadata[name].keys())

        # 버전 번호 파싱
        version_numbers = []
        for v in versions:
            if v.startswith('v') and v[1:].isdigit():
                version_numbers.append((int(v[1:]), v))

        if not version_numbers:
            # 숫자가 아닌 버전이면 마지막 등록된 것 반환
            return versions[-1]

        # 버전 번호가 가장 큰 것 반환
        version_numbers.sort(reverse=True)
        return version_numbers[0][1]

    def list(self, name: Optional[str] = None) -> List[str]:
        """
        모델 목록 조회

        Args:
            name: 특정 모델의 모든 버전 조회 (None이면 전체 모델)

        Returns:
            모델 목록 (name/version 형식)
        """
        if name:
            # 특정 모델의 모든 버전
            if name not in self.metadata:
                print(f"✗ 모델을 찾을 수 없습니다: {name}")
                return []

            versions = list(self.metadata[name].keys())
            models = [f"{name}/{v}" for v in versions]
        else:
            # 전체 모델 목록
            models = []
            for model_name in self.metadata.keys():
                for version in self.metadata[model_name].keys():
                    models.append(f"{model_name}/{version}")

        return models

    def get_metadata(self, name: str, version: str = "latest") -> Optional[Dict]:
        """
        모델의 상세 메타데이터 조회
        (get()과 동일하지만 명시적인 이름)
        """
        return self.get(name, version)

    def list_families(self) -> List[str]:
        """등록된 모델 패밀리(이름) 목록 반환"""
        return list(self.metadata.keys())

    def print_summary(self):
        """레지스트리 요약 정보 출력"""
        print("\n" + "="*60)
        print("모델 레지스트리 요약")
        print("="*60)

        if not self.metadata:
            print("등록된 모델이 없습니다.")
            return

        for model_name in self.metadata.keys():
            versions = list(self.metadata[model_name].keys())
            print(f"\n📦 {model_name}")
            print(f"   버전: {versions}")

            # latest 버전 정보 출력
            latest_ver = self._get_latest_version(model_name)
            if latest_ver:
                info = self.metadata[model_name][latest_ver]
                print(f"   최신: {latest_ver}")
                print(f"   프레임워크: {info.get('framework', 'N/A')}")
                if info.get('metrics'):
                    metrics_str = ", ".join([f"{k}={v:.3f}" for k, v in info['metrics'].items()])
                    print(f"   메트릭: {metrics_str}")

        print("\n" + "="*60)
