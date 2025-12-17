# 모델 레지스트리 시스템 (Model Registry System)

ML 모델 파일(`.pth`, `.pt`)의 무분별한 버전 파편화를 막고, 중앙에서 체계적으로 관리하기 위한 **경량화된 레지스트리 시스템**입니다. 파일 시스템 기반으로 동작하며, 메타데이터와 모델 아티팩트를 1:1로 매핑하여 관리합니다.

---

## 1. 프로젝트 개요

### 해결하고자 하는 문제
- **파일명 혼란:** `model_v1.pt`, `model_final.pt`, `real_final.pt` 등 비직관적인 파일명으로 인한 버전 관리의 어려움.
- **메타데이터 부재:** 모델 파일만 남고, 해당 모델의 학습 데이터셋이나 성능 지표(Accuracy 등)가 유실되는 문제.
- **재현성 부족:** 과거 모델의 스펙을 파악하기 어려워 실험 재현이 불가능함.

### 솔루션 핵심
- **중앙화된 관리:** 로컬 파일 시스템(`models/`)을 백엔드로 사용하여 모든 모델 관리.
- **불변성(Immutability):** 등록된 버전은 수정되지 않으며, 새로운 변경은 새 버전으로 기록.
- **메타데이터 동기화:** 모델 아티팩트와 성능 지표, 학습 정보를 하나의 단위로 취급.

---

## 2. 주요 기능 및 과제 요구사항 충족 (Core Features)

1. **모델 등록 및 패밀리 지원 (Model Registration)**
   - 메타데이터(이름, 버전, 프레임워크)와 함께 모델을 등록하며, `resnet18`과 같은 **모델 패밀리(Family)** 단위로 관리합니다.
2. **모델 조회와 Fallback (Model Retrieval)**
   - 이름과 버전으로 모델을 가져오며, **"latest" 쿼리** 및 저장소 위치 변경에 대응하는 **경로 자동 조정(Fallback)** 기능을 제공합니다.
   - 메타데이터에는 상대 경로만 저장하고, 조회(`get`) 시점의 실행 위치에 맞춰 절대 경로를 동적으로 계산합니다. 이를 통해 프로젝트 위치가 바뀌어도 유연하게 동작합니다. 
3. **메타데이터 및 리니지 관리 (Metadata Management)**
   - 아키텍처, 입력 Shape, 성능 지표를 저장하고, **리니지(Lineage)**(데이터셋, 날짜 등)를 추적하여 모델의 기원(Lineage)을 관리합니다.
4. **저장소 백엔드 (Storage Backend)**
   - 외부 DB 없이 **로컬 파일 시스템(Local File System)**을 백엔드로 사용하여 완벽하게 동작합니다.
5. **시맨틱 버전 관리 (Versioning)**
   - `v1`, `v2` 형식의 **시맨틱 버저닝**을 지원하며, 특정 모델의 모든 버전 목록을 나열할 수 있습니다.

---

## 3. 🚀 차별화 요소 및 최적화 (Advanced Features)

과제 요구사항을 넘어, 실제 프로덕션 환경을 고려하여 추가로 구현한 기능들입니다.

### 1. 목적별 조회 전략 분리 (`latest` vs `best`) << 로마자로 변경할수있니 ? ## 태크의 숫자들과 헷갈림.
- **문제 의식:** 단순히 "가장 나중에 등록된 모델(`latest`)"이 항상 "가장 성능 좋은 모델"인 것은 아닙니다.
- **해결책:** 조회 목적에 따라 옵션을 분리하여 유연성을 확보했습니다.
    - `version="latest"`: 시간상 가장 최신 버전을 조회 (실험 이력 추적용)
    - `version="best"`: 메트릭(Accuracy 등)이 가장 높은 버전을 조회 (실제 배포용)
    - 

### 2. 스마트 캐싱 (Smart Caching)
- 데모 실행 시 `os.path.exists` 체크를 통해 대용량 모델 파일의 **중복 다운로드를 방지**합니다.
- 불필요한 네트워크 트래픽을 줄이고 반복 실행 속도를 획기적으로 개선했습니다.

### 3. 데이터 정합성 검증 (Safety Fallback)
- **개념:** 메타데이터(`registry.yaml`)와 실제 파일 시스템(`models/`) 간의 상태가 일치하지 않는 상황을 방지합니다.
- **동작:** 모델 조회 시 메타데이터에 정보가 있더라도 실제 파일이 삭제되었는지 `os.path.exists`로 한 번 더 검증(Double Check)하고, 파일이 없을 경우 명시적인 경고(Warning)를 제공합니다.

### 4. pretrained 폴더와 models 폴더의 분리

---

## 4. 설치 및 실행

### 사전 요구사항
* Python 3.10+ (권장)
* 로컬 파일 시스템 접근 권한

### 설정 및 데모 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 데모 스크립트 실행
python example.py
```

> **참고:** 데모 스크립트는 **ResNet-18** 및 **MobileNetV2** 모델을 자동으로 다운로드하여 레지스트리에 등록하고, 조회 및 목록 출력 기능을 시연합니다.

---

## 5. 사용 가이드 및 API 명세

### Python API 예제

```python
from registry import ModelRegistry

# 1. 레지스트리 초기화
registry = ModelRegistry(storage_path="./models", metadata_file="./registry.yaml")

# 2. 모델 등록 (성능 지표 포함)
registry.register(name="resnet18", model_path="r18.pth", metrics={"acc": 0.95})

# 3. 최고 성능 모델 조회
best_model = registry.get("resnet18", version="best")
print(f"Best Version: {best_model['version']} (Acc: {best_model['metrics']['acc']})")
```

### API 상세 명세

| 메서드 | 설명 | 파라미터 (Input) | 실행 결과 (Output) |
|:---:|---|---|---|
| **`register`** | 모델을 등록하고<br>새 버전을 부여합니다. | `name="resnet18"`<br>`model_path="./r18.pth"`<br>`metrics={"acc": 0.92}` | 등록된 ID 반환:<br>`"resnet18/v1"` |
| **`get`** | 특정 모델의<br>메타데이터를 조회합니다. | `name="resnet18"`<br>`version="latest"`<br>*(또는 "best", "v1")* | 모델 정보 딕셔너리:<br>`{ "version": "v1", "file_path": "...", ... }` |
| **`list`** | 등록된 모델의<br>목록을 반환합니다. | `name="resnet18"`<br>*(생략 시 전체 목록)* | 리스트 반환:<br>`["resnet18/v1", "resnet18/v2"]` |
| **`list_families`** | 관리 중인 모델 패밀리<br>이름들을 반환합니다. | *(없음)* | 리스트 반환:<br>`["resnet18", "mobilenetv2"]` |

---

## 6. 시스템 아키텍처

### 프로젝트 구조
```
model_registry_assignment/
├── registry.py        # [Core] 레지스트리 구현 (등록, 조회, 버전 관리)
├── registry.yaml      # [Data] 메타데이터 저장소 (YAML)
├── models/            # [Storage] 로컬 파일 시스템 저장소
├── example.py         # [Demo] 기능 시연 스크립트
└── requirements.txt   # 의존성 목록
```

### 데이터 설계 (YAML 선택 이유)
- **가독성:** 사람이 직접 읽고 수정하기 쉬움.
- **경량성:** 별도의 DB 설치 없이 로컬 환경에서 완벽하게 동작.
- **적합성:** 복잡한 쿼리보다 단일 파일 기반의 관리가 과제 규모에 적합.

> 상세한 설계 결정과 트레이드오프 분석은 `SYSTEM_DESIGN.md` 문서를 참고해주세요.

---

## 7. 등록된 모델 정보 (Demo)

| 모델 패밀리 | 버전 | 용도 | 데이터셋 | 크기 |
|-------------|------|------|----------|------|
| **ResNet-18** | v1, v2 | 이미지 분류 | ImageNet | ~45MB |
| **MobileNetV2** | v1 | 모바일/경량화 | ImageNet | ~14MB |

---

## 8. 라이선스 및 참고
본 프로젝트는 **[SwiftSight] AI Research Engineer 과제** 제출용으로 작성되었습니다.
