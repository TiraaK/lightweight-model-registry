# 모델 레지스트리 시스템

ML 모델 파일을 체계적으로 관리하기 위한 간단하고 효율적인 레지스트리 시스템

## 프로젝트 개요

하루에도 수십 개씩 생성되는 모델 파일(`.pth`, `.pt`)들을 체계적으로 정리하고 관리하기 위한 시스템입니다. 파일명으로 버전을 구분하던 비효율적인 방식(`model_v1.pt`, `model_final.pt`, `model_real_final.pt` 등)에서 벗어나, 중앙화된 레지스트리로 모델과 메타데이터를 체계적으로 관리합니다.

## 주요 기능

- **모델 등록**: 모델 파일을 레지스트리에 등록하고 메타데이터 저장
- **버전 관리**: 시맨틱 버저닝 자동 지원 (v1, v2, v3...)
- **모델 조회**: 이름/버전으로 모델 검색, "latest" 쿼리 지원
- **메타데이터 관리**: 성능 메트릭, 데이터셋, 프레임워크 정보 등 저장
- **모델 패밀리**: 동일 모델의 여러 버전을 패밀리 단위로 그룹화

## 설치 및 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

필요한 패키지:
- `torch>=2.0.0` - PyTorch (모델 로딩용)
- `torchvision>=0.15.0` - 사전 학습 모델 다운로드
- `pyyaml>=6.0` - 메타데이터 저장

### 2. 데모 실행

```bash
python example.py
```

데모 스크립트는 다음을 자동으로 수행합니다:
1. ResNet-18, MobileNetV2 사전 학습 모델 다운로드
2. 모델 레지스트리에 등록
3. 다양한 조회 기능 시연
4. 결과를 콘솔에 출력

### 3. 생성되는 파일

실행 후 다음 파일들이 생성됩니다:
- `./models/` - 모델 파일 저장소 (버전별 폴더 구조)
- `./registry.yaml` - 메타데이터 파일 (YAML 형식)
- `*.pth` 파일들 - 다운로드된 모델 파일

## 사용 예제

### 기본 사용법

```python
from registry import ModelRegistry

# 1. 레지스트리 초기화
registry = ModelRegistry(
    storage_path="./models",
    metadata_file="./registry.yaml"
)

# 2. 모델 등록
registry.register(
    name="resnet18",
    model_path="resnet18_pretrained.pth",
    framework="pytorch",
    metrics={"accuracy": 0.92, "f1": 0.89},
    description="사전 학습된 ResNet-18"
)

# 3. 모델 조회
model_info = registry.get("resnet18", "latest")
print(model_info["file_path"])  # 파일 경로
print(model_info["metrics"])    # 성능 메트릭

# 4. 모델 목록
all_models = registry.list()              # 전체 모델
resnet_versions = registry.list("resnet18")  # 특정 모델의 모든 버전

# 5. 요약 출력
registry.print_summary()
```

### API 명세

| 메서드 | 설명 | 예시 |
|--------|------|------|
| `register(name, model_path, ...)` | 모델 등록 | `registry.register("resnet18", "model.pth")` |
| `get(name, version="latest")` | 모델 조회 | `registry.get("resnet18", "v2")` |
| `list(name=None)` | 모델 목록 조회 | `registry.list()` |
| `list_families()` | 모델 패밀리 목록 | `registry.list_families()` |
| `print_summary()` | 요약 정보 출력 | `registry.print_summary()` |

### 🔍 핵심 로직 상세: 모델 조회 (`get`) 프로세스

사용자가 `registry.get("resnet18", "latest")`를 호출했을 때, 시스템 내부는 다음과 같은 단계로 처리됩니다.

| 단계 | 동작 (Operation) | 상세 로직 |
|:---:|---|---|
| **1** | **메타데이터 검색** | `registry.yaml` 메모리 캐시(`self.metadata`)에서 "resnet18" 키가 있는지 확인합니다. |
| **2** | **버전 결정 (Resolution)** | 요청 버전이 `"latest"`인 경우, 등록된 버전 목록(`v1`, `v2`...) 중 **숫자(int)가 가장 큰 버전**을 자동으로 선택합니다. (예: `v2`) |
| **3** | **정보 추출** | 선택된 버전(`v2`)의 메타데이터 딕셔너리를 복사(Copy)하여 가져옵니다. |
| **4** | **경로 조합 (Path Joining)** | YAML에는 상대 경로(`models/resnet/v1/...`)만 저장되어 있습니다. 이를 현재 실행 위치(`base_path`)와 결합하여 **OS에 맞는 절대 경로**로 변환합니다. |
| **5** | **파일 검증 (Validation)** | 변환된 경로에 실제 파일(`.pth`)이 존재하는지 최종 확인 후, 정보를 반환합니다. |

## 프로젝트 구조

```
model_registry_assignment/
├── README.md                      # 본 파일
├── SYSTEM_DESIGN.md               # 시스템 설계 문서
├── requirements.txt               # 의존성
├── registry.py                    # 레지스트리 클래스 구현
├── example.py                     # 데모 스크립트
├── models/                        # 모델 저장소
│   ├── resnet18/
│   │   ├── v1/
│   │   │   └── resnet18_pretrained.pth
│   │   └── v2/
│   │       └── resnet18_pretrained.pth
│   └── mobilenetv2/
│       └── v1/
│           └── mobilenetv2_pretrained.pth
└── registry.yaml                  # 메타데이터 파일
```

## 디자인 선택

### 메타데이터 전략 및 스키마 설계 의도

본 프로젝트는 모델의 **재현성(Reproducibility)**과 **추적성(Traceability)**을 최우선으로 고려하여 메타데이터 스키마를 설계했습니다. `registry.yaml`의 각 필드는 단순 기록을 넘어, 다음과 같은 구체적인 목적을 가집니다.

#### 1. 계층적 구조 (Hierarchical Structure)
- **설계:** `Model Family (이름)` → `Version (버전)` → `Details (상세 정보)`
- **의도:** 파일 시스템의 디렉토리 구조(`models/name/version/`)와 1:1로 매핑하여, **논리적 데이터(YAML)와 물리적 파일(Disk) 간의 일관성**을 직관적으로 유지합니다.

#### 2. 필드별 설계 의도 (Schema Rationale)
- **`architecture` & `input_shape` (사용성 보장):**
  - 모델 가중치(`pth`)만으로는 모델 구조나 입력 크기를 알 수 없어 추론 시 에러가 발생하기 쉽습니다. 이를 명시하여 사용자가 **별도 문서 없이도 모델을 즉시 로드하고 사용**할 수 있도록 했습니다.
- **`metrics` (의사결정 지원):**
  - 여러 버전 중 배포할 모델을 선택할 때, 정량적인 지표(Accuracy, F1-score 등)를 통해 **데이터 기반의 의사결정**을 내릴 수 있게 합니다.
- **`dataset` & `framework` (리니지 추적):**
  - "이 모델이 어떤 데이터로 학습되었는가?"에 대한 기원(Lineage)을 추적하여, 데이터 편향 문제나 프레임워크 호환성 이슈 발생 시 **원인 파악(Root Cause Analysis)**을 용이하게 합니다.
- **`registered_at` (감사 및 이력):**
  - 모델의 생성 및 등록 시점을 기록하여 프로젝트의 진행 흐름을 시간순으로 파악합니다.

### 왜 YAML을 선택했나?

1. **가독성**: 사람이 읽고 편집하기 쉬움
2. **단순성**: 과제 요구사항에 맞는 가벼운 솔루션
3. **파이썬 호환성**: PyYAML 라이브러리로 간단히 처리

SQLite나 더 복잡한 DB는 이 과제 규모에 과도한 엔지니어링이라고 판단했습니다.

### ⚡ 성능 및 UX 최적화 (Optimization)

- **중복 다운로드 방지 (Smart Caching)**:
  - 데모 스크립트(`example.py`) 실행 시 로컬에 모델 가중치 파일이 이미 존재하는지 검사(`os.path.exists`)합니다.
  - 파일이 존재할 경우 불필요한 네트워크 다운로드와 디스크 쓰기 작업을 자동으로 건너뛰어, **재실행 속도를 획기적으로 단축**하고 네트워크 리소스를 절약합니다.

### 버전 관리 전략

- 자동 버전 증가: `v1` → `v2` → `v3`
- 명시적 버전 지정도 가능
- "latest" 쿼리로 최신 버전 자동 조회

## AI 도구 사용 노트

### 사용한 AI 도구

- **Claude Code (Anthropic)** - 코드 작성 및 리팩토링 지원

### 특히 도움이 되었던 프롬프트

1. **"모델 레지스트리의 핵심 기능을 구현하는 Python 클래스를 작성해줘. 등록, 조회, 목록 기능이 필요하고, 메타데이터는 YAML로 저장해."**
   - 핵심 클래스 구조와 메서드 설계에 도움

2. **"버전 자동 증가 로직을 추가해줘. v1, v2, v3 형식으로 자동으로 증가하되, 명시적 버전 지정도 가능하게."**
   - `_get_next_version()` 메서드 구현에 활용

3. **"사용자 친화적인 API 명세서를 표 형태로 작성해줘."**
   - README 작성 시 문서화 방식 개선

### AI 제안에 동의하지 않았거나 수정한 사례

**상황**: 처음에 Claude가 복잡한 에러 핸들링과 로깅 시스템을 제안했습니다.

**수정 이유**: 과제 FAQ에 "4-6시간 작업, 과도한 엔지니어링 지양, 코어 기능에 집중"이라는 가이드라인이 있었기 때문에, 에러 핸들링을 기본적인 수준(`FileNotFoundError` 체크 정도)으로 축소하고, 로깅 대신 간단한 print 문으로 변경했습니다.

**결과**: 코드가 더 읽기 쉽고 유지보수하기 간단해졌으며, 과제의 핵심 요구사항에 집중할 수 있었습니다.

## 등록된 모델

1. **ResNet-18** (~45MB)
   - 용도: 일반 이미지 분류
   - 데이터셋: ImageNet
   - Top-1 Accuracy: 69.7%

2. **MobileNetV2** (~14MB)
   - 용도: 경량 이미지 분류 (모바일/엣지 환경)
   - 데이터셋: ImageNet
   - Top-1 Accuracy: 71.8%

## 라이선스

This project is for educational purposes (SwiftSight AI Research Engineer Assignment).
