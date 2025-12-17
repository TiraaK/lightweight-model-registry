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

### 왜 YAML을 선택했나?

1. **가독성**: 사람이 읽고 편집하기 쉬움
2. **단순성**: 과제 요구사항에 맞는 가벼운 솔루션
3. **파이썬 호환성**: PyYAML 라이브러리로 간단히 처리

SQLite나 더 복잡한 DB는 이 과제 규모에 과도한 엔지니어링이라고 판단했습니다.

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
