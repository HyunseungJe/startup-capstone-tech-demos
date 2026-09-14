# Capability Demos

다음 미팅에서 구현 가능성을 빠르게 확인하기 위한 소형 프로토타입 모음입니다. 각 데모는 자체 실행 방법과 의존성을 가지며, 공통 코드가 실제로 반복될 때만 `shared/`로 옮깁니다.

## Demos

| Demo | Goal | Stack | Status |
|---|---|---|---|
| [01 Pose to Animation · Python](demos/01-pose-to-animation/) | Video → common pose JSON → skeleton MP4 | Python, uv, Streamlit, MediaPipe | Prototype |
| [02 CLIP Asset Search · Python](demos/02-clip-asset-search/) | Natural language → similar game asset images | Python, uv, Gradio, CLIP | Prototype |

새 데모는 `demos/02-name/`, `demos/03-name/`처럼 번호가 붙은 독립 폴더로 추가합니다. 각 폴더에는 최소한 목표, 실행법, 입력, 출력, 현재 상태를 설명하는 `README.md`를 둡니다.

## Repository layout

```text
.
├─ demos/
│  ├─ 01-pose-to-animation/
│  └─ 02-clip-asset-search/
├─ sample-data/
│  └─ videos/
├─ shared/
└─ README.md
```

공통 테스트 자료는 GitHub에 포함하지 않습니다. [Google Drive에서 sample-data 받기](https://drive.google.com/drive/folders/1J1n67Ce8aDmwh-ePbjPJ1pfDyyaEnTL9) 후 저장소 루트의 `sample-data/`에 둡니다. 새 데모가 특정 데이터의 수정본을 필요로 하면 해당 데모 내부에 두어 다른 데모의 기준 입력을 바꾸지 않습니다.

## Working convention

- 작업 요청에는 가능하면 대상 데모 폴더를 명시합니다.
- 데모별 환경과 의존성은 해당 폴더 안에서 관리합니다.
- 데모 사이의 중복은 허용하고, 반복 사용이 확인된 코드만 `shared/`로 이동합니다.
- 생성 결과, 캐시, 로컬 가상환경은 커밋하지 않습니다.
