# Jina CLIP v2 Game Asset Search

uv + Python으로 실행하는 로컬 게임 이미지 에셋 자연어 검색 데모입니다.
파일명이나 태그를 검색에 사용하지 않고 `jinaai/jina-clip-v2` 이미지·텍스트 벡터의
코사인 유사도로 정렬합니다. 한국어와 영어 검색을 모두 지원합니다.

## 설치 및 실행

노트북처럼 CUDA GPU를 사용하지 않는 PC에서는 PowerShell에서:

```powershell
cd C:\work\tech-demo\demos\02-clip-asset-search
uv run --extra cpu python app.py
```

NVIDIA GPU를 사용하는 데스크탑에서는 CUDA 13.0 빌드로 실행합니다.

```powershell
cd C:\work\tech-demo\demos\02-clip-asset-search
uv run --extra cu130 python app.py
```

브라우저에서 http://127.0.0.1:7860 을 엽니다.
uv가 선택한 프로필에 맞춰 독립 가상환경과 PyTorch를 준비합니다.
두 프로필은 동시에 선택할 수 없습니다.
첫 인덱싱 또는 검색 시 Hugging Face에서 `jinaai/jina-clip-v2`와 원격 구현 코드를 다운로드합니다.
모델은 약 0.9B 파라미터이므로 인터넷 연결, 디스크 공간과 첫 로딩 시간이 필요합니다.
CUDA GPU가 있으면 자동으로 사용하고, 없으면 CPU로 폴백합니다. 현재 장치는 앱 상단에 표시됩니다.
설치 프로필과 별개로 실행 장치를 강제하려면 `DEVICE`를 지정할 수 있습니다.

```powershell
$env:DEVICE = "cpu"  # auto, cpu, cuda 중 하나
uv run --extra cu130 python app.py
Remove-Item Env:DEVICE
```

기본값은 `auto`입니다. 머신별 설정을 담는 `.env` 파일은 Git에서 제외됩니다.
모델을 한 번 실제 사용해 필요한 지연 로딩 파일까지 받은 뒤 오프라인 실행하려면
`$env:HF_HUB_OFFLINE = "1"`을 설정할 수 있습니다.

1. PNG/JPG/JPEG/WebP 이미지가 들어 있는 폴더의 절대 경로를 입력합니다.
2. **인덱싱 / 다시 인덱싱**을 누릅니다. 하위 폴더도 포함합니다.
3. 한국어 또는 영어 문장으로 검색합니다. 예: `붉은색 회복 포션`, `a rusty sword`.
4. 기본 상위 12개 이미지와 상대 경로, 코사인 유사도를 확인합니다.

투명 이미지는 밝은 회색 배경에 합성합니다. 읽기 실패한 이미지는 건너뛰고 개수를 표시합니다.
인덱스는 프로그램 폴더의 `.cache`에 폴더별로 저장하며 재실행 시 재사용합니다.
이미지 변경은 자동 감지하지 않습니다. 추가·수정·삭제 후에는 다시 인덱싱하세요.
UI에는 로컬 이미지 파일 대신 생성한 썸네일을 전달합니다.

## 명령줄 실행

```powershell
uv run --extra cpu python app.py --index C:\assets\icons
uv run --extra cpu python app.py --folder C:\assets\icons --search "붉은색 회복 포션"
```

이 저장소의 데모용 혼합 corpus를 사용하려면 다음 경로를 입력합니다.

```text
C:\work\tech-demo\sample-data\assets\showcase-corpus
```

## 시연 및 확인

- 검·방패·포션·상자처럼 서로 다른 종류의 개별 이미지 수십~수백 개로 시작합니다.
- 파일명이 숫자인 이미지도 의미로 검색되는지 확인합니다.
- 같은 종류의 색상 변형을 섞어 검색 결과를 비교합니다.
- 앱을 재시작한 다음 동일한 폴더에서 인덱싱 없이 검색해 캐시 재사용을 확인합니다.
- 빈 폴더는 오류를 표시하고, 손상 이미지는 인덱싱 결과에 건너뛴 파일로 표시하는지 확인합니다.

Jina CLIP v2는 512×512 입력과 다국어 이미지 검색을 지원하며, 이 데모는 512차원
Matryoshka 임베딩을 사용합니다. 작은 픽셀아트나 세부 속성 구분의 품질은 실제 에셋으로 확인해야 합니다.
점수는 확률이 아니며 관련 없는 문장에도 가장 가까운 이미지가 반환됩니다.
스프라이트 시트는 한 장으로 취급하고 3D 원본 파일은 처리하지 않습니다.

`jinaai/jina-clip-v2` 모델은 **CC BY-NC 4.0** 라이선스입니다. 이 구성은 로컬·비상업적
기술 데모 용도이며, 상업적 사용 전에는 Jina AI의 라이선스 조건을 별도로 확인해야 합니다.

## 검증 상태

- Windows, Python 3.12, PyTorch CUDA 환경에서 RTX 5060 Ti 인식을 확인했습니다.
- GPU에서 이미지와 한국어 텍스트를 각각 512차원으로 인코딩했습니다.
- 노트북 CPU 환경의 단위 테스트는 `uv run --extra cpu python -m unittest -v test_app.py`로 실행합니다.
