# 03 · World Tool Calling

자연어 명령을 Groq 모델에 보내고, 모델이 선택한 Python 툴로 간단한 월드 데이터를 변경하는 CLI 데모입니다.
3D 렌더러 없이 **입력 → tool call → 함수 실행 → 결과 전달 → 후속 호출/최종 답변**을 확인합니다.

## 실행

Python 3.11 이상과 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```powershell
cd C:\work\tech-demo\demos\03-tool-calling
Copy-Item .env.example .env
# .env의 GROQ_API_KEY에 본인의 Groq API 키를 입력
uv run python app.py
```

`.env`는 Git에서 제외됩니다. 이미 복사했다면 다시 복사하지 말고 기존 파일을 편집하세요.
`GROQ_MODEL`로 모델을 바꿀 수 있으며 기본값은 `openai/gpt-oss-120b`입니다.
OpenAI Python SDK의 `base_url`을 `https://api.groq.com/openai/v1`로 설정하여 사용합니다.
OpenAI API 키는 필요하지 않습니다.

한 번의 명령만 실행하고 최종 월드 JSON을 출력하려면:

```powershell
uv run python app.py --prompt "나무를 원점에 놓고, 바위를 x=3, y=0, z=2에 놓아줘"
```

## 대화 예시

```text
you> 나무를 원점에 놓아줘
you> 그 나무를 오른쪽으로 3만큼 옮겨줘
you> 빛나는 파란 수정 에셋을 만들어서 x=2, y=0, z=4에 배치해줘
you> 현재 장면에 뭐가 있어?
you> 나무를 삭제해줘
you> /scene
you> /quit
```

매 호출마다 `[tool] 이름(인자)`와 `[result] JSON`이 출력됩니다.
`/scene`은 LLM 호출 없이 현재 데이터를 출력합니다. `/quit`은 종료합니다.
좌표는 절대 `[x, y, z]`이며 +x는 오른쪽, +y는 위, +z는 전방입니다.
상대 이동은 모델이 현재 위치를 조회한 후 절대 좌표로 계산합니다.

## 툴

| 툴 | 인자 | 동작 |
|---|---|---|
| `search_asset` | `query` | 로컬 카탈로그에서 한·영 키워드 부분 문자열 검색 |
| `generate_asset` | `description` | 가상 에셋 메타데이터 생성 |
| `place_asset` | `asset_id`, `position` | 에셋의 새 인스턴스 배치, `object_id` 반환 |
| `move_asset` | `object_id`, `position` | 배치된 객체를 절대 좌표로 이동 |
| `delete_asset` | `object_id` | 배치된 객체 삭제, 카탈로그 에셋은 유지 |
| `inspect_scene` | 없음 | 전체 에셋 및 배치 객체 조회 |

예를 들어 나무 하나를 배치하면 객체 데이터는 아래 형태입니다.

```json
{
  "assets": {
    "tree": {"id": "tree", "description": "나무 tree oak", "source": "catalog"}
  },
  "objects": {
    "object_1": {"id": "object_1", "asset_id": "tree", "position": [0, 0, 0]}
  }
}
```

위 예시는 일부만 발췌한 것으로, 초기 카탈로그에는 나무·바위·집이 있습니다.
`asset_id`는 재사용 가능한 에셋, `object_id`는 월드에 배치된 개별 객체입니다.
동일 에셋을 여러 번 배치해도 각각 따로 이동·삭제할 수 있습니다.

## 코드 읽는 순서

1. `world.py`: 툴 JSON Schema, 데이터, 실제 변경 함수
2. `agent.py`: 메시지 이력과 tool-result 반복 루프
3. `app.py`: Groq 클라이언트 설정과 CLI

별도 에이전트 프레임워크는 사용하지 않습니다. 모델이 요청한 툴만 허용하고,
JSON Schema로 필수 인자·타입·좌표를 검사합니다. 잘못된 호출은 오류 결과로 모델에 전달합니다.

## 검증

위 데모 폴더에서 실행합니다. API 키 없이 테스트할 수 있습니다.

```powershell
uv run python -m unittest discover -s tests -v
```

월드 변경·잘못된 인자·대화 유지·다중 툴 결과 연결·반복 제한을 검사합니다.
오프라인 테스트의 모델 응답은 고정된 응답이므로 실제 LLM의 툴 선택 품질을 검증하지는 않습니다.

## 현재 범위 및 제한

- 월드와 대화는 메모리에만 저장되며 종료하면 초기화됩니다.
- 검색은 임베딩 검색이 아닙니다. 짧은 한·영 키워드를 사용합니다.
- 생성은 메타데이터만 만듭니다. 실제 메시·이미지·파일은 생성하지 않습니다.
- 기본적으로 사용자 요청당 모델 응답을 최대 10회 처리합니다. 한도에 도달하면 알립니다.
- 여러 작업은 순서대로 즉시 반영합니다. 뒤의 호출이 실패해도 앞의 변경을 되돌리지 않습니다.
- API 오류는 CLI에 표시합니다. 키·모델 권한·사용 한도·네트워크를 확인한 후 다시 요청하세요.
- 자연어와 월드 조회 결과는 Groq로 전송됩니다.

## 참고

- [Groq OpenAI compatibility](https://console.groq.com/docs/openai)
- [Groq tool use](https://console.groq.com/docs/tool-use/overview)
