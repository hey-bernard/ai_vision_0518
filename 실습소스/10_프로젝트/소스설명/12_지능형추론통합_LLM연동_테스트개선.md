# 12. 지능형 추론 통합 (LLM 연동) — 소스 설명

대응 노트북: `notebooks/12_지능형추론통합_LLM연동_테스트개선.ipynb`

## 한눈에 보기

Fusion이 준 **숫자 결과**를, LLM이 읽을 수 있는 **문장+JSON 컨텍스트**로 바꾸고  
원인 가설·조치 권고를 받습니다. 실제 API가 없어도 **Mock LLM**으로 테스트합니다.

산출물: `outputs/llm_integration/`

## 초보자를 위한 용어 사전

| 용어 | 쉬운 설명 |
|------|-----------|
| **LLM** | 대규모 언어 모델. 글을 이해하고 생성하는 AI (ChatGPT 계열 등). |
| **프롬프트 (prompt)** | LLM에게 주는 지시문+입력 데이터. |
| **시스템 프롬프트** | 역할·출력 형식을 고정하는 상위 지시. |
| **추상화 계층 / Interface** | 실제 모델이 바뀌어도 같은 방식으로 호출하게 감싸는 껍데기. |
| **Mock** | 진짜 API 대신 가짜 응답을 돌려 개발·테스트하는 대역. |
| **가드레일** | 위험한/잘못된 출력을 막거나 형식을 강제하는 안전장치. |
| **JSON 스키마 검증** | 응답에 필수 키가 있는지, 타입이 맞는지 검사. |

## 구성 요소

| 구성 | 역할 |
|------|------|
| `ContextBuilder` | fusion 숫자 → 검사 컨텍스트 텍스트/객체 |
| `LLMInterface` / Provider | Mock 또는 OpenAI 호환 호출 |
| `ResponseValidator` | summary, urgency, confidence 등 필수 필드 검사 |
| `PromptOptimizer` | 테스트 실패를 보고 프롬프트 개선 |

## 기대 JSON 응답 예

```json
{
  "summary": "한 줄 요약",
  "root_cause_hypothesis": ["가능 원인"],
  "recommended_actions": ["조치"],
  "urgency": "LOW|MEDIUM|HIGH",
  "confidence": 0.0
}
```

## 실행 흐름

```text
fusion_result.json (또는 데모)
→ ContextBuilder
→ LLM (Mock/실API)
→ Validator
→ 리포트 저장 / 테스트 점수
```

## 주의

LLM은 **환각(그럴듯한 거짓말)** 을 할 수 있으므로,  
수치에 근거하게 시키고 confidence·사람 검토 임계값을 둡니다.
