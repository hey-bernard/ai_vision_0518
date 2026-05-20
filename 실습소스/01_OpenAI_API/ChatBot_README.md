# OpenAI API 챗봇 예제

OpenAI Chat Completions API를 사용하는 대화형 챗봇 예제입니다.  
콘솔 버전(`chatbot.py`)과 Tkinter GUI 버전(`chatbot_gui.py`)을 제공합니다.

---

## 프로젝트 구성

| 파일 | 설명 |
|------|------|
| `chatbot.py` | 터미널(콘솔)에서 대화하는 챗봇 |
| `chatbot_gui.py` | Tkinter GUI 챗봇 |
| `01_OpenAI API Key 사용.ipynb` | API 키 설정 및 API 호출 학습용 노트북 |

---

## 주요 기능

### 공통 기능 (`chatbot.py`, `chatbot_gui.py`)

- **OpenAI API 연동**  
  `gpt-4o-mini` 모델로 Chat Completions API를 호출합니다.

- **환경 변수로 API 키 관리**  
  `python-dotenv`로 `C:/env/.env` 파일에서 `OPENAI_API_KEY`를 읽습니다.  
  소스 코드에 API 키를 직접 넣지 않습니다.

- **대화 히스토리(맥락) 유지**  
  이전 질문·답변을 `history` 리스트에 저장하고, 매 요청마다 전체 대화를 API에 전달합니다.  
  연속 질문(예: 이름을 말한 뒤 “내 이름이 뭐였지?”)에도 이전 맥락을 반영합니다.

- **`chat_with_gpt` 함수**  
  사용자 메시지를 받아 API를 호출하고, 응답 문자열을 반환합니다.  
  호출 시 `history`에 user/assistant 메시지가 순서대로 추가됩니다.

- **대화 표시 형식**  
  - 사용자: `You:` 접두어  
  - 봇: `Bot:` 접두어  

- **종료**  
  `quit` 입력 시 프로그램을 종료합니다. (대소문자 구분 없음)

### GUI 전용 기능 (`chatbot_gui.py`)

- 스크롤 가능한 채팅 영역
- 입력창 + **전송** / **종료** 버튼
- Enter 키로 메시지 전송
- API 호출 중 입력·전송 버튼 비활성화 (중복 전송 방지)
- 별도 스레드에서 API 호출 → 응답 대기 중에도 창이 멈추지 않음
- 사용자/봇/오류 메시지별 색상·폰트(맑은 고딕) 스타일

---

## 사전 준비

### 1. Python 패키지 설치

```bash
pip install openai python-dotenv
```

GUI 버전은 Python 표준 라이브러리 `tkinter`를 사용합니다.  
(Windows Python 설치 시 보통 포함되어 있습니다.)

### 2. API 키 설정

`C:/env/.env` 파일을 만들고 다음 내용을 작성합니다.

```env
OPENAI_API_KEY=sk-여기에_본인_API_키
```

> 다른 경로의 `.env`를 쓰려면 각 파일의 `load_dotenv("C:/env/.env")` 경로를 수정하세요.

### 3. OpenAI API 키 발급

[OpenAI Platform](https://platform.openai.com/)에서 API 키를 발급받아 `.env`에 등록합니다.

---

## 사용법

### 콘솔 챗봇 (`chatbot.py`)

```bash
python chatbot.py
```

**실행 예**

```
You: 안녕, 나는 철수야.
Bot: 안녕하세요, 철수님! ...

You: 내 이름이 뭐였지?
Bot: 철수라고 하셨어요.

You: quit
```

| 동작 | 방법 |
|------|------|
| 질문 입력 | `You:` 뒤에 메시지 입력 후 Enter |
| 종료 | `quit` 입력 |

---

### GUI 챗봇 (`chatbot_gui.py`)

```bash
python chatbot_gui.py
```

**화면 구성**

1. **상단 헤더** — 제목, 사용 모델 안내  
2. **채팅 영역** — `You:` / `Bot:` 대화 기록 (스크롤 가능)  
3. **하단 입력** — 메시지 입력창, **전송**, **종료** 버튼  

| 동작 | 방법 |
|------|------|
| 메시지 전송 | 입력 후 **전송** 클릭 또는 Enter |
| 종료 | `quit` 입력, **종료** 버튼, 또는 창 닫기 |
| API 응답 대기 | 입력창·전송 버튼이 잠시 비활성화됨 |

---

## 동작 원리

### 대화 히스토리 구조

`history`는 OpenAI API `messages` 형식의 리스트입니다.

```python
[
    {"role": "user", "content": "첫 번째 질문"},
    {"role": "assistant", "content": "첫 번째 답변"},
    {"role": "user", "content": "두 번째 질문"},
    ...
]
```

`chat_with_gpt`는 매 호출마다:

1. 사용자 메시지를 `history`에 추가  
2. `history` 전체를 API에 전달  
3. 응답을 `history`에 assistant 메시지로 추가  
4. 응답 문자열 반환  

### 사용 모델

기본 모델: **`gpt-4o-mini`**  
변경하려면 `chatbot.py` / `chatbot_gui.py`의 `chat_with_gpt` 함수 안 `model=` 값을 수정합니다.

---

## 문제 해결

| 증상 | 확인 사항 |
|------|-----------|
| API 키 오류 | `C:/env/.env` 존재 여부, `OPENAI_API_KEY` 이름·값 확인 |
| `ModuleNotFoundError: dotenv` | `pip install python-dotenv` |
| `ModuleNotFoundError: openai` | `pip install openai` |
| GUI 입력창이 안 보임 | 최신 `chatbot_gui.py` 사용 (하단 입력 영역 grid 고정) |
| GUI에서 `tkinter` 오류 | Python 재설치 시 tcl/tk 포함 옵션 확인 |
| 응답이 느림 | 네트워크·OpenAI 서버 상태 확인 (정상 동작) |

---

## Windows 실행 파일(.exe) 만들기

GUI 챗봇(`chatbot_gui.py`)을 단일 실행 파일로 빌드할 수 있습니다.

### 빌드 결과

| 항목 | 경로 |
|------|------|
| 실행 파일 | `dist\OpenAI-Chatbot.exe` |
| 빌드 설정 | `OpenAI-Chatbot.spec` |
| 빌드 스크립트 | `build_exe.bat` (더블클릭 또는 명령 프롬프트에서 실행) |

### 사전 설치

```bash
pip install pyinstaller openai python-dotenv
```

### 빌드 방법

**방법 1 — 배치 파일**

```text
build_exe.bat
```

**방법 2 — 명령어**

```bash
pyinstaller --clean OpenAI-Chatbot.spec
```

### exe 실행

1. `C:/env/.env`에 `OPENAI_API_KEY`가 설정되어 있어야 합니다. (소스 실행과 동일)
2. `dist\OpenAI-Chatbot.exe`를 더블클릭해 실행합니다.
3. 콘솔 창 없이 GUI만 표시됩니다 (`console=False`).

### 참고

- PyInstaller 빌드 시 `build\`, `dist\` 폴더가 생성됩니다.
- exe 용량은 포함된 라이브러리에 따라 수십~100MB 이상일 수 있습니다.
- 소스 코드를 수정한 뒤에는 `build_exe.bat` 또는 `pyinstaller --clean OpenAI-Chatbot.spec`으로 다시 빌드하세요.

---

## 참고

- API 사용량에 따라 OpenAI 요금이 부과됩니다.  
- 대화가 길어질수록 `history`가 커져 토큰 사용량과 비용이 증가할 수 있습니다.  
- 학습·실험 목적의 예제이며, 운영 환경에서는 히스토리 길이 제한·에러 처리 강화 등을 추가하는 것이 좋습니다.
