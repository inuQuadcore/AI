# Translation API

채팅 앱 AI 서버의 번역 API. FastAPI 기반으로 T2TT(텍스트→번역)와 S2TT(음성→원문+번역) 모델을 래핑합니다.

---

## 목차

- [배경 및 설계 의도](#배경-및-설계-의도)
- [아키텍처](#아키텍처)
- [API 엔드포인트](#api-엔드포인트)
- [시나리오별 호출 흐름](#시나리오별-호출-흐름)
- [프로젝트 구조](#프로젝트-구조)
- [개발 환경 실행](#개발-환경-실행)
- [배포](#배포)
- [실제 모델 서버 연동](#실제-모델-서버-연동)
- [환경변수 레퍼런스](#환경변수-레퍼런스)

---

## 배경 및 설계 의도

### AI 모델 구성

AI 서버는 두 가지 모델만 제공합니다:

| 모델 | 입력 | 출력 |
|------|------|------|
| **T2TT** (Text-to-Translated-Text) | 텍스트 원문 | 번역된 텍스트 |
| **S2TT** (Speech-to-Translated-Text) | 음성 파일 | 원문 전사 + 번역된 텍스트 |

### S2TT 내부 동작

S2TT 모델은 내부적으로 2단계 처리를 합니다:

```
음성 입력
  └──→ [1단계: STT] 원문 텍스트 전사
         └──→ [2단계: 번역] 번역된 텍스트
```

**중요:** 이 서버의 S2TT 응답에는 두 단계의 결과가 모두 포함됩니다 (`original_text` + `translated_text`). 이를 통해 별도의 STT 모델 없이도 음성 메시지 원문을 채팅 UI에 표시할 수 있습니다.

---

## 아키텍처

```
프론트엔드 (채팅 앱)
    │
    │ HTTP 요청
    ▼
┌─────────────────────────────────┐
│  Translation API (이 서버)       │
│  FastAPI + uvicorn              │
│                                 │
│  ┌─────────────────────────┐   │
│  │ API 레이어 (라우터)      │   │
│  │ /api/v1/translate/      │   │
│  └────────────┬────────────┘   │
│               │                 │
│  ┌────────────▼────────────┐   │
│  │ 서비스 레이어            │   │
│  │ - 언어 코드 검증         │   │
│  │ - mock/실제 모델 분기    │   │
│  │ - 에러 처리/타임아웃     │   │
│  └────────────┬────────────┘   │
│               │                 │
│  ┌────────────▼────────────┐   │
│  │ Mock 모듈 (개발용)       │   │
│  │ 또는 실제 모델 서버 호출 │   │
│  └─────────────────────────┘   │
└─────────────────────────────────┘
    │
    │ HTTP 요청 (USE_MOCK=False 시)
    ▼
AI 모델 서버 (T2TT / S2TT)
```

---

## API 엔드포인트

### 1. POST `/api/v1/translate/text` — 텍스트 번역 (T2TT)

**Request**
```json
{
  "text": "Hello, how are you?",
  "source_lang": "en",
  "target_lang": "ko"
}
```

**Response 200**
```json
{
  "translated_text": "안녕, 잘 지내?"
}
```

**Error Responses**

| 코드 | error 값 | 설명 |
|------|----------|------|
| 400 | `unsupported_language` | 지원하지 않는 언어 코드 |
| 502 | `model_error` / `model_unavailable` | 모델 서버 오류 |
| 504 | `model_timeout` | 모델 서버 응답 시간 초과 |

---

### 2. POST `/api/v1/translate/speech` — 음성 번역 (S2TT)

**Request** — `multipart/form-data`

| 필드 | 타입 | 설명 |
|------|------|------|
| `file` | audio file | 음성 파일 (wav, mp3, ogg, webm, m4a) |
| `source_lang` | string | 음성 원본 언어 코드 (예: `en`) |
| `target_lang` | string | 번역 대상 언어 코드 (예: `ko`) |

**Response 200**
```json
{
  "original_text": "Hello, how are you?",
  "translated_text": "안녕, 잘 지내?"
}
```

**Error Responses**

| 코드 | error 값 | 설명 |
|------|----------|------|
| 400 | `invalid_audio_format` | 지원하지 않는 오디오 형식 |
| 400 | `unsupported_language` | 지원하지 않는 언어 코드 |
| 413 | `file_too_large` | 파일 크기 25MB 초과 |
| 502 | `model_error` / `model_unavailable` | 모델 서버 오류 |
| 504 | `model_timeout` | 모델 서버 응답 시간 초과 (음성 처리는 최대 60초) |

---

### 3. GET `/health` — 서버 상태 확인

**Response 200**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "mock_mode": true
}
```

---

## 시나리오별 호출 흐름

채팅 앱의 4가지 번역 시나리오에 대한 API 호출 방식입니다.

### 번역 ON + 텍스트 메시지

```
상대방 메시지 수신
    └──→ POST /api/v1/translate/text
           └──→ 로딩 인디케이터 표시
                  └──→ 응답 수신 시 translated_text 말풍선 아래 표시
```

### 번역 OFF + 텍스트 메시지

```
상대방 메시지 수신 (원문만 표시)
    └──→ [번역 보기] 클릭
           └──→ POST /api/v1/translate/text
                  └──→ 로딩 인디케이터 표시
                         └──→ 응답 수신 시 translated_text 표시
```

### 번역 ON + 음성 메시지

```
재생 버튼 클릭
    └──→ POST /api/v1/translate/speech
           └──→ 로딩 인디케이터 표시
                  └──→ 응답 수신 시:
                         ├── original_text 표시 (STT 전사 결과)
                         └── translated_text 표시 (번역 결과)
```

### 번역 OFF + 음성 메시지

```
재생 버튼 클릭
    └──→ POST /api/v1/translate/speech  ← 원문 + 번역 모두 받음
           └──→ 로딩 인디케이터 표시
                  └──→ 응답 수신 시:
                         ├── original_text 표시
                         └── translated_text는 프론트에서 숨김 처리
                                └──→ [번역 보기] 클릭 시 즉시 표시 (추가 API 호출 없음)
```

> **설계 포인트:** 번역 OFF + 음성의 경우에도 서버에서 번역까지 한 번에 처리합니다. S2TT 모델이 내부적으로 STT → 번역을 순차 처리하기 때문에 추가 비용이 크지 않으며, 사용자가 [번역 보기]를 클릭했을 때 즉시 응답할 수 있습니다.

---

## 지원 언어

| 코드 | 언어 |
|------|------|
| `ko` | 한국어 |
| `en` | 영어 |
| `ja` | 일본어 |
| `zh` | 중국어 |
| `es` | 스페인어 |
| `fr` | 프랑스어 |
| `de` | 독일어 |

---

## 프로젝트 구조

```
AI/
├── app/
│   ├── main.py              # FastAPI 앱 진입점, CORS, 라우터 등록
│   ├── core/
│   │   └── config.py        # 환경변수 기반 설정 (모델 URL, 언어 목록 등)
│   ├── api/
│   │   └── v1/
│   │       └── translate.py # API 엔드포인트 정의 (라우터)
│   ├── services/
│   │   └── translate.py     # 비즈니스 로직 (검증, mock/실제 분기, 에러 처리)
│   ├── schemas/
│   │   └── translate.py     # Pydantic 요청/응답 스키마
│   └── models/
│       └── mock.py          # 개발용 Mock 모델 (더미 응답 반환)
├── .env.example             # 환경변수 템플릿
├── requirements.txt         # Python 의존성
├── Dockerfile               # 컨테이너 빌드 설정
├── docker-compose.yml       # 로컬 컨테이너 실행 설정
└── README.md
```

---

## 개발 환경 실행

### 사전 요구사항

- Python 3.11+

### 설치 및 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경변수 설정
cp .env.example .env
# .env 파일에서 필요한 값 수정

# 3. 서버 실행 (Mock 모드 — 모델 서버 없이 동작)
uvicorn app.main:app --reload --port 8000
```

### API 문서 확인

서버 실행 후 브라우저에서:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 배포

### Docker로 실행

```bash
# 이미지 빌드
docker build -t translation-api .

# 컨테이너 실행
docker run -d \
  -p 8000:8000 \
  -e USE_MOCK=False \
  -e T2TT_MODEL_URL=http://ai-model-server:8001/t2tt \
  -e S2TT_MODEL_URL=http://ai-model-server:8001/s2tt \
  translation-api
```

### Docker Compose로 실행

```bash
# 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 종료
docker-compose down
```

---

## 실제 모델 서버 연동

현재는 `USE_MOCK=True`로 더미 응답을 반환합니다. 실제 모델 서버가 준비되면:

### 1단계: `.env` 수정

```env
USE_MOCK=False
T2TT_MODEL_URL=http://실제-모델-서버-주소:8001/t2tt
S2TT_MODEL_URL=http://실제-모델-서버-주소:8001/s2tt
```

### 2단계: 모델 서버 API 스펙 확인

**T2TT 모델 서버가 받아야 할 형식:**
```
POST {T2TT_MODEL_URL}
Content-Type: application/json

{
  "text": "Hello",
  "source_lang": "en",
  "target_lang": "ko"
}

Response:
{
  "translated_text": "안녕"
}
```

**S2TT 모델 서버가 받아야 할 형식:**
```
POST {S2TT_MODEL_URL}
Content-Type: multipart/form-data

file: <audio_bytes>
source_lang: "en"
target_lang: "ko"

Response:
{
  "original_text": "Hello",
  "translated_text": "안녕"
}
```

> 모델 서버의 응답 형식이 다를 경우 `app/services/translate.py`의 `_call_t2tt_model()`, `_call_s2tt_model()` 함수만 수정하면 됩니다. 엔드포인트와 서비스 로직은 변경 불필요.

### 3단계: CORS 도메인 제한 (선택)

`app/main.py`에서 프론트엔드 도메인으로 제한:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-chat-app.com"],
    ...
)
```

---

## 환경변수 레퍼런스

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `APP_NAME` | `Translation API` | 앱 이름 |
| `APP_VERSION` | `0.1.0` | 앱 버전 |
| `DEBUG` | `True` | 디버그 모드 |
| `USE_MOCK` | `True` | `True`: Mock 모드 / `False`: 실제 모델 서버 사용 |
| `T2TT_MODEL_URL` | `http://localhost:8001/t2tt` | T2TT 모델 서버 URL |
| `S2TT_MODEL_URL` | `http://localhost:8001/s2tt` | S2TT 모델 서버 URL |
| `MAX_AUDIO_SIZE_MB` | `25` | 음성 파일 최대 크기 (MB) |
