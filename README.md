# Everybuddy AI — Translation API

채팅 앱의 번역 기능을 담당하는 AI 서버입니다.
FastAPI 기반으로 T2TT(텍스트 번역)와 S2TT(음성 번역) 모델을 래핑하여 Spring Boot 백엔드에 번역 API를 제공합니다.

---

## 배경 및 설계 의도

### AI 모델 구성

이 서버는 두 가지 모델을 제공합니다:

| 모델 | 입력 | 출력 |
|------|------|------|
| **T2TT** (Text-to-Translated-Text) | 텍스트 원문 | 번역된 텍스트 |
| **S2TT** (Speech-to-Translated-Text) | 음성 파일 | 원문 전사 텍스트 + 번역된 텍스트 |

### S2TT 내부 동작

S2TT 모델은 내부적으로 2단계 처리를 합니다:

```
음성 입력
  └──→ [1단계: STT] 원문 텍스트 전사
         └──→ [2단계: 번역] 번역된 텍스트
```

이 서버의 S2TT 응답에는 두 단계의 결과가 모두 포함됩니다 (`original_text` + `translated_text`).
별도의 STT 모델 없이도 음성 메시지 원문을 채팅 UI에 표시할 수 있습니다.

### 번역 OFF + 음성 시나리오 설계 포인트

번역 OFF 상태에서도 음성 번역 API를 호출하여 원문과 번역문을 한 번에 받아둡니다.
S2TT 모델이 내부적으로 STT → 번역을 순차 처리하기 때문에 추가 비용이 크지 않으며,
사용자가 [번역 보기]를 클릭했을 때 추가 API 호출 없이 즉시 응답할 수 있습니다.

---

## 아키텍처

### 전체 구성도

```
┌─────────────────────────────────────────────────────────────────┐
│  사용자 (모바일 / 웹)                                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTPS
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS 서버                                                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Spring Boot 백엔드                                        │   │
│  │  - 채팅, 사용자 관리 등 메인 비즈니스 로직                   │   │
│  │  - 번역 요청을 GPU 서버로 포워딩                             │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP (내부망 프록시 경유)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  GPU 서버                                                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  FastAPI — Translation API  (이 레포)                     │   │
│  │  - 요청 유효성 검증 (언어 코드, 파일 형식/크기)              │   │
│  │  - 에러 처리 및 타임아웃 관리                               │   │
│  │  - AI 모델 호출 및 응답 래핑                                │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                          │ localhost                             │
│  ┌──────────────────────▼───────────────────────────────────┐   │
│  │  AI 모델 서버                                              │   │
│  │  - T2TT 모델: 텍스트 → 번역 텍스트                         │   │
│  │  - S2TT 모델: 음성 → 원문 전사 + 번역 텍스트               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

> FastAPI와 AI 모델 서버는 같은 GPU 서버에 함께 배포됩니다.
> 두 컨테이너 간 통신은 외부 네트워크를 거치지 않고 `localhost`로 호출합니다.

### 서버 간 연결 구성

```
AWS 서버 (Spring Boot)
    │
    │  POST http://█████████████:█████/translate/...
    ▼
프록시 서버 (Bastion)      ← ██.███.██.███:█████
    │
    │  → GPU 서버:8000 으로 포워딩
    ▼
GPU 서버 (FastAPI)         ← 내부망 접근
    │
    │  localhost:8001
    ▼
AI 모델 서버
```

### FastAPI 내부 레이어

```
HTTP 요청 (from Spring Boot)
    │
    ▼
api/v1/translate.py      — 엔드포인트, 파일 형식/크기 검증
    │
    ▼
services/translate.py    — 언어 코드 검증, 모델 호출, 에러 처리
    │
    ▼
AI 모델 서버 (localhost)
```

---

## API 엔드포인트

> Swagger UI: `/docs`

### `GET /health` — 서버 상태 확인

```json
{ "status": "ok" }
```

### `POST /translate/text` — 텍스트 번역 (T2TT)

```json
// Request
{ "text": "Hello", "source_language": "en", "target_language": "ko" }

// Response
{ "translated_text": "안녕하세요" }
```

### `POST /translate/speech` — 음성 번역 (S2TT)

```
// Request: multipart/form-data
audio, source_language, target_language

// Response
{ "original_text": "Hello", "translated_text": "안녕하세요" }
```

**에러 응답 공통 형식**
```json
{ "error": "에러코드", "message": "상세 메시지" }
```

| 코드 | 설명 |
|------|------|
| 400 | 지원하지 않는 언어 코드 / 오디오 형식 |
| 413 | 파일 크기 초과 |
| 502 | 모델 서버 오류 |
| 504 | 모델 서버 응답 시간 초과 |

---

## 시나리오별 호출 흐름

| 시나리오 | 호출 시점 | 엔드포인트 |
|----------|----------|-----------|
| 번역 ON + 텍스트 수신 | 메시지 도착 즉시 | `POST /translate/text` |
| 번역 OFF + 텍스트 수신 | [번역 보기] 클릭 시 | `POST /translate/text` |
| 번역 ON + 음성 수신 | 재생 버튼 클릭 시 | `POST /translate/speech` |
| 번역 OFF + 음성 수신 | 재생 버튼 클릭 시 | `POST /translate/speech` (번역문은 프론트에서 숨김 처리) |

---

## 프로젝트 구조

```
AI/
├── .github/workflows/deploy.yml  # CI/CD (GitHub Actions)
├── app/
│   ├── main.py                   # FastAPI 앱 진입점, CORS, Loki 로깅
│   ├── core/config.py            # 환경변수 설정
│   ├── api/v1/translate.py       # 엔드포인트 정의
│   ├── services/translate.py     # 비즈니스 로직
│   ├── schemas/translate.py      # 요청/응답 스키마
│   └── models/mock.py            # 개발용 Mock
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 로컬 실행

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

기본값은 `USE_MOCK=True`로 실제 모델 서버 없이 더미 응답을 반환합니다.

---

## 지원 언어

`ko` `en` `ja` `zh` `es` `fr` `de`
