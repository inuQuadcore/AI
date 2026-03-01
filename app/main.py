"""
FastAPI 앱 진입점

실행 방법:
  uvicorn app.main:app --reload --port 8000

API 문서:
  Swagger UI: http://localhost:8000/docs
  ReDoc:      http://localhost:8000/redoc
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1.translate import router as translate_router


# ── FastAPI 앱 생성 ──
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## 번역 API 서버

채팅 앱에서 사용하는 번역 API.
AI 모델(S2TT, T2TT)을 래핑하여 텍스트/음성 번역 기능을 제공합니다.

### 엔드포인트
- **POST /api/v1/translate/text** — 텍스트 번역 (T2TT)
- **POST /api/v1/translate/speech** — 음성 번역 (S2TT)

### 시나리오별 사용법
| 시나리오 | 엔드포인트 | 비고 |
|----------|-----------|------|
| 번역 ON + 텍스트 | `/translate/text` | 수신 즉시 호출 |
| 번역 OFF + 텍스트 | `/translate/text` | [번역 보기] 클릭 시 |
| 번역 ON + 음성 | `/translate/speech` | 원문+번역 동시 반환 |
| 번역 OFF + 음성 | `/translate/speech` | 프론트에서 번역문 숨김 처리 |
    """,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── CORS 설정 ──
# 프론트엔드(채팅 앱)에서 API 호출할 수 있도록 허용
# 배포 시에는 allow_origins를 실제 프론트 도메인으로 제한할 것
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: 배포 시 실제 도메인으로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 라우터 등록 ──
app.include_router(translate_router, prefix="/api/v1")


# ── 헬스체크 ──
@app.get(
    "/health",
    tags=["시스템"],
    summary="서버 상태 확인",
)
async def health_check():
    """
    서버가 정상 동작 중인지 확인하는 엔드포인트.
    로드밸런서, 모니터링 도구에서 사용.
    """
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "mock_mode": settings.USE_MOCK,
    }
