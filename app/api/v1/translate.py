"""
번역 API 엔드포인트 (v1)

2개의 엔드포인트:
  POST /api/v1/translate/text    → 텍스트 번역 (T2TT)
  POST /api/v1/translate/speech  → 음성 번역 (S2TT)

4개 시나리오 매핑:
┌────────────────────────┬──────────────────────────────────────────────┐
│ 시나리오               │ 프론트에서 호출하는 엔드포인트              │
├────────────────────────┼──────────────────────────────────────────────┤
│ 번역 ON  + 텍스트      │ POST /translate/text  (수신 즉시)           │
│ 번역 OFF + 텍스트      │ POST /translate/text  ([번역 보기] 클릭 시) │
│ 번역 ON  + 음성        │ POST /translate/speech (재생 버튼 클릭 시)  │
│ 번역 OFF + 음성        │ POST /translate/speech (재생 버튼 클릭 시)  │
│                        │  → 프론트에서 translated_text 숨김 처리     │
│                        │  → [번역 보기] 클릭 시 노출 (추가 호출 X)  │
└────────────────────────┴──────────────────────────────────────────────┘
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.core.config import settings
from app.schemas.translate import (
    TextTranslateRequest,
    TextTranslateResponse,
    SpeechTranslateResponse,
    ErrorResponse,
)
from app.services.translate import translate_text, translate_speech


router = APIRouter(prefix="/translate", tags=["번역"])


# ══════════════════════════════════════════════
# 1. 텍스트 번역 엔드포인트 (T2TT)
# ══════════════════════════════════════════════

@router.post(
    "/text",
    response_model=TextTranslateResponse,
    summary="텍스트 번역",
    description="""
    텍스트를 지정된 언어로 번역합니다.

    **사용 시나리오:**
    - 번역 ON + 텍스트 메시지: 상대방 메시지 수신 즉시 호출
    - 번역 OFF + 텍스트 메시지: [번역 보기] 버튼 클릭 시 호출
    """,
    responses={
        400: {"model": ErrorResponse, "description": "지원하지 않는 언어"},
        502: {"model": ErrorResponse, "description": "모델 서버 에러"},
        504: {"model": ErrorResponse, "description": "모델 서버 타임아웃"},
    },
)
async def text_translate(request: TextTranslateRequest):
    """
    T2TT 모델을 호출하여 텍스트를 번역한다.

    Flow:
      프론트엔드 → 이 엔드포인트 → services.translate_text() → 모델 서버(또는 mock)
    """
    translated = await translate_text(
        text=request.text,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
    )
    return TextTranslateResponse(translated_text=translated)


# ══════════════════════════════════════════════
# 2. 음성 번역 엔드포인트 (S2TT)
# ══════════════════════════════════════════════

@router.post(
    "/speech",
    response_model=SpeechTranslateResponse,
    summary="음성 번역",
    description="""
    음성 파일을 텍스트로 전사(STT)하고 번역합니다.

    **S2TT 내부 동작:**
    1. 음성 → 원문 텍스트 전사 (STT)
    2. 원문 텍스트 → 번역된 텍스트

    **응답에 원문(original_text)과 번역문(translated_text)이 모두 포함됩니다.**

    **사용 시나리오:**
    - 번역 ON + 음성: 재생 버튼 클릭 → 원문 + 번역문 모두 표시
    - 번역 OFF + 음성: 재생 버튼 클릭 → 원문만 표시
      → [번역 보기] 클릭 시 이미 받은 translated_text를 노출 (추가 API 호출 없음)
    """,
    responses={
        400: {"model": ErrorResponse, "description": "지원하지 않는 언어 또는 파일 형식"},
        413: {"model": ErrorResponse, "description": "파일 크기 초과"},
        502: {"model": ErrorResponse, "description": "모델 서버 에러"},
        504: {"model": ErrorResponse, "description": "모델 서버 타임아웃"},
    },
)
async def speech_translate(
    file: UploadFile = File(
        ...,
        description="음성 파일 (wav, mp3, ogg, webm, m4a)",
    ),
    source_lang: str = Form(
        ...,
        description="음성의 원본 언어 코드 (예: en)",
    ),
    target_lang: str = Form(
        ...,
        description="번역 대상 언어 코드 (예: ko)",
    ),
):
    """
    S2TT 모델을 호출하여 음성을 텍스트로 전사하고 번역한다.

    Flow:
      프론트엔드 → 이 엔드포인트 → services.translate_speech() → 모델 서버(또는 mock)
                                                                    ├→ 1단계: STT (원문)
                                                                    └→ 2단계: 번역

    파일 검증:
      1. Content-Type 체크 (허용된 오디오 형식만)
      2. 파일 크기 체크 (최대 MAX_AUDIO_SIZE_MB)
    """

    # ── 파일 형식 검증 ──
    if file.content_type not in settings.ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_audio_format",
                "message": (
                    f"지원하지 않는 오디오 형식: {file.content_type}. "
                    f"지원 형식: {settings.ALLOWED_AUDIO_TYPES}"
                ),
            },
        )

    # ── 파일 읽기 + 크기 검증 ──
    audio_bytes = await file.read()
    max_size = settings.MAX_AUDIO_SIZE_MB * 1024 * 1024  # MB → bytes

    if len(audio_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "message": f"파일 크기가 {settings.MAX_AUDIO_SIZE_MB}MB를 초과합니다",
            },
        )

    # ── S2TT 모델 호출 ──
    result = await translate_speech(
        audio_bytes=audio_bytes,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    return SpeechTranslateResponse(
        original_text=result["original_text"],
        translated_text=result["translated_text"],
    )
