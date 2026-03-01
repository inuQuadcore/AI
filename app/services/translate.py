"""
번역 서비스 모듈 (비즈니스 로직)

API 엔드포인트와 모델 사이의 중간 계층.
역할:
  1. USE_MOCK 설정에 따라 mock / 실제 모델 서버 분기
  2. 모델 서버 호출 시 에러 처리, 타임아웃 관리
  3. 입력 검증 (지원 언어 체크 등)

나중에 실제 모델 서버 연동 시:
  - _call_t2tt_model(), _call_s2tt_model() 함수만 수정하면 됨
  - 엔드포인트 코드는 변경 불필요
"""

import httpx
from fastapi import HTTPException

from app.core.config import settings
from app.models.mock import mock_t2tt, mock_s2tt


# ══════════════════════════════════════════════
# 헬퍼: 언어 코드 검증
# ══════════════════════════════════════════════

def _validate_languages(source_lang: str, target_lang: str) -> None:
    """
    지원 언어 목록에 있는지 검증.
    지원하지 않는 언어 코드가 오면 400 에러.
    """
    if source_lang not in settings.SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_language",
                "message": f"지원하지 않는 소스 언어: {source_lang}. "
                           f"지원 언어: {settings.SUPPORTED_LANGUAGES}",
            },
        )
    if target_lang not in settings.SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_language",
                "message": f"지원하지 않는 타겟 언어: {target_lang}. "
                           f"지원 언어: {settings.SUPPORTED_LANGUAGES}",
            },
        )


# ══════════════════════════════════════════════
# 1. T2TT (텍스트 → 번역 텍스트)
# ══════════════════════════════════════════════

async def _call_t2tt_model(
    text: str,
    source_lang: str,
    target_lang: str,
) -> str:
    """
    실제 T2TT 모델 서버 호출

    모델 서버가 올라가면 이 함수 내부만 수정.
    예상 모델 서버 API:
      POST {T2TT_MODEL_URL}
      Body: { "text": "...", "source_lang": "...", "target_lang": "..." }
      Response: { "translated_text": "..." }
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            settings.T2TT_MODEL_URL,
            json={
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["translated_text"]


async def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
) -> str:
    """
    텍스트 번역 서비스 함수

    호출 흐름:
      엔드포인트 → translate_text() → mock 또는 실제 모델 서버

    Args:
        text: 번역할 원문 텍스트
        source_lang: 원문 언어 코드
        target_lang: 번역 대상 언어 코드

    Returns:
        번역된 텍스트

    Raises:
        HTTPException 400: 지원하지 않는 언어
        HTTPException 502: 모델 서버 에러
        HTTPException 504: 모델 서버 타임아웃
    """
    # 1) 언어 코드 검증
    _validate_languages(source_lang, target_lang)

    # 2) 같은 언어면 원문 그대로 반환 (불필요한 모델 호출 방지)
    if source_lang == target_lang:
        return text

    # 3) Mock / 실제 모델 분기
    if settings.USE_MOCK:
        return await mock_t2tt(text, source_lang, target_lang)

    # 4) 실제 모델 서버 호출 + 에러 처리
    try:
        return await _call_t2tt_model(text, source_lang, target_lang)
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail={
                "error": "model_timeout",
                "message": "T2TT 모델 서버 응답 시간 초과",
            },
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "model_error",
                "message": f"T2TT 모델 서버 에러: {e.response.status_code}",
            },
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "model_unavailable",
                "message": "T2TT 모델 서버에 연결할 수 없습니다",
            },
        )


# ══════════════════════════════════════════════
# 2. S2TT (음성 → 원문 텍스트 + 번역 텍스트)
# ══════════════════════════════════════════════

async def _call_s2tt_model(
    audio_bytes: bytes,
    source_lang: str,
    target_lang: str,
) -> dict[str, str]:
    """
    실제 S2TT 모델 서버 호출

    모델 서버가 올라가면 이 함수 내부만 수정.
    예상 모델 서버 API:
      POST {S2TT_MODEL_URL}
      Body: multipart/form-data (file, source_lang, target_lang)
      Response: { "original_text": "...", "translated_text": "..." }
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            settings.S2TT_MODEL_URL,
            files={"file": ("audio.wav", audio_bytes)},
            data={
                "source_lang": source_lang,
                "target_lang": target_lang,
            },
        )
        response.raise_for_status()
        return response.json()


async def translate_speech(
    audio_bytes: bytes,
    source_lang: str,
    target_lang: str,
) -> dict[str, str]:
    """
    음성 번역 서비스 함수

    S2TT 모델 내부에서 2단계 처리:
      1단계: 음성 → 원문 텍스트 전사 (STT)
      2단계: 원문 텍스트 → 번역된 텍스트

    호출 흐름:
      엔드포인트 → translate_speech() → mock 또는 실제 모델 서버

    Args:
        audio_bytes: 음성 파일 바이너리
        source_lang: 음성의 원본 언어 코드
        target_lang: 번역 대상 언어 코드

    Returns:
        {
            "original_text": "STT 전사 결과",
            "translated_text": "번역 결과"
        }

    Raises:
        HTTPException 400: 지원하지 않는 언어
        HTTPException 502: 모델 서버 에러
        HTTPException 504: 모델 서버 타임아웃
    """
    # 1) 언어 코드 검증
    _validate_languages(source_lang, target_lang)

    # 2) Mock / 실제 모델 분기
    if settings.USE_MOCK:
        return await mock_s2tt(audio_bytes, source_lang, target_lang)

    # 3) 실제 모델 서버 호출 + 에러 처리
    try:
        return await _call_s2tt_model(audio_bytes, source_lang, target_lang)
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail={
                "error": "model_timeout",
                "message": "S2TT 모델 서버 응답 시간 초과 (음성 처리는 시간이 더 걸립니다)",
            },
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "model_error",
                "message": f"S2TT 모델 서버 에러: {e.response.status_code}",
            },
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "model_unavailable",
                "message": "S2TT 모델 서버에 연결할 수 없습니다",
            },
        )
