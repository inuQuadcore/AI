"""
요청/응답 스키마 정의
- Pydantic 모델로 API의 입출력 형태를 명확하게 정의
- FastAPI가 자동으로 검증 + Swagger 문서 생성에 활용
"""

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════
# 1. 텍스트 번역 (T2TT)
# ══════════════════════════════════════════════

class TextTranslateRequest(BaseModel):
    """
    텍스트 번역 요청 스키마

    사용 시나리오:
    - 번역 ON + 텍스트 메시지: 상대방 메시지 수신 즉시 프론트에서 호출
    - 번역 OFF + 텍스트 메시지: [번역 보기] 클릭 시 프론트에서 호출
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="번역할 원문 텍스트",
        examples=["Hello, how are you?"],
    )
    source_lang: str = Field(
        ...,
        min_length=2,
        max_length=5,
        description="원문 언어 코드 (ISO 639-1)",
        examples=["en"],
    )
    target_lang: str = Field(
        ...,
        min_length=2,
        max_length=5,
        description="번역 대상 언어 코드 (ISO 639-1)",
        examples=["ko"],
    )


class TextTranslateResponse(BaseModel):
    """
    텍스트 번역 응답 스키마

    프론트엔드 동작:
    - 번역 ON: translated_text를 원문 아래에 바로 표시
    - 번역 OFF: [번역 보기] 클릭 시 translated_text 표시
    """
    translated_text: str = Field(
        ...,
        description="번역된 텍스트",
        examples=["안녕, 잘 지내?"],
    )


# ══════════════════════════════════════════════
# 2. 음성 번역 (S2TT)
# ══════════════════════════════════════════════

# 음성 번역은 multipart/form-data로 받으므로
# Request 스키마는 FastAPI의 Form + UploadFile로 엔드포인트에서 직접 정의

class SpeechTranslateResponse(BaseModel):
    """
    음성 번역 응답 스키마

    S2TT 모델 내부 동작:
    1단계: 음성 → 원문 텍스트 전사 (STT)
    2단계: 원문 텍스트 → 번역된 텍스트

    프론트엔드 동작:
    - 번역 ON: original_text + translated_text 모두 표시
    - 번역 OFF: original_text만 표시, [번역 보기] 클릭 시 translated_text 노출
      (방법 B: 서버에서 둘 다 받아두고 프론트에서 숨김 처리)
    """
    original_text: str = Field(
        ...,
        description="STT로 전사된 원문 텍스트 (S2TT 1단계 결과)",
        examples=["Hello, how are you?"],
    )
    translated_text: str = Field(
        ...,
        description="번역된 텍스트 (S2TT 2단계 결과)",
        examples=["안녕, 잘 지내?"],
    )


# ══════════════════════════════════════════════
# 3. 공통 에러 응답
# ══════════════════════════════════════════════

class ErrorResponse(BaseModel):
    """API 에러 응답 스키마"""
    error: str = Field(..., description="에러 코드")
    message: str = Field(..., description="에러 상세 메시지")
