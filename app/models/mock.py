"""
Mock 모델 모듈 (개발/테스트용)

실제 AI 모델 서버가 없을 때 더미 응답을 반환한다.
- config.py의 USE_MOCK = True일 때 활성화
- 실제 모델 서버가 올라가면 USE_MOCK = False로 변경

Mock 동작:
- T2TT: 원문에 "[번역됨]" 접두사 + 언어 정보 붙여서 반환
- S2TT: 더미 전사 텍스트 + 더미 번역 텍스트 반환
"""

import asyncio


# ── 더미 번역 데이터 ──
# 테스트 시 좀 더 현실적인 응답을 위한 샘플 매핑
MOCK_TRANSLATIONS: dict[str, dict[str, str]] = {
    "Hello": {"ko": "안녕하세요", "ja": "こんにちは", "zh": "你好"},
    "How are you?": {"ko": "어떻게 지내세요?", "ja": "お元気ですか？", "zh": "你好吗？"},
    "Thank you": {"ko": "감사합니다", "ja": "ありがとう", "zh": "谢谢"},
}


async def mock_t2tt(text: str, source_lang: str, target_lang: str) -> str:
    """
    T2TT Mock 함수

    실제 모델 서버의 T2TT API를 흉내낸다.
    네트워크 지연을 시뮬레이션하기 위해 0.3초 대기.

    Args:
        text: 번역할 원문 텍스트
        source_lang: 원문 언어 코드
        target_lang: 번역 대상 언어 코드

    Returns:
        번역된 텍스트 (mock)
    """
    # 네트워크 지연 시뮬레이션
    await asyncio.sleep(0.3)

    # 매핑에 있으면 해당 번역 반환, 없으면 기본 포맷
    if text in MOCK_TRANSLATIONS and target_lang in MOCK_TRANSLATIONS[text]:
        return MOCK_TRANSLATIONS[text][target_lang]

    return f"[{target_lang}] {text}"


async def mock_s2tt(
    audio_bytes: bytes,
    source_lang: str,
    target_lang: str,
) -> dict[str, str]:
    """
    S2TT Mock 함수

    실제 모델 서버의 S2TT API를 흉내낸다.
    S2TT 내부 2단계 과정을 시뮬레이션:
      1단계: 음성 → 원문 텍스트 전사 (STT)
      2단계: 원문 텍스트 → 번역된 텍스트

    네트워크 + 모델 추론 지연을 시뮬레이션하기 위해 1초 대기.

    Args:
        audio_bytes: 음성 파일 바이너리 데이터
        source_lang: 음성의 원본 언어 코드
        target_lang: 번역 대상 언어 코드

    Returns:
        {
            "original_text": "STT 전사 결과 (원문)",
            "translated_text": "번역된 텍스트"
        }
    """
    # 모델 추론 지연 시뮬레이션 (STT + 번역이라 더 오래 걸림)
    await asyncio.sleep(1.0)

    # 오디오 크기 기반으로 더미 텍스트 생성 (테스트 시 구분용)
    audio_size_kb = len(audio_bytes) / 1024
    original_text = (
        f"[Mock STT] Audio received: {audio_size_kb:.1f}KB, "
        f"lang={source_lang}"
    )
    translated_text = f"[Mock Translation → {target_lang}] {original_text}"

    return {
        "original_text": original_text,
        "translated_text": translated_text,
    }
