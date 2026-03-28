"""
설정 모듈
- 환경변수 기반으로 모델 서버 URL 등을 관리
- .env 파일 또는 환경변수에서 값을 읽어옴
- 나중에 실제 모델 서버가 올라가면 URL만 변경하면 됨
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── 앱 기본 설정 ──
    APP_NAME: str = "Translation API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    APP_ENV: str = "production"

    # ── 모델 설정 ──
    MODEL_NAME: str = ""        # 사용 모델명 (.env에서 지정)
    DEVICE: str = "cuda"        # GPU 사용 (cpu / cuda)

    # ── 모델 서버 URL ──
    # FastAPI와 AI 모델이 같은 서버에 있으므로 localhost 사용
    T2TT_MODEL_URL: str = "http://localhost:8001/t2tt"
    S2TT_MODEL_URL: str = "http://localhost:8001/s2tt"

    # ── Mock 모드 ──
    # True: 모델 서버 없이 더미 응답 반환 (개발용)
    # False: 실제 모델 서버에 요청 (배포용)
    USE_MOCK: bool = False

    # ── 지원 언어 목록 ──
    SUPPORTED_LANGUAGES: list[str] = [
        "ko", "en", "ja", "zh", "es", "fr", "de",
    ]

    # ── 음성 파일 설정 ──
    MAX_AUDIO_SIZE_MB: int = 50  # 최대 업로드 크기 (MB)
    ALLOWED_AUDIO_TYPES: list[str] = [
        "audio/wav",
        "audio/mpeg",       # mp3
        "audio/ogg",
        "audio/webm",
        "audio/mp4",        # m4a
        "audio/x-m4a",
    ]

    # ── 타임아웃 설정 ──
    REQUEST_TIMEOUT: int = 120  # 모델 추론 타임아웃 (초)

    # ── 로깅 (Loki) ──
    LOKI_URL: str = ""          # Loki 주소 (.env에서 지정, 비어있으면 비활성화)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# 싱글톤 인스턴스
settings = Settings()
