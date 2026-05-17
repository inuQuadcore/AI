# Everybuddy AI — Triton Inference Server

채팅 앱의 번역 기능을 담당하는 AI 추론 서버입니다.  
NVIDIA Triton Inference Server 위에서 **T2TT(텍스트 번역)** 와 **S2TT(음성 번역)** 를 제공합니다.

---

## 아키텍처

```
Spring Boot  →  Triton Inference Server (KServe v2 HTTP / Binary)
```

Spring이 Triton을 직접 호출합니다. FastAPI 중간 계층 없음.

```
GPU 서버 (laurel)
├── triton         (port 8000 HTTP · 8001 gRPC · 8002 metrics)
├── node-exporter  (port 9100 — 서버 메트릭)
└── dcgm-exporter  (port 9400 — GPU 메트릭)
```

---

## 레포 구조

```
AI/
├── .env.example                          # 환경변수 템플릿 (민감정보 제외)
├── .github/workflows/deploy.yml          # CI/CD (GitHub Actions)
├── docker-compose.yml                    # Triton + 모니터링 컨테이너 정의
└── triton/
    ├── Dockerfile                        # Triton 커스텀 이미지
    └── model_repository/
        ├── gemma_t2tt/
        │   ├── config.pbtxt              # 모델 I/O 스펙 (Triton 설정)
        │   └── 1/model.py               # 텍스트 번역 추론 로직
        └── gemma_s2tt/
            ├── config.pbtxt
            └── 1/model.py               # 음성 번역 추론 로직
```

---

## 모델

| 모델 | 입력 | 출력 | 할당 GPU |
|---|---|---|---|
| `gemma_s2tt` | 오디오 파일 bytes, 목표 언어 | ASR 결과, 감지 언어, 번역 결과 | GPU 0, 1 |
| `gemma_t2tt` | 텍스트, 원본 언어, 목표 언어 | 번역 결과 | GPU 2, 3 |

기반 모델: `google/gemma-4-E4B-it` (멀티모달 · 4B 파라미터)

### GPU 구성

```
Tesla V100-DGXS-32GB × 4
Driver 535.x / CUDA 12.2

GPU 0 │ gemma_s2tt instance 0  ├─ ~16GB / 32GB (~50%)
GPU 1 │ gemma_s2tt instance 1  ├─ ~16GB / 32GB (~50%)
GPU 2 │ gemma_t2tt instance 0  ├─ ~16GB / 32GB (~50%)
GPU 3 │ gemma_t2tt instance 1  └─ ~16GB / 32GB (~50%)
```

---

## Triton 인터페이스 스펙

### T2TT (텍스트 번역)

**엔드포인트:** `POST /v2/models/gemma_t2tt/infer`

| 이름 | 타입 | 설명 |
|---|---|---|
| `TEXT` | BYTES | 번역할 텍스트 |
| `SOURCE_LANGUAGE` | BYTES | 원본 언어 (예: `"korean"`) |
| `TARGET_LANGUAGE` | BYTES | 목표 언어 (예: `"english"`) |

**출력:** `SOURCE_TEXT` · `SOURCE_LANGUAGE` · `TRANSLATED_TEXT` · `TARGET_LANGUAGE` · `RAW_RESPONSE` · `INFERENCE_SECONDS`

### S2TT (음성 번역)

**엔드포인트:** `POST /v2/models/gemma_s2tt/infer`

| 이름 | 타입 | 설명 |
|---|---|---|
| `AUDIO_BYTES` | BYTES | 오디오 파일 원본 바이트 (m4a, wav, webm 등) |
| `TARGET_LANGUAGE` | BYTES | 목표 언어 (예: `"korean"`) |

**출력:** `SOURCE_TEXT` · `SOURCE_LANGUAGE` · `TRANSLATED_TEXT` · `TARGET_LANGUAGE` · `RAW_RESPONSE` · `INFERENCE_SECONDS`

> ⚠️ **백엔드 연동 주의**  
> `AUDIO_BYTES`는 반드시 **raw bytes** 를 Triton binary HTTP extension으로 전송해야 합니다.  
> JSON 프로토콜은 BYTES를 base64로 인코딩하므로, ffmpeg가 오디오로 인식하지 못해 오류가 발생합니다.

---

## 환경 설정

### .env 파일 (서버에서 최초 1회 작성, git 미포함)

```bash
# ~/hdd/.env

# Docker Hub에 올라가 있는 커스텀 Triton 이미지
TRITON_IMAGE={DOCKERHUB_USERNAME}/tritonserver-gemma:latest

# 서버 내 볼륨 마운트 경로
MODEL_REPO_PATH={HOME}/hdd/capston/triton/model_repository
MODEL_PATH={HOME}/hdd/capston/models

# 모델 설정
MODEL_ID=google/gemma-4-E4B-it
```

`.env.example` 파일을 참고해서 작성하세요.

### GitHub Secrets (CI/CD에서 사용)

| Secret | 설명 |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub 계정명 |
| `DOCKERHUB_TOKEN` | Docker Hub 액세스 토큰 |
| `SERVER_HOST` | GPU 서버 내부 IP |
| `SERVER_PORT` | SSH 포트 |
| `SERVER_USER` | 서버 계정명 |
| `BASTION_HOST` | Bastion 서버 IP |
| `BASTION_PORT` | Bastion SSH 포트 |
| `BASTION_USER` | Bastion 계정명 |
| `SSH_PRIVATE_KEY` | SSH 개인키 |

---

## Dockerfile

```
베이스 이미지: nvcr.io/nvidia/tritonserver:24.12-py3
PyTorch:       2.6.0+cu118  ← laurel 서버 드라이버 535.x 검증값
               cu118 wheel을 사용하면 드라이버 호환성 범위가 넓어집니다
오디오:        ffmpeg (apt), soundfile, librosa
모델 로딩:     transformers 5.8.1, accelerate 1.13.0, sentencepiece, safetensors
```

Dockerfile 변경 시 main push → GitHub Actions가 자동으로 이미지를 빌드해 Docker Hub에 push합니다.

---

## 배포

`main` 브랜치에 push하면 자동 배포됩니다.

```
push to main
  │
  ├─ [Dockerfile 변경 시] Docker 이미지 빌드 → Docker Hub push
  │
  └─ GPU 서버 배포 (Bastion 경유 SSH)
       ├── docker-compose.yml → 서버 ~/hdd/ 복사
       ├── triton/model_repository/ → 서버 복사
       ├── docker pull (최신 이미지)
       └── docker-compose up -d --force-recreate triton
```

**모니터링 컨테이너(`node-exporter`, `dcgm-exporter`)는 배포 시 재시작되지 않습니다.**

---

## 서버 디렉토리 구조

```
~/hdd/
├── docker-compose.yml       ← CI/CD가 자동 복사
├── .env                     ← 수동 작성 (git 미포함)
└── capston/
    ├── triton/
    │   └── model_repository/   ← MODEL_REPO_PATH (CI/CD가 자동 복사)
    │       ├── gemma_t2tt/
    │       └── gemma_s2tt/
    └── models/
        └── gemma-4-E4B-it/     ← MODEL_PATH (HuggingFace에서 별도 다운로드)
```

> 모델 가중치 파일(`gemma-4-E4B-it/`)은 git 미포함입니다.  
> 서버에 없다면 HuggingFace에서 직접 다운로드해야 합니다.

---

## 헬스체크 & 테스트

### Triton 상태 확인

```bash
# 서버 전체 ready
curl http://localhost:8000/v2/health/ready

# 모델별 ready
curl http://localhost:8000/v2/models/gemma_t2tt/ready
curl http://localhost:8000/v2/models/gemma_s2tt/ready

# 로드된 모델 목록
curl http://localhost:8000/v2/repository/index
```

### T2TT 동작 테스트

```bash
curl -X POST http://localhost:8000/v2/models/gemma_t2tt/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"name": "TEXT",            "shape": [1], "datatype": "BYTES", "data": ["안녕하세요"]},
      {"name": "SOURCE_LANGUAGE", "shape": [1], "datatype": "BYTES", "data": ["korean"]},
      {"name": "TARGET_LANGUAGE", "shape": [1], "datatype": "BYTES", "data": ["english"]}
    ]
  }'
```

### S2TT 동작 테스트 (Python 클라이언트)

```bash
# tritonclient 설치 (최초 1회)
pip install "tritonclient[http]" numpy

# 테스트 실행
python triton/client/infer_s2tt.py \
  --url localhost:8000 \
  --audio ./test.m4a \
  --target-lang korean
```

### 로그 확인

```bash
# 실시간 로그
docker logs triton -f

# 모델 로딩 상태
docker logs triton 2>&1 | grep -E "READY|UNAVAILABLE|ERROR"

# GPU 상태
nvidia-smi
watch -n 1 nvidia-smi
```

---

## 장애 대응

| 증상 | 확인 명령어 | 가능한 원인 |
|---|---|---|
| 컨테이너 미기동 | `docker-compose ps` | `.env` 누락, 볼륨 경로 오류 |
| 모델 UNAVAILABLE | `docker logs triton \| grep UNAVAILABLE` | 모델 파일 없음 |
| config.pbtxt 파싱 오류 | `docker logs triton \| grep "failed to read"` | protobuf 문법 오류 |
| S2TT ffmpeg 오류 | `docker logs triton \| grep ffmpeg` | 오디오 bytes 전송 방식 오류 (raw bytes 필요) |
| GPU 미인식 | `docker exec triton nvidia-smi` | NVIDIA Container Toolkit 문제 |
| 드라이버 경고 | 로그 상단 Driver Release 경고 | 무시 가능 (cu118 wheel로 호환성 확보됨) |

### 자주 발생하는 상황

**컨테이너 재시작**
```bash
cd ~/hdd
docker-compose restart triton
# 또는 강제 재생성
docker-compose up -d --force-recreate triton
```

**볼륨 경로 오류**
```bash
cat ~/hdd/.env   # MODEL_REPO_PATH, MODEL_PATH 값 확인
```

**모델 파일 없음**
```bash
ls {MODEL_PATH}/gemma-4-E4B-it/   # 모델 가중치 존재 여부 확인
```
