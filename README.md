# Everybuddy AI — Triton Inference Server

채팅 앱의 번역 기능을 담당하는 AI 추론 서버입니다.
NVIDIA Triton Inference Server 기반으로 T2TT(텍스트 번역)와 S2TT(음성 번역)를 제공합니다.

---

## 아키텍처

```
Spring Boot → Triton Inference Server (KServe v2 HTTP)
```

Spring이 Triton을 직접 호출합니다. FastAPI 중간 계층은 없습니다.

```
GPU 서버
├── triton          (port 8000 — KServe v2 HTTP)
├── node-exporter   (port 9100 — 메트릭)
└── dcgm-exporter   (port 9400 — GPU 메트릭)
```

---

## 레포 구조

```
AI/
├── .env.example                         # 환경변수 템플릿
├── .github/workflows/deploy.yml         # CI/CD
├── docker-compose.yml                   # Triton + 모니터링 컨테이너 정의
└── triton/
    └── model_repository/
        ├── gemma_t2tt/
        │   ├── config.pbtxt             # 모델 I/O 계약 (Triton 설정)
        │   └── 1/model.py               # Python backend 추론 코드
        └── gemma_s2tt/
            ├── config.pbtxt
            └── 1/model.py
```

---

## GPU 서버 구성

### 서버 디렉토리 구조

```
~/hdd/
├── docker-compose.yml       ← 이 레포에서 SCP로 배포됨
├── .env                     ← 수동 작성 (gitignore, 아래 참고)
├── capston/
│   ├── triton/
│   │   └── model_repository/   ← MODEL_REPO_PATH (Triton이 마운트)
│   │       ├── gemma_t2tt/
│   │       └── gemma_s2tt/
│   └── models/                 ← MODEL_PATH (모델 파일, Python 실행환경)
│       ├── gemma-4-E4B-it/
│       └── triton_envs/
│           └── gemma_triton_py312.tar.gz
```

> `capston/triton/model_repository` 는 이 레포의 `triton/model_repository` 와 내용이 같습니다.
> 서버에 직접 반영하려면 변경된 파일을 서버 경로에 복사해야 합니다.

### .env 작성 (최초 1회, 서버에서 직접)

```bash
cat > ~/hdd/.env << 'EOF'
MODEL_REPO_PATH=/home/inu0608/hdd/capston/triton/model_repository
MODEL_PATH=/home/inu0608/hdd/capston/models
MODEL_ID=google/gemma-4-E4B-it
DEVICE_MAP=cuda:0
REQUEST_TIMEOUT=120
MAX_AUDIO_SIZE_MB=50
EOF
```

`.env`는 gitignore 처리되어 있어 배포 시 자동으로 반영되지 않습니다. 서버에 한 번만 작성하면 됩니다.

### Python 실행환경 빌드 (최초 1회, 서버에서 직접)

Triton Python backend가 사용할 conda 환경을 패킹합니다.

```bash
conda create -p ~/hdd/capston/venvs/gemma-triton-py312 python=3.12 -y
conda activate ~/hdd/capston/venvs/gemma-triton-py312
export PYTHONNOUSERSITE=True

pip install "numpy<2" torch transformers accelerate \
            librosa soundfile sentencepiece safetensors conda-pack

mkdir -p ~/hdd/capston/models/triton_envs
conda pack \
  -p ~/hdd/capston/venvs/gemma-triton-py312 \
  -o ~/hdd/capston/models/triton_envs/gemma_triton_py312.tar.gz
```

---

## 배포

`main` 브랜치에 push하면 자동 배포됩니다.

**배포 흐름:**
1. `docker-compose.yml` → 서버 `~/hdd/` 로 SCP 복사
2. `docker-compose up -d --force-recreate triton` 실행
3. `/v2/health/ready` 헬스체크 (최대 120초 대기)

**`node-exporter`, `dcgm-exporter` 는 배포 시 재시작되지 않습니다.**

---

## 모델 추가 / 교체

### 새 모델 추가

1. `triton/model_repository/` 아래 새 디렉토리 생성

```
triton/model_repository/
└── {model_name}_t2tt/
    ├── config.pbtxt   ← 기존 gemma_t2tt/config.pbtxt 복사 후 name 필드만 변경
    └── 1/model.py     ← 추론 로직 구현
```

2. 서버 `capston/triton/model_repository/` 에도 동일하게 반영

3. Triton 재시작

```bash
cd ~/hdd
docker-compose up -d --force-recreate triton
```

### 모델 교체 시 .env 수정

```bash
# 서버에서
vi ~/hdd/.env
# MODEL_ID, DEVICE_MAP 수정 후 저장
docker-compose up -d --force-recreate triton
```

---

## 헬스체크 및 테스트

### Triton 상태 확인

```bash
# 서버 전체 ready
curl http://localhost:8000/v2/health/ready

# 특정 모델 ready
curl http://localhost:8000/v2/models/gemma_t2tt/ready
curl http://localhost:8000/v2/models/gemma_s2tt/ready
```

### 텍스트 번역 테스트

```bash
curl -X POST http://localhost:8000/v2/models/gemma_t2tt/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"name": "TEXT",            "shape": [1], "datatype": "BYTES", "data": ["Hello"]},
      {"name": "SOURCE_LANGUAGE", "shape": [1], "datatype": "BYTES", "data": ["en"]},
      {"name": "TARGET_LANGUAGE", "shape": [1], "datatype": "BYTES", "data": ["ko"]}
    ]
  }'
```

### 로그 확인

```bash
# Triton 로그
docker logs triton -f

# 모델 로딩 상태만
docker logs triton 2>&1 | grep -E "READY|UNAVAILABLE|ERROR"
```

---

## 장애 대응

| 증상 | 확인 명령어 | 원인 |
|---|---|---|
| 컨테이너 미기동 | `docker-compose ps` | .env 누락 또는 볼륨 경로 오류 |
| 모델 UNAVAILABLE | `docker logs triton \| grep UNAVAILABLE` | 모델 파일 없음 또는 Python 환경 없음 |
| 추론 에러 | `docker logs triton -f` | model.py 런타임 오류 |
| GPU 미인식 | `docker exec triton nvidia-smi` | NVIDIA Container Toolkit 문제 |

### 자주 발생하는 문제

**볼륨 경로 오류 (`volume name is too short`)**
→ 서버에 `.env` 파일이 없거나 `MODEL_REPO_PATH`, `MODEL_PATH` 값이 비어있음

```bash
cat ~/hdd/.env   # 값 확인
```

**모델 UNAVAILABLE**
→ `gemma_triton_py312.tar.gz` 가 없거나 경로가 다름

```bash
ls ~/hdd/capston/models/triton_envs/
```

**컨테이너 재시작**

```bash
cd ~/hdd
docker-compose restart triton
```
