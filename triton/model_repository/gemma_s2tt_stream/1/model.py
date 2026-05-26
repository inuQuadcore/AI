import json
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_MODEL_DIR = Path("/models/gemma-4-E4B-it")
SAMPLE_RATE = 16000


def _resolve_model_source() -> str:
    model_path = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_DIR))
    local_path = Path(model_path)
    if local_path.exists():
        return str(local_path)
    return os.environ.get("MODEL_ID", MODEL_ID)


def _build_prompt(target_language: str) -> str:
    return (
        f"Transcribe the audio in the original language, then translate its MEANING into {target_language}. "
        f"The TRANSLATION field MUST be a natural, fluent {target_language} translation of the meaning. "
        f"Do NOT transliterate or write foreign sounds in {target_language} characters. "
        f"Translate what the words mean, not how they sound.\n"
        "Return exactly this format:\n"
        "ASR: <original-language transcription>\n"
        "DETECTED_SOURCE_LANGUAGE: <detected source language>\n"
        f"TRANSLATION: <natural {target_language} translation of the meaning>\n"
        f"Target language: {target_language}"
    )


def _parse_tagged_output(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def _clean_response_text(text: str) -> str:
    for marker in ["<turn|>", "<|turn>", "<eos>", "<bos>", "<pad>"]:
        text = text.replace(marker, "")
    return text.strip()


def _load_audio_with_ffmpeg(path: Path) -> np.ndarray:
    file_size = path.stat().st_size
    pb_utils.Logger.log_info(f"[stream] ffmpeg decoding: {path} (size={file_size} bytes)")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(path),
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "f32le",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        stderr_msg = completed.stderr.decode("utf-8", errors="replace")
        pb_utils.Logger.log_error(f"[stream] ffmpeg decode failed: {stderr_msg}")
        raise RuntimeError(f"ffmpeg failed to decode audio: {stderr_msg}")

    audio = np.frombuffer(completed.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Decoded audio is empty. 빈 파일이거나 지원하지 않는 코덱입니다.")

    pb_utils.Logger.log_info(f"[stream] ffmpeg decoded {audio.size} samples ({audio.size / SAMPLE_RATE:.2f}s)")
    return audio


def _fmt_timestamp(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _tensor_to_string(tensor) -> str:
    value = tensor.as_numpy().reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        if not torch.cuda.is_available():
            self.device_map = "cpu"
        else:
            gpu_id = args.get("model_instance_device_id", "0")
            self.device_map = f"cuda:{gpu_id}"

        self.model_source = _resolve_model_source()
        pb_utils.Logger.log_info(f"[stream] Loading model from: {self.model_source}")
        pb_utils.Logger.log_info(f"[stream] Using device map: {self.device_map}")

        self.processor = AutoProcessor.from_pretrained(self.model_source)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_source,
            dtype="auto",
            device_map=self.device_map,
        )
        self.model.eval()

        try:
            from silero_vad import load_silero_vad
            self.vad_model = load_silero_vad()
            pb_utils.Logger.log_info("[stream] SileroVAD loaded successfully.")
        except Exception as exc:
            self.vad_model = None
            pb_utils.Logger.log_warning(f"[stream] SileroVAD load failed: {exc}")

    def _run_gemma(self, audio_array: np.ndarray, target_language: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text",  "text": _build_prompt(target_language)},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)
        started_at = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)
        elapsed = time.perf_counter() - started_at

        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        cleaned = _clean_response_text(response)
        fields = _parse_tagged_output(cleaned)

        return (
            fields.get("ASR", ""),
            fields.get("DETECTED_SOURCE_LANGUAGE", ""),
            fields.get("TRANSLATION", cleaned),
            elapsed,
        )

    def execute(self, requests):
        for request in requests:
            response_sender = request.get_response_sender()

            try:
                audio_tensor       = pb_utils.get_input_tensor_by_name(request, "AUDIO_BYTES")
                target_lang_tensor = pb_utils.get_input_tensor_by_name(request, "TARGET_LANGUAGE")

                if audio_tensor is None or target_lang_tensor is None:
                    raise ValueError("AUDIO_BYTES and TARGET_LANGUAGE are required.")

                audio_bytes = audio_tensor.as_numpy().reshape(-1)[0]
                if isinstance(audio_bytes, np.bytes_):
                    audio_bytes = bytes(audio_bytes)
                elif isinstance(audio_bytes, str):
                    audio_bytes = audio_bytes.encode("utf-8")

                if len(audio_bytes) == 0:
                    raise ValueError("AUDIO_BYTES가 비어 있습니다.")

                target_language = _tensor_to_string(target_lang_tensor)

                with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
                    temp_path = Path(tmp.name)
                    tmp.write(audio_bytes)

                try:
                    audio_array = _load_audio_with_ffmpeg(temp_path)
                finally:
                    temp_path.unlink(missing_ok=True)

                # VAD로 음성 구간 감지
                if self.vad_model is None:
                    raise RuntimeError("VAD 모델이 로드되지 않았습니다.")

                from silero_vad import get_speech_timestamps
                speech_timestamps = get_speech_timestamps(
                    torch.from_numpy(audio_array),
                    self.vad_model,
                    sampling_rate=SAMPLE_RATE,
                    return_seconds=True,
                )

                total_segments = len(speech_timestamps)
                pb_utils.Logger.log_info(f"[stream] VAD detected {total_segments} segments")

                if total_segments == 0:
                    # 음성 구간 없음 → 빈 세그먼트로 완료
                    empty_segment = json.dumps({
                        "index": 0,
                        "total_segments": 0,
                        "start_seconds": 0.0,
                        "end_seconds": 0.0,
                        "timestamp": "",
                        "source_text": "",
                        "source_language": "",
                        "translated_text": "",
                        "inference_seconds": 0.0,
                        "is_final": True,
                    }, ensure_ascii=False)
                    response_sender.send(
                        pb_utils.InferenceResponse([
                            pb_utils.Tensor("SEGMENT_JSON",
                                np.array([empty_segment.encode("utf-8")], dtype=np.object_)),
                        ]),
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                    )
                    continue

                # 구간마다 추론 후 즉시 전송
                for idx, ts in enumerate(speech_timestamps):
                    start_s = ts["start"]
                    end_s   = ts["end"]
                    is_last = (idx == total_segments - 1)

                    segment_audio = audio_array[int(start_s * SAMPLE_RATE):int(end_s * SAMPLE_RATE)]
                    src_text, src_lang, trans_text, elapsed = self._run_gemma(segment_audio, target_language)

                    segment = json.dumps({
                        "index": idx + 1,
                        "total_segments": total_segments,
                        "start_seconds": round(start_s, 3),
                        "end_seconds": round(end_s, 3),
                        "timestamp": f"{_fmt_timestamp(start_s)} --> {_fmt_timestamp(end_s)}",
                        "source_text": src_text,
                        "source_language": src_lang,
                        "translated_text": trans_text,
                        "inference_seconds": round(elapsed, 3),
                        "is_final": is_last,
                    }, ensure_ascii=False)

                    pb_utils.Logger.log_info(
                        f"[stream] Sending segment {idx+1}/{total_segments} "
                        f"({start_s:.1f}s~{end_s:.1f}s)"
                    )

                    flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL if is_last else 0
                    response_sender.send(
                        pb_utils.InferenceResponse([
                            pb_utils.Tensor("SEGMENT_JSON",
                                np.array([segment.encode("utf-8")], dtype=np.object_)),
                        ]),
                        flags=flags,
                    )

            except Exception as exc:
                pb_utils.Logger.log_error(f"[stream] gemma_s2tt_stream failed: {exc!r}")
                pb_utils.Logger.log_error(traceback.format_exc())

                error_segment = json.dumps({
                    "error": str(exc),
                    "is_final": True,
                }, ensure_ascii=False)
                response_sender.send(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("SEGMENT_JSON",
                                np.array([error_segment.encode("utf-8")], dtype=np.object_)),
                        ],
                        error=pb_utils.TritonError(str(exc)),
                    ),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )

    def finalize(self):
        pb_utils.Logger.log_info("[stream] Cleaning up gemma_s2tt_stream model.")
