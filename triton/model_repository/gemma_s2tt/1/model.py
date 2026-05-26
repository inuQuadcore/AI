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
        "Transcribe the audio in the original language and translate it. "
        "Return exactly this format:\n"
        "ASR: <original-language transcription>\n"
        "DETECTED_SOURCE_LANGUAGE: <detected source language>\n"
        "TRANSLATION: <translated text>\n"
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
    pb_utils.Logger.log_info(f"ffmpeg decoding: {path} (size={file_size} bytes)")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ignore_length", "1",  # Spring pipe 출력 WAV의 data chunk size=0x7FFFFFFF 대응
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
        timeout=30,
    )
    if completed.returncode != 0:
        stderr_msg = completed.stderr.decode("utf-8", errors="replace")
        pb_utils.Logger.log_error(f"ffmpeg decode failed: {stderr_msg}")
        raise RuntimeError(f"ffmpeg failed to decode audio: {stderr_msg}")

    audio = np.frombuffer(completed.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Decoded audio is empty. 빈 파일이거나 지원하지 않는 코덱입니다.")

    pb_utils.Logger.log_info(f"ffmpeg decoded {audio.size} samples ({audio.size / SAMPLE_RATE:.2f}s)")
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


def _empty_segments_json() -> str:
    return json.dumps(
        {"sample_rate": SAMPLE_RATE, "vad_seconds": 0.0, "segments": []},
        ensure_ascii=False,
    )


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        if not torch.cuda.is_available():
            self.device_map = "cpu"
        else:
            gpu_id = args.get("model_instance_device_id", "0")
            self.device_map = f"cuda:{gpu_id}"

        self.model_source = _resolve_model_source()
        pb_utils.Logger.log_info(f"Loading model from: {self.model_source}")
        pb_utils.Logger.log_info(f"Using device map: {self.device_map}")

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
            pb_utils.Logger.log_info("SileroVAD loaded successfully.")
        except Exception as exc:
            self.vad_model = None
            pb_utils.Logger.log_warning(f"SileroVAD load failed — USE_VAD requests will fall back to single inference: {exc}")

    def _run_gemma(self, audio_array: np.ndarray, target_language: str):
        """오디오 배열 하나를 Gemma에 넣어 ASR + 번역 수행. (source_text, source_language, translated_text, raw_response, elapsed) 반환."""
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
            cleaned,
            elapsed,
        )

    def _run_vad(self, audio_array: np.ndarray, target_language: str):
        """SileroVAD로 음성 구간을 분리하고 구간별 Gemma 추론 수행."""
        from silero_vad import get_speech_timestamps

        vad_start = time.perf_counter()
        speech_timestamps = get_speech_timestamps(
            torch.from_numpy(audio_array),
            self.vad_model,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
        )
        vad_seconds = time.perf_counter() - vad_start

        segments = []
        translated_parts = []
        last_source_text = ""
        last_source_language = ""
        last_raw = ""
        total_inference_seconds = 0.0

        for idx, ts in enumerate(speech_timestamps):
            start_s = ts["start"]
            end_s = ts["end"]
            segment_audio = audio_array[int(start_s * SAMPLE_RATE):int(end_s * SAMPLE_RATE)]

            src_text, src_lang, trans_text, raw, inf_sec = self._run_gemma(segment_audio, target_language)
            total_inference_seconds += inf_sec

            segments.append({
                "index": idx + 1,
                "start_seconds": round(start_s, 3),
                "end_seconds": round(end_s, 3),
                "timestamp": f"{_fmt_timestamp(start_s)} --> {_fmt_timestamp(end_s)}",
                "source_text": src_text,
                "source_language": src_lang,
                "translated_text": trans_text,
                "inference_seconds": round(inf_sec, 3),
            })
            translated_parts.append(trans_text)
            last_source_text = src_text
            last_source_language = src_lang
            last_raw = raw

        segments_json = json.dumps(
            {
                "sample_rate": SAMPLE_RATE,
                "vad_seconds": round(vad_seconds, 4),
                "segments": segments,
            },
            ensure_ascii=False,
        )
        translated_text_full = " ".join(translated_parts)

        return (
            last_source_text,
            last_source_language,
            translated_text_full,
            last_raw,
            segments_json,
            total_inference_seconds,
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                audio_tensor       = pb_utils.get_input_tensor_by_name(request, "AUDIO_BYTES")
                target_lang_tensor = pb_utils.get_input_tensor_by_name(request, "TARGET_LANGUAGE")
                use_vad_tensor     = pb_utils.get_input_tensor_by_name(request, "USE_VAD")

                if audio_tensor is None or target_lang_tensor is None:
                    raise ValueError("AUDIO_BYTES and TARGET_LANGUAGE are required.")

                audio_bytes = audio_tensor.as_numpy().reshape(-1)[0]
                if isinstance(audio_bytes, np.bytes_):
                    audio_bytes = bytes(audio_bytes)
                elif isinstance(audio_bytes, str):
                    audio_bytes = audio_bytes.encode("utf-8")

                if len(audio_bytes) == 0:
                    raise ValueError("AUDIO_BYTES가 비어 있습니다. Spring ffmpeg 변환 결과를 확인하세요.")

                target_language = _tensor_to_string(target_lang_tensor)

                use_vad = (
                    use_vad_tensor is not None
                    and _tensor_to_string(use_vad_tensor).strip().lower() == "true"
                )

                with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
                    temp_path = Path(tmp.name)
                    tmp.write(audio_bytes)

                try:
                    audio_array = _load_audio_with_ffmpeg(temp_path)
                    segments_json = _empty_segments_json()

                    if use_vad:
                        if self.vad_model is None:
                            pb_utils.Logger.log_warning("USE_VAD=true but VAD model not loaded — falling back to single inference.")
                            source_text, source_language, translated_text, raw_response, elapsed = self._run_gemma(audio_array, target_language)
                        else:
                            try:
                                source_text, source_language, translated_text, raw_response, segments_json, elapsed = self._run_vad(audio_array, target_language)
                            except Exception as vad_exc:
                                pb_utils.Logger.log_warning(f"VAD path failed — falling back to single inference: {vad_exc}")
                                source_text, source_language, translated_text, raw_response, elapsed = self._run_gemma(audio_array, target_language)
                    else:
                        source_text, source_language, translated_text, raw_response, elapsed = self._run_gemma(audio_array, target_language)

                    responses.append(pb_utils.InferenceResponse([
                        pb_utils.Tensor("SOURCE_TEXT",       np.array([source_text.encode("utf-8")],     dtype=np.object_)),
                        pb_utils.Tensor("SOURCE_LANGUAGE",   np.array([source_language.encode("utf-8")], dtype=np.object_)),
                        pb_utils.Tensor("TRANSLATED_TEXT",   np.array([translated_text.encode("utf-8")], dtype=np.object_)),
                        pb_utils.Tensor("TARGET_LANGUAGE",   np.array([target_language.encode("utf-8")], dtype=np.object_)),
                        pb_utils.Tensor("RAW_RESPONSE",      np.array([raw_response.encode("utf-8")],    dtype=np.object_)),
                        pb_utils.Tensor("INFERENCE_SECONDS", np.array([elapsed], dtype=np.float32)),
                        pb_utils.Tensor("SEGMENTS_JSON",     np.array([segments_json.encode("utf-8")],   dtype=np.object_)),
                    ]))
                finally:
                    temp_path.unlink(missing_ok=True)

            except Exception as exc:
                pb_utils.Logger.log_error(f"gemma_s2tt failed: {exc!r}")
                pb_utils.Logger.log_error(traceback.format_exc())
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(str(exc)),
                    )
                )

        return responses

    def finalize(self):
        pb_utils.Logger.log_info("Cleaning up gemma_s2tt model.")
