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
    """ffmpeg를 직접 호출해 입력 파일을 16kHz mono float32 PCM으로 변환.
    librosa/audioread 의존 없이 m4a, webm, mp3, wav 등 모든 포맷 처리 가능.
    """
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(path),
        "-ac", "1",       # mono
        "-ar", "16000",   # 16kHz
        "-f", "f32le",    # float32 little-endian PCM
        "pipe:1",         # stdout으로 출력
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

    return audio


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
        pb_utils.Logger.log_info(f"Loading model from: {self.model_source}")
        pb_utils.Logger.log_info(f"Using device map: {self.device_map}")

        self.processor = AutoProcessor.from_pretrained(self.model_source)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_source,
            dtype="auto",
            device_map=self.device_map,
        )
        self.model.eval()

    def execute(self, requests):
        responses = []

        for request in requests:
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

                target_language = _tensor_to_string(target_lang_tensor)

                with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
                    temp_path = Path(tmp.name)
                    tmp.write(audio_bytes)

                try:
                    audio_array = _load_audio_with_ffmpeg(temp_path)

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

                    response = self.processor.decode(
                        outputs[0][input_len:], skip_special_tokens=False
                    )
                    cleaned_response = _clean_response_text(response)
                    fields = _parse_tagged_output(cleaned_response)

                    source_text     = fields.get("ASR", "")
                    source_language = fields.get("DETECTED_SOURCE_LANGUAGE", "")
                    translated_text = fields.get("TRANSLATION", cleaned_response)

                    responses.append(pb_utils.InferenceResponse([
                        pb_utils.Tensor("SOURCE_TEXT",       np.array([source_text.encode("utf-8")],     dtype=np.object_)),
                        pb_utils.Tensor("SOURCE_LANGUAGE",   np.array([source_language.encode("utf-8")], dtype=np.object_)),
                        pb_utils.Tensor("TRANSLATED_TEXT",   np.array([translated_text.encode("utf-8")], dtype=np.object_)),
                        pb_utils.Tensor("TARGET_LANGUAGE",   np.array([target_language.encode("utf-8")], dtype=np.object_)),
                        pb_utils.Tensor("RAW_RESPONSE",      np.array([cleaned_response.encode("utf-8")],dtype=np.object_)),
                        pb_utils.Tensor("INFERENCE_SECONDS", np.array([elapsed], dtype=np.float32)),
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
