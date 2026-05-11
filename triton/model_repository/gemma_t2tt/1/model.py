import json
import os
import time
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


def _clean_response_text(text: str) -> str:
    for marker in ["<turn|>", "<|turn>", "<eos>", "<bos>", "<pad>"]:
        text = text.replace(marker, "")
    return text.strip()


def _tensor_to_string(tensor) -> str:
    value = tensor.as_numpy().reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _build_prompt(text: str, source_language: str, target_language: str) -> str:
    source_hint = f"from {source_language} " if source_language else ""
    return (
        f"Translate {source_hint}to {target_language}. "
        "Return only the translation.\n"
        f"{text}"
    )


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.device_map = os.environ.get("DEVICE_MAP", "cuda:0")
        if self.device_map.startswith("cuda") and not torch.cuda.is_available():
            self.device_map = "cpu"

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
                text_tensor        = pb_utils.get_input_tensor_by_name(request, "TEXT")
                source_lang_tensor = pb_utils.get_input_tensor_by_name(request, "SOURCE_LANGUAGE")
                target_lang_tensor = pb_utils.get_input_tensor_by_name(request, "TARGET_LANGUAGE")

                if text_tensor is None or target_lang_tensor is None:
                    raise ValueError("TEXT and TARGET_LANGUAGE are required.")

                text            = _tensor_to_string(text_tensor)
                source_language = _tensor_to_string(source_lang_tensor) if source_lang_tensor is not None else ""
                target_language = _tensor_to_string(target_lang_tensor)

                messages = [
                    {
                        "role": "user",
                        "content": _build_prompt(text, source_language, target_language),
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

                responses.append(pb_utils.InferenceResponse([
                    pb_utils.Tensor("SOURCE_TEXT",       np.array([text.encode("utf-8")],             dtype=np.object_)),
                    pb_utils.Tensor("SOURCE_LANGUAGE",   np.array([source_language.encode("utf-8")],  dtype=np.object_)),
                    pb_utils.Tensor("TRANSLATED_TEXT",   np.array([cleaned_response.encode("utf-8")], dtype=np.object_)),
                    pb_utils.Tensor("TARGET_LANGUAGE",   np.array([target_language.encode("utf-8")],  dtype=np.object_)),
                    pb_utils.Tensor("RAW_RESPONSE",      np.array([response.encode("utf-8")],         dtype=np.object_)),
                    pb_utils.Tensor("INFERENCE_SECONDS", np.array([elapsed], dtype=np.float32)),
                ]))
            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(str(exc)),
                    )
                )

        return responses

    def finalize(self):
        pb_utils.Logger.log_info("Cleaning up gemma_t2tt model.")
