import io
import json
import time
import traceback
import wave

import numpy as np
import triton_python_backend_utils as pb_utils

SAMPLE_RATE = 44100
DEFAULT_LANG = "ko"
DEFAULT_VOICE = "M1"


def _tensor_to_string(tensor) -> str:
    value = tensor.as_numpy().reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        pb_utils.Logger.log_info("Loading Supertonic TTS model...")
        from supertonic import TTS
        self.tts = TTS(auto_download=True)
        self._voice_cache: dict = {}
        pb_utils.Logger.log_info("Supertonic TTS model loaded.")

    def _get_voice_style(self, voice_name: str):
        key = voice_name.upper()
        if key not in self._voice_cache:
            self._voice_cache[key] = self.tts.get_voice_style(voice_name=key)
        return self._voice_cache[key]

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT_INPUT")
                if text_tensor is None:
                    raise ValueError("TEXT_INPUT is required.")
                text = _tensor_to_string(text_tensor)

                lang_tensor = pb_utils.get_input_tensor_by_name(request, "LANGUAGE")
                lang = _tensor_to_string(lang_tensor) if lang_tensor is not None else DEFAULT_LANG

                voice_tensor = pb_utils.get_input_tensor_by_name(request, "VOICE")
                voice_name = _tensor_to_string(voice_tensor) if voice_tensor is not None else DEFAULT_VOICE

                style = self._get_voice_style(voice_name)

                started_at = time.perf_counter()
                wav, _ = self.tts.synthesize(text=text, lang=lang, voice_style=style)
                elapsed = time.perf_counter() - started_at

                wav_int16 = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(wav_int16.tobytes())
                audio_bytes = buf.getvalue()

                responses.append(pb_utils.InferenceResponse([
                    pb_utils.Tensor("AUDIO_BYTES",       np.array([audio_bytes], dtype=np.object_)),
                    pb_utils.Tensor("AUDIO_FORMAT",      np.array([b"wav"],      dtype=np.object_)),
                    pb_utils.Tensor("SAMPLE_RATE",       np.array([SAMPLE_RATE], dtype=np.int32)),
                    pb_utils.Tensor("INFERENCE_SECONDS", np.array([elapsed],     dtype=np.float32)),
                ]))

            except Exception as exc:
                pb_utils.Logger.log_error(f"text2tts failed: {exc!r}")
                pb_utils.Logger.log_error(traceback.format_exc())
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(str(exc)),
                    )
                )

        return responses

    def finalize(self):
        pb_utils.Logger.log_info("Cleaning up text2tts model.")
