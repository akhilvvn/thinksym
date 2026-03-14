

import logging
import os
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3.2:3b"


HF_MODELS = {
    "Mistral 7B Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "Zephyr 7B Beta":      "HuggingFaceH4/zephyr-7b-beta",
    "Phi-3 Mini (fast)":   "microsoft/Phi-3-mini-4k-instruct",
    "Qwen2.5 7B Instruct": "Qwen/Qwen2.5-7B-Instruct",
}
DEFAULT_MODEL_KEY = "Mistral 7B Instruct"
HF_API_URL        = "https://api-inference.huggingface.co/models/{model}"

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "You have been cleared by the ThinkSym security framework to answer this request. "
    "Respond concisely and accurately."
)


@dataclass
class LLMResponse:
    text:        str
    model:       str
    tokens_used: int   = 0
    latency_ms:  float = 0.0
    backend:     str   = "ollama"
    error:       str   = ""

    @property
    def success(self) -> bool:
        return bool(self.text) and not self.error


def ollama_available() -> bool:

    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def ollama_list_models() -> list[str]:

    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


def _call_ollama(
    prompt:     str,
    model:      str = OLLAMA_DEFAULT_MODEL,
    max_tokens: int = 512,
) -> LLMResponse:

    url  = f"{OLLAMA_BASE_URL}/api/chat"
    body = {
        "model":  model,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.7},
        "messages": [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": prompt},
        ],
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=body, timeout=120)
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            data   = resp.json()
            text   = data.get("message", {}).get("content", "").strip()
            tokens = data.get("eval_count", 0)
            return LLMResponse(text=text, model=model, tokens_used=tokens,
                               latency_ms=round(latency, 1), backend="ollama")
        return LLMResponse(
            text="", model=model, backend="ollama",
            error=f"Ollama error {resp.status_code}: {resp.text[:200]}",
        )
    except requests.Timeout:
        return LLMResponse(text="", model=model, backend="ollama",
                           error="Ollama timed out (120s).")
    except Exception as exc:
        return LLMResponse(text="", model=model, backend="ollama", error=str(exc))


def _call_hf_api(
    prompt:         str,
    model_id:       str,
    hf_token:       str,
    max_new_tokens: int = 512,
) -> LLMResponse:

    url     = HF_API_URL.format(model=model_id)
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {
        "inputs": f"[INST] {SYSTEM_PROMPT}\n\n{prompt} [/INST]",
        "parameters": {
            "max_new_tokens":   max_new_tokens,
            "temperature":      0.7,
            "top_p":            0.9,
            "do_sample":        True,
            "return_full_text": False,
        },
    }
    t0 = time.perf_counter()
    try:
        resp    = requests.post(url, headers=headers, json=payload, timeout=30)
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                text = data[0].get("generated_text", "").strip()
                return LLMResponse(text=text, model=model_id,
                                   latency_ms=round(latency, 1), backend="huggingface")
            return LLMResponse(text="", model=model_id, backend="huggingface",
                               error=f"Unexpected response: {data}")
        elif resp.status_code == 503:
            return LLMResponse(text="", model=model_id, backend="huggingface",
                               error="Model cold-starting on HF servers — retry in ~20s.")
        else:
            return LLMResponse(text="", model=model_id, backend="huggingface",
                               error=f"HF API {resp.status_code}: {resp.text[:200]}")
    except requests.Timeout:
        return LLMResponse(text="", model=model_id, backend="huggingface",
                           error="HF API timed out (30s).")
    except Exception as exc:
        return LLMResponse(text="", model=model_id, backend="huggingface", error=str(exc))


def get_llm_response(
    prompt:         str,
    hf_token:       str        = "",
    model_key:      str        = DEFAULT_MODEL_KEY,
    max_new_tokens: int        = 512,
    ollama_model:   str        = OLLAMA_DEFAULT_MODEL,
) -> LLMResponse:


    if ollama_available():
        models = ollama_list_models()

        if ollama_model and any(ollama_model in m for m in models):
            target = next(m for m in models if ollama_model in m)
        elif models:
            target = models[0]
        else:
            target = ollama_model

        logger.info("Routing to Ollama: model=%s", target)
        result = _call_ollama(prompt, target, max_new_tokens)
        if result.success:
            return result
        logger.warning("Ollama call failed: %s — trying HF API", result.error)


    token = hf_token or os.getenv("HF_TOKEN", "")
    if token:
        model_id = HF_MODELS.get(model_key, HF_MODELS[DEFAULT_MODEL_KEY])
        logger.info("Routing to HF API: model=%s", model_id)
        result = _call_hf_api(prompt, model_id, token, max_new_tokens)
        if result.success:
            return result
        logger.warning("HF API call failed: %s", result.error)
        return result


    return LLMResponse(
        text="",
        model="none",
        backend="none",
        error=(
            "No LLM backend available. "
            "Start Ollama (`ollama serve`) and pull a model (`ollama pull llama3.2:3b`), "
            "or provide a HuggingFace API token in the sidebar."
        ),
    )
