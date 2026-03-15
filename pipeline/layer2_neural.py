

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
import torch
import torch.nn.functional as F

from config import (
    FINETUNED_MODERNBERT_DIR as CFG_FINETUNED_MODERNBERT_DIR,
    MODERNBERT_MODEL as CFG_MODERNBERT_MODEL,
    LLAMA_GUARD_MODEL as CFG_LLAMA_GUARD_MODEL,
    OLLAMA_BASE_URL as CFG_OLLAMA_BASE_URL,
    LLAMA_GUARD_TIMEOUT as CFG_LLAMA_GUARD_TIMEOUT,
    USE_FINETUNED_MODERNBERT as CFG_USE_FINETUNED_MODERNBERT,
)

logger = logging.getLogger(__name__)


_tokenizer = None
_model = None
_device = None
_ref_store = None
_classifier_model = None
_classifier_available = False


MODERNBERT_MODEL = CFG_MODERNBERT_MODEL
LLAMA_GUARD_MODEL = CFG_LLAMA_GUARD_MODEL
OLLAMA_BASE_URL = CFG_OLLAMA_BASE_URL
LLAMA_GUARD_TIMEOUT = CFG_LLAMA_GUARD_TIMEOUT
USE_FINETUNED_MODERNBERT = CFG_USE_FINETUNED_MODERNBERT
FINETUNED_MODERNBERT_DIR = Path(CFG_FINETUNED_MODERNBERT_DIR)
MAX_LENGTH = 512

DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLE_PROMPTS_PATH = DATA_DIR / "sample_prompts.json"
HARMBENCH_PATH = DATA_DIR / "harmbench_behaviors.json"


MODEL_NAME = MODERNBERT_MODEL


LG3_CATEGORY_MAP: dict[str, str] = {
    "S1": "harmful_content",
    "S2": "harmful_content",
    "S3": "harmful_content",
    "S4": "harmful_content",
    "S5": "misinformation",
    "S6": "harmful_content",
    "S7": "pii_extraction",
    "S8": "data_exfiltration",
    "S9": "harmful_content",
    "S10": "harmful_content",
    "S11": "harmful_content",
    "S12": "harmful_content",
    "S13": "misinformation",
    "S14": "code_exploit",
}


_S2_CODE_TOKENS = {
    "hack", "exploit", "inject", "injection", "malware", "ransomware", "bypass",
    "vulnerability", "shell", "payload", "backdoor", "rootkit", "xss", "csrf",
    "overflow", "credential", "brute force", "ddos"
}
_S2_SOCIAL_TOKENS = {
    "fraud", "scam", "wire transfer", "phishing", "impersonat", "social engineer",
    "bec", "cfo", "ceo", "fake email", "pretext", "romance scam", "advance fee"
}
_S2_PII_TOKENS = {
    "password", "pin", "login", "credential", "bank account", "identity theft",
    "dox", "personal data"
}


def _refine_s2_category(prompt: str) -> str:

    lower = prompt.lower()
    if any(t in lower for t in _S2_CODE_TOKENS):
        return "code_exploit"
    if any(t in lower for t in _S2_SOCIAL_TOKENS):
        return "social_engineering"
    if any(t in lower for t in _S2_PII_TOKENS):
        return "pii_extraction"
    return "harmful_content"


_LG3_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "Task: Check if there is unsafe content in 'User' messages in conversations "
    "according our safety policy with the below categories.\n"
    "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
    "S1: Violent Crimes.\nS2: Non-Violent Crimes.\nS3: Sex Crimes.\n"
    "S4: Child Exploitation.\nS5: Defamation.\nS6: Specialist Advice.\n"
    "S7: Privacy.\nS8: Intellectual Property.\nS9: Indiscriminate Weapons.\n"
    "S10: Hate.\nS11: Suicide & Self-Harm.\nS12: Sexual Content.\n"
    "S13: Elections.\nS14: Code Interpreter Abuse.\n"
    "<END UNSAFE CONTENT CATEGORIES>\n\n"
    "<BEGIN CONVERSATION>\n\nUser: {user_message}\n\n<END CONVERSATION>\n\n"
    "Provide your safety assessment for ONLY THE LAST User message. "
    "Output 'safe' if it is safe, or 'unsafe' followed by the category code(s) "
    "if it is unsafe.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
)


CATEGORY_SEMANTIC_FLOORS: dict[str, float] = {
    "prompt_injection": 0.72,
    "jailbreak": 0.72,
    "system_prompt_leak": 0.72,
    "harmful_content": 0.70,
    "code_exploit": 0.70,
    "social_engineering": 0.71,
    "pii_extraction": 0.71,
    "misinformation": 0.73,
    "resource_abuse": 0.74,
    "data_exfiltration": 0.72,
}


@dataclass
class LlamaGuardResult:

    is_safe: bool
    raw_output: str
    category_codes: list[str]
    primary_category: str
    confidence: float
    inference_ms: float = 0.0
    available: bool = True


@dataclass
class ThreatScore:

    category: str
    llama_guard_score: float
    semantic_score: float
    combined_score: float
    evidence: list[str] = field(default_factory=list)

    @property
    def keyword_score(self) -> float:

        return self.llama_guard_score

    @property
    def matched_keywords(self) -> list[str]:

        return self.evidence


@dataclass
class NeuralClassifierResult:

    prompt: str

    threat_scores: dict[str, ThreatScore] = field(default_factory=dict)

    llama_guard: Optional[LlamaGuardResult] = None
    llama_guard_used: bool = False

    top_threat_category: str = "none"
    top_threat_score: float = 0.0
    second_threat_category: str = "none"
    second_threat_score: float = 0.0

    composite_threat_score: float = 0.0
    is_adversarial: bool = False

    model_name: str = MODERNBERT_MODEL
    embedding_dim: int = 768
    device_used: str = "cpu"

    model_load_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    total_time_ms: float = 0.0

    @property
    def scallop_facts(self) -> list[tuple[float, str, float]]:
        return [
            (ts.combined_score, cat, ts.combined_score)
            for cat, ts in self.threat_scores.items()
            if ts.combined_score > 0.03
        ]


def _call_llama_guard(prompt: str) -> LlamaGuardResult:

    formatted = _LG3_TEMPLATE.format(user_message=prompt[:3000])
    try:
        t0 = time.perf_counter()
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": LLAMA_GUARD_MODEL,
                "prompt": formatted,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 32, "stop": ["\n\n"]},
            },
            timeout=LLAMA_GUARD_TIMEOUT,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code == 404:
            logger.warning("llama-guard3 not found in Ollama. Run: ollama pull llama-guard3")
            return LlamaGuardResult(
                is_safe=True,
                raw_output="model_not_found",
                category_codes=[],
                primary_category="",
                confidence=0.5,
                inference_ms=elapsed_ms,
                available=False,
            )
        if resp.status_code != 200:
            logger.warning("Ollama returned HTTP %d", resp.status_code)
            return LlamaGuardResult(
                is_safe=True,
                raw_output=f"http_{resp.status_code}",
                category_codes=[],
                primary_category="",
                confidence=0.5,
                inference_ms=elapsed_ms,
                available=False,
            )

        raw = resp.json().get("response", "").strip().lower()
        if raw.startswith("unsafe"):
            codes = [f"S{n}" for n in re.findall(r"s(\d+)", raw)]
            primary_code = codes[0] if codes else "S1"
            primary_cat = _refine_s2_category(prompt) if primary_code == "S2" else LG3_CATEGORY_MAP.get(primary_code, "harmful_content")
            return LlamaGuardResult(
                is_safe=False,
                raw_output=raw,
                category_codes=codes,
                primary_category=primary_cat,
                confidence=0.92,
                inference_ms=elapsed_ms,
                available=True,
            )

        return LlamaGuardResult(
            is_safe=True,
            raw_output=raw,
            category_codes=[],
            primary_category="",
            confidence=0.05,
            inference_ms=elapsed_ms,
            available=True,
        )

    except requests.exceptions.ConnectionError:
        logger.warning("Ollama not reachable at %s", OLLAMA_BASE_URL)
        return LlamaGuardResult(
            is_safe=True,
            raw_output="connection_error",
            category_codes=[],
            primary_category="",
            confidence=0.5,
            available=False,
        )
    except requests.exceptions.Timeout:
        logger.warning("LlamaGuard timed out after %ds", LLAMA_GUARD_TIMEOUT)
        return LlamaGuardResult(
            is_safe=True,
            raw_output="timeout",
            category_codes=[],
            primary_category="",
            confidence=0.5,
            available=False,
        )
    except Exception as exc:
        logger.error("LlamaGuard error: %s", exc)
        return LlamaGuardResult(
            is_safe=True,
            raw_output=f"error:{exc}",
            category_codes=[],
            primary_category="",
            confidence=0.5,
            available=False,
        )


def _load_modernbert(device_pref: str = "cuda") -> tuple:

    global _tokenizer, _model, _device
    if _tokenizer is not None:
        return _tokenizer, _model, _device

    from transformers import AutoTokenizer, AutoModel

    _device = device_pref if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu"
    logger.info("Loading %s on %s …", MODERNBERT_MODEL, _device)

    t0 = time.perf_counter()
    _tokenizer = AutoTokenizer.from_pretrained(MODERNBERT_MODEL)
    _model = AutoModel.from_pretrained(
        MODERNBERT_MODEL,
           dtype=torch.float16 if _device == "cuda" else torch.float32,
    )
    _model.to(_device)
    _model.eval()
    logger.info("ModernBERT loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
    return _tokenizer, _model, _device


def _load_finetuned_modernbert(device_pref: str = "cuda") -> tuple | None:
    global _tokenizer, _classifier_model, _device, _classifier_available
    if not USE_FINETUNED_MODERNBERT or not FINETUNED_MODERNBERT_DIR.exists():
        return None
    if _classifier_available and _tokenizer is not None and _classifier_model is not None:
        return _tokenizer, _classifier_model, _device

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _device = device_pref if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu"
    logger.info("Loading fine-tuned ModernBERT from %s on %s …", FINETUNED_MODERNBERT_DIR, _device)
    t0 = time.perf_counter()
    _tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODERNBERT_DIR)
    _classifier_model = AutoModelForSequenceClassification.from_pretrained(
        FINETUNED_MODERNBERT_DIR,
           dtype=torch.float16 if _device == "cuda" else torch.float32,
    )
    _classifier_model.to(_device)
    _classifier_model.eval()
    _classifier_available = True
    logger.info("Fine-tuned ModernBERT loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
    return _tokenizer, _classifier_model, _device


def _load_model(device_pref: str = "cuda") -> tuple:
    return _load_modernbert(device_pref)


def _embed(text: str, tokenizer, model, device: str) -> torch.Tensor:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    attention_mask = inputs["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    embedding = sum_embeddings / sum_mask
    return F.normalize(embedding, p=2, dim=-1)


def _get_embedding(text: str, tokenizer, model, device: str) -> torch.Tensor:
    return _embed(text, tokenizer, model, device)


def _predict_finetuned_scores(text: str, tokenizer, model, device: str) -> dict[str, float]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
    id2label = getattr(model.config, "id2label", {}) or {}
    return {
        str(id2label.get(idx, idx)).lower(): float(prob)
        for idx, prob in enumerate(probs)
    }


class ReferenceEmbeddingStore:


    def __init__(self, tokenizer, model, device: str):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.embeddings: dict[str, list[torch.Tensor]] = {}
        self._build()

    def _build(self):
        logger.info("Building reference embedding store (sample + harmbench) …")
        t0 = time.perf_counter()

        if SAMPLE_PROMPTS_PATH.exists():
            with open(SAMPLE_PROMPTS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            for sample in data.get("adversarial", []):
                cat = sample["attack_type"]
                prompt = sample["prompt"]
                emb = _embed(prompt, self.tokenizer, self.model, self.device)
                self.embeddings.setdefault(cat, []).append(emb)
            # Build benign reference bank for discriminative scoring
            self._benign_embeddings: list[torch.Tensor] = []
            for sample in data.get("benign", []):
                prompt = sample["prompt"]
                emb = _embed(prompt, self.tokenizer, self.model, self.device)
                self._benign_embeddings.append(emb)
        else:
            self._benign_embeddings = []

        if HARMBENCH_PATH.exists():
            with open(HARMBENCH_PATH, encoding="utf-8") as f:
                hb = json.load(f)
            for row in hb.get("behaviors", []):
                cat = row["attack_type"]
                prompt = row["prompt"]
                emb = _embed(prompt, self.tokenizer, self.model, self.device)
                self.embeddings.setdefault(cat, []).append(emb)
        else:
            logger.warning("harmbench_behaviors.json missing — semantic coverage reduced")

        logger.info(
            "Reference store built: %d categories, %d embeddings, %d benign refs in %.0f ms",
            len(self.embeddings),
            sum(len(v) for v in self.embeddings.values()),
            len(self._benign_embeddings),
            (time.perf_counter() - t0) * 1000,
        )

    def top_k_similarity(self, query_emb: torch.Tensor, category: str, k: int = 5) -> float:
        refs = self.embeddings.get(category, [])
        if not refs:
            return 0.0
        sims = sorted(
            [float(F.cosine_similarity(query_emb, ref, dim=-1).item()) for ref in refs],
            reverse=True,
        )
        top_k = sims[:k]
        return sum(top_k) / len(top_k)

    def benign_baseline_similarity(self, query_emb: torch.Tensor) -> float:
        """Return average cosine similarity of the query to all benign reference examples.
        Used to discriminate adversarial signal from generic topic overlap."""
        refs = getattr(self, "_benign_embeddings", [])
        if not refs:
            return 0.70  # conservative default: treat as high benign overlap
        sims = [float(F.cosine_similarity(query_emb, ref, dim=-1).item()) for ref in refs]
        return sum(sims) / len(sims)


class NeuralClassifierLayer:


    CATEGORIES = [
        "prompt_injection",
        "jailbreak",
        "pii_extraction",
        "system_prompt_leak",
        "harmful_content",
        "code_exploit",
        "social_engineering",
        "misinformation",
        "resource_abuse",
        "data_exfiltration",
    ]

    def __init__(self, device: str = "cuda"):
        self._device_pref = device
        self._initialized = False

    def _ensure_initialized(self) -> float:
        global _ref_store
        if self._initialized:
            return 0.0
        t0 = time.perf_counter()
        finetuned = _load_finetuned_modernbert(self._device_pref)
        self._using_finetuned_classifier = finetuned is not None
        if finetuned is not None:
            tokenizer, model, device = finetuned
            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            self._ref_store = None
        else:
            tokenizer, model, device = _load_modernbert(self._device_pref)
            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            if _ref_store is None:
                _ref_store = ReferenceEmbeddingStore(tokenizer, model, device)
            self._ref_store = _ref_store
        self._initialized = True
        return (time.perf_counter() - t0) * 1000

    def classify(self, prompt: str) -> NeuralClassifierResult:
        t_total = time.perf_counter()
        load_time = self._ensure_initialized()

        result = NeuralClassifierResult(
            prompt=prompt,
            model_name=str(FINETUNED_MODERNBERT_DIR) if getattr(self, "_using_finetuned_classifier", False) else MODERNBERT_MODEL,
            device_used=self._device,
            model_load_time_ms=load_time,
        )


        lg3 = _call_llama_guard(prompt)
        result.llama_guard = lg3
        result.llama_guard_used = lg3.available

        lg3_scores: dict[str, float] = {cat: 0.0 for cat in self.CATEGORIES}
        if not lg3.is_safe and lg3.available:
            if lg3.primary_category in lg3_scores:
                lg3_scores[lg3.primary_category] = lg3.confidence
            else:
                lg3_scores["harmful_content"] = lg3.confidence

            for code in lg3.category_codes[1:]:
                sec_cat = _refine_s2_category(prompt) if code == "S2" else LG3_CATEGORY_MAP.get(code, "harmful_content")
                if sec_cat in lg3_scores:
                    lg3_scores[sec_cat] = max(lg3_scores[sec_cat], lg3.confidence * 0.80)


        t_inf = time.perf_counter()
        sem_scores: dict[str, float] = {}
        benign_prob = 0.0
        if getattr(self, "_using_finetuned_classifier", False):
            class_scores = _predict_finetuned_scores(prompt, self._tokenizer, self._model, self._device)
            result.inference_time_ms = round((time.perf_counter() - t_inf) * 1000, 2)
            benign_prob = class_scores.get("benign", 0.0)
            adv_denominator = max(1e-6, 1.0 - benign_prob)
            for category in self.CATEGORIES:
                raw_score = class_scores.get(category, 0.0)
                calibrated = max(0.0, raw_score - benign_prob) / adv_denominator
                sem_scores[category] = round(min(1.0, calibrated), 4)
        else:
            query_emb = _embed(prompt, self._tokenizer, self._model, self._device)
            result.inference_time_ms = round((time.perf_counter() - t_inf) * 1000, 2)

            benign_baseline = self._ref_store.benign_baseline_similarity(query_emb)
            for category in self.CATEGORIES:
                adv_sim = self._ref_store.top_k_similarity(query_emb, category, k=5)
                discriminative = max(0.0, adv_sim - benign_baseline)
                sem_scores[category] = round(min(1.0, discriminative / 0.18), 4)


        lg3_safe_and_available = lg3.available and lg3.is_safe
        # When LG3 is unavailable we have no reliable classifier — reduce trust in
        # raw semantic scores significantly rather than amplifying them.
        lg3_available = lg3.available
        high_sensitivity_when_lg3_safe = {
            "prompt_injection",
            "jailbreak",
            "system_prompt_leak",
        }

        all_scores: list[tuple[float, str]] = []
        for category in self.CATEGORIES:
            lg3_score = lg3_scores[category]
            if lg3_safe_and_available:
                sem_weight = 0.65 if getattr(self, "_using_finetuned_classifier", False) else (0.50 if category in high_sensitivity_when_lg3_safe else 0.20)
            elif lg3_available and not lg3.is_safe:
                sem_weight = 1.0
            else:
                sem_weight = 0.85 if getattr(self, "_using_finetuned_classifier", False) else 0.35
            sem_score = round(sem_scores[category] * sem_weight, 4)
            combined = round(max(lg3_score, sem_score), 4)

            evidence: list[str] = []
            if lg3_score > 0:
                evidence.append(f"LlamaGuard3:{lg3.category_codes}")
            if sem_score > 0.05:
                note = (
                    f"ModernBERT-finetuned:{sem_score:.3f}"
                    if getattr(self, "_using_finetuned_classifier", False)
                    else f"ModernBERT-discriminative:{sem_score:.3f}"
                )
                if lg3_safe_and_available:
                    note += "(discounted—LG3=safe)"
                elif not lg3_available:
                    note += "(LG3-unavailable-weight=0.35)"
                evidence.append(note)

            result.threat_scores[category] = ThreatScore(
                category=category,
                llama_guard_score=lg3_score,
                semantic_score=sem_score,
                combined_score=combined,
                evidence=evidence,
            )
            all_scores.append((combined, category))

        all_scores.sort(reverse=True)
        if all_scores:
            result.top_threat_category = all_scores[0][1]
            result.top_threat_score = all_scores[0][0]
        if len(all_scores) >= 2:
            result.second_threat_category = all_scores[1][1]
            result.second_threat_score = all_scores[1][0]

        top3 = [s for s, _ in all_scores[:3]]
        weighted_top3 = sum(score * weight for score, weight in zip(top3, [0.60, 0.25, 0.15]))
        result.composite_threat_score = round(max(result.top_threat_score, weighted_top3), 4)

        from config import NEURAL_THREAT_THRESHOLD
        result.is_adversarial = result.composite_threat_score >= NEURAL_THREAT_THRESHOLD

        result.total_time_ms = round((time.perf_counter() - t_total) * 1000, 2)

        logger.info(
            "Neural classifier: LG3=%s top=%s(%.3f), composite=%.3f, adversarial=%s [%.1fms]",
            "on" if lg3.available else "off",
            result.top_threat_category,
            result.top_threat_score,
            result.composite_threat_score,
            result.is_adversarial,
            result.total_time_ms,
        )
        return result
