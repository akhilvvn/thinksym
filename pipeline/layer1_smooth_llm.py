

import random
import string
import time
import logging
from dataclasses import dataclass, field
from typing import Callable

from config import SMOOTHLLM_N_PERTURBATIONS, SMOOTHLLM_CHAR_BUDGET, PERTURBATION_MODES

logger = logging.getLogger(__name__)


@dataclass
class PerturbedCopy:

    copy_id:          int
    original_length:  int
    perturbed_text:   str
    perturbation_mode: str
    chars_modified:   int
    modification_rate: float


@dataclass
class SmoothLLMResult:


    original_prompt:     str


    n_perturbations:     int
    perturbation_budget: float
    perturbed_copies:    list[PerturbedCopy] = field(default_factory=list)


    avg_char_edit_distance:  float = 0.0
    max_char_edit_distance:  float = 0.0
    semantic_coherence_score: float = 0.0


    high_variance_detected:  bool  = False
    structural_fragility:    float = 0.0


    flagged_as_fragile:     bool  = False
    confidence:             float = 0.0
    recommendation:         str   = "PROCEED"


    processing_time_ms:     float = 0.0


    canonical_prompt:       str   = ""

    @property
    def flag_level(self) -> str:
        if self.structural_fragility >= 0.70:
            return "HIGH"
        elif self.structural_fragility >= 0.45:
            return "MEDIUM"
        else:
            return "LOW"


def _perturb_insert(text: str, budget: float) -> tuple[str, int]:

    chars = list(text)
    n_inserts = max(1, int(len(chars) * budget * 0.5))
    for _ in range(n_inserts):
        pos = random.randint(0, len(chars))
        chars.insert(pos, random.choice(string.ascii_letters + string.digits + " "))
    return "".join(chars), n_inserts


def _perturb_delete(text: str, budget: float) -> tuple[str, int]:

    chars = list(text)
    n_deletes = max(1, int(len(chars) * budget * 0.4))
    n_deletes = min(n_deletes, len(chars) - 1)
    positions = sorted(random.sample(range(len(chars)), n_deletes), reverse=True)
    for pos in positions:
        chars.pop(pos)
    return "".join(chars), n_deletes


def _perturb_swap(text: str, budget: float) -> tuple[str, int]:

    chars = list(text)
    n_swaps = max(1, int(len(chars) * budget * 0.6))
    n_swaps = min(n_swaps, len(chars) - 1)
    swapped = 0
    positions = random.sample(range(len(chars) - 1), min(n_swaps, len(chars) - 1))
    for pos in positions:
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        swapped += 1
    return "".join(chars), swapped


def _perturb_substitute(text: str, budget: float) -> tuple[str, int]:

    similar_chars = {
        'a': ['@', '4', 'α'], 'e': ['3', 'é', 'ε'], 'i': ['1', '!', 'ì'],
        'o': ['0', 'ø', 'ο'], 's': ['$', '5', 'ś'], 'l': ['1', '|', 'ł'],
    }
    chars = list(text)
    n_subs = max(1, int(len(chars) * budget))
    positions = random.sample(range(len(chars)), min(n_subs, len(chars)))
    subbed = 0
    for pos in positions:
        c = chars[pos].lower()
        if c in similar_chars:
            chars[pos] = random.choice(similar_chars[c])
        else:
            chars[pos] = random.choice(string.ascii_letters)
        subbed += 1
    return "".join(chars), subbed


PERTURBATION_FN_MAP: dict[str, Callable] = {
    "insert":     _perturb_insert,
    "delete":     _perturb_delete,
    "swap":       _perturb_swap,
    "substitute": _perturb_substitute,
}


def _levenshtein(s1: str, s2: str) -> int:


    s1, s2 = s1[:500], s2[:500]
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _jaccard_similarity(text1: str, text2: str) -> float:

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 and not words2:
        return 1.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def _adversarial_fragility_score(original: str, copies: list[str]) -> float:

    if not copies:
        return 0.0


    adversarial_tokens = [
        "ignore", "override", "pretend", "jailbreak", "DAN", "unrestricted",
        "developer mode", "base64", "system prompt", "instructions:", "roleplay"
    ]
    orig_lower = original.lower()
    adversarial_token_density = sum(1 for t in adversarial_tokens if t.lower() in orig_lower)
    token_fragility = min(adversarial_token_density / 5.0, 1.0)


    coherence_scores = [_jaccard_similarity(original, copy) for copy in copies]
    avg_coherence = sum(coherence_scores) / len(coherence_scores)
    coherence_fragility = max(0.0, 1.0 - avg_coherence)


    variance = sum((s - avg_coherence) ** 2 for s in coherence_scores) / len(coherence_scores)
    variance_fragility = min(variance * 10, 1.0)


    fragility = (
        0.40 * token_fragility +
        0.35 * coherence_fragility +
        0.25 * variance_fragility
    )
    return round(min(fragility, 1.0), 4)


class SmoothLLMLayer:


    def __init__(
        self,
        n_perturbations: int = SMOOTHLLM_N_PERTURBATIONS,
        char_budget:      float = SMOOTHLLM_CHAR_BUDGET,
        modes:            list[str] | None = None,
        seed:             int | None = None,
    ):
        self.n_perturbations = n_perturbations
        self.char_budget     = char_budget
        self.modes           = modes or PERTURBATION_MODES
        if seed is not None:
            random.seed(seed)

    def _create_perturbed_copy(self, text: str, copy_id: int) -> PerturbedCopy:

        mode = random.choice(self.modes)
        fn   = PERTURBATION_FN_MAP[mode]
        perturbed_text, n_modified = fn(text, self.char_budget)
        return PerturbedCopy(
            copy_id           = copy_id,
            original_length   = len(text),
            perturbed_text    = perturbed_text,
            perturbation_mode = mode,
            chars_modified    = n_modified,
            modification_rate = round(n_modified / max(len(text), 1), 4),
        )

    def analyze(self, prompt: str) -> SmoothLLMResult:

        t_start = time.perf_counter()

        result = SmoothLLMResult(
            original_prompt     = prompt,
            n_perturbations     = self.n_perturbations,
            perturbation_budget = self.char_budget,
            canonical_prompt    = prompt.strip(),
        )


        copies = []
        for i in range(self.n_perturbations):
            copy = self._create_perturbed_copy(prompt, i)
            result.perturbed_copies.append(copy)
            copies.append(copy.perturbed_text)


        edit_distances = [_levenshtein(prompt, c) for c in copies]
        result.avg_char_edit_distance = round(sum(edit_distances) / len(edit_distances), 2)
        result.max_char_edit_distance = round(max(edit_distances), 2)


        coherence_scores = [_jaccard_similarity(prompt, c) for c in copies]
        result.semantic_coherence_score = round(sum(coherence_scores) / len(coherence_scores), 4)


        result.structural_fragility = _adversarial_fragility_score(prompt, copies)
        result.high_variance_detected = result.structural_fragility > 0.5


        result.confidence = round(result.structural_fragility, 4)
        result.flagged_as_fragile = result.structural_fragility >= 0.40

        if result.structural_fragility >= 0.70:
            result.recommendation = "BLOCK"
        elif result.structural_fragility >= 0.40:
            result.recommendation = "WARN"
        else:
            result.recommendation = "PROCEED"

        result.processing_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

        logger.info(
            "SmoothLLM: fragility=%.3f, recommendation=%s, time=%.1fms",
            result.structural_fragility, result.recommendation, result.processing_time_ms
        )
        return result
