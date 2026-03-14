

import logging
import math
import time
import importlib
from dataclasses import dataclass, field
from typing import Any

from config import ESCALATION_WINDOW, SCALLOP_ESCALATION_DELTA, SCALLOP_SIGMOID_K

logger = logging.getLogger(__name__)


try:
    scallopy = importlib.import_module("scallopy")
    SCALLOP_AVAILABLE = True
except ImportError:
    scallopy = None
    SCALLOP_AVAILABLE = False


@dataclass
class ScallopFact:
    relation: str
    args: tuple
    probability: float


@dataclass
class ScallopRule:
    head: str
    body: str
    comment: str = ""


@dataclass
class EscalationEvidence:
    turn_ids: list[int]
    threat_categories: list[str]
    score_trajectory: list[float]
    delta_scores: list[float]
    escalation_rate: float
    peak_score: float


@dataclass
class ScallopReasonerResult:
    input_facts: list[ScallopFact] = field(default_factory=list)
    rules_applied: list[str] = field(default_factory=list)

    adversarial_score: float = 0.0
    policy_violation_score: float = 0.0
    temporal_escalation_score: float = 0.0
    harmful_content_score: float = 0.0
    injection_score: float = 0.0
    jailbreak_score: float = 0.0

    session_turn_count: int = 0
    escalation_evidence: EscalationEvidence | None = None
    multi_turn_attack_detected: bool = False

    symbolic_risk_score: float = 0.0
    violated_policies: list[str] = field(default_factory=list)
    derived_facts: list[tuple[str, Any, float]] = field(default_factory=list)

    engine_used: str = "scallopy" if SCALLOP_AVAILABLE else "python-probabilistic"
    scallop_program: str = ""
    processing_time_ms: float = 0.0

    @property
    def risk_label(self) -> str:
        if self.symbolic_risk_score >= 0.80:
            return "CRITICAL"
        if self.symbolic_risk_score >= 0.60:
            return "HIGH"
        if self.symbolic_risk_score >= 0.40:
            return "MEDIUM"
        if self.symbolic_risk_score >= 0.20:
            return "LOW"
        return "SAFE"


SCALLOP_PROGRAM_TEMPLATE = """
type neural_threat(category: String, score: f32)
type session_turn(turn_id: i32, category: String, score: f32)

rel cat_risk(c, p) = neural_threat(c, s), p = sigmoid(k*(s-t(c)))

rel compound_attack(c1, c2, p) =
  cat_risk(c1, p1), cat_risk(c2, p2), c1 != c2,
  p = noisy_or(p1, p2)

rel escalation(cat, p) =
  session_turn(t1, cat, s1), session_turn(t2, cat, s2), t2 == t1 + 1,
  p = sigmoid(k*(s2 - s1 - delta))

rel attack_chain(p) =
  session_turn(t0, c0, s0), session_turn(t1, c1, s1), session_turn(t2, c2, s2),
  t0 < t1, t1 < t2,
  chain(c0, c1, c2),
  p = combine(s0, s1, s2)

rel policy_violated(policy, p) = posterior(policy, p), p > policy_threshold(policy)
"""


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sigmoid(value: float, sharpness: float = SCALLOP_SIGMOID_K) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-sharpness * value))
    except OverflowError:
        return 0.0 if value < 0 else 1.0


def _noisy_or(probabilities: list[float]) -> float:

    if not probabilities:
        return 0.0
    product = 1.0
    for p in probabilities:
        product *= (1.0 - _clamp01(p))
    return _clamp01(1.0 - product)


def _soft_threshold(score: float, threshold: float) -> float:

    return _sigmoid(score - threshold)


def _pairwise_compound_score(neural_scores: dict[str, float]) -> tuple[float, list[tuple[str, str, float]]]:

    categories = list(neural_scores.keys())
    pair_scores: list[tuple[str, str, float]] = []

    for i, cat1 in enumerate(categories):
        for cat2 in categories[i + 1 :]:
            s1 = neural_scores.get(cat1, 0.0)
            s2 = neural_scores.get(cat2, 0.0)
            if s1 < 0.10 and s2 < 0.10:
                continue

            moderate_bonus = 0.10 if (s1 >= 0.28 and s2 >= 0.28) else 0.0
            pair = _clamp01((s1 * s2) + moderate_bonus)
            if pair > 0.08:
                pair_scores.append((cat1, cat2, round(pair, 4)))

    if not pair_scores:
        return 0.0, []

    combined = _noisy_or([p for _, _, p in pair_scores])
    return round(combined, 4), sorted(pair_scores, key=lambda x: x[2], reverse=True)


def _group_session_by_cat(session_facts: list[tuple[int, str, float]]) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for turn_id, category, score in session_facts:
        grouped.setdefault(category, []).append((turn_id, score))
    for category in grouped:
        grouped[category].sort(key=lambda x: x[0])
    return grouped


def _temporal_escalation_score(session_facts: list[tuple[int, str, float]]) -> tuple[float, list[tuple[int, str, float, float]]]:

    if len(session_facts) < 2:
        return 0.0, []

    grouped = _group_session_by_cat(session_facts)
    events: list[tuple[int, str, float, float]] = []

    for category, turns in grouped.items():
        for index in range(1, len(turns)):
            t_prev, s_prev = turns[index - 1]
            t_curr, s_curr = turns[index]
            if t_curr != t_prev + 1:
                continue
            if s_prev < 0.16 or s_curr < 0.16:
                continue
            delta = s_curr - s_prev
            strength = _soft_threshold(delta, SCALLOP_ESCALATION_DELTA)
            if strength > 0.45:
                events.append((t_curr, category, round(delta, 4), round(strength, 4)))

    if not events:
        return 0.0, []

    score = _noisy_or([e[3] for e in events])
    return round(score, 4), sorted(events, key=lambda x: x[3], reverse=True)


def _cross_category_chain_score(session_facts: list[tuple[int, str, float]]) -> tuple[float, list[tuple[list[str], float]]]:

    if len(session_facts) < 3:
        return 0.0, []

    dominant_by_turn: dict[int, tuple[str, float]] = {}
    for turn_id, category, score in session_facts:
        if turn_id not in dominant_by_turn or score > dominant_by_turn[turn_id][1]:
            dominant_by_turn[turn_id] = (category, score)

    ordered = [(turn, *dominant_by_turn[turn]) for turn in sorted(dominant_by_turn.keys())]
    recent = ordered[-ESCALATION_WINDOW:]
    if len(recent) < 3:
        return 0.0, []

    chains = [
        ["social_engineering", "pii_extraction", "code_exploit"],
        ["social_engineering", "prompt_injection", "code_exploit"],
        ["prompt_injection", "jailbreak", "code_exploit"],
        ["prompt_injection", "system_prompt_leak", "pii_extraction"],
        ["social_engineering", "harmful_content", "code_exploit"],
        ["prompt_injection", "harmful_content", "code_exploit"],
    ]

    cats = [cat for _, cat, _ in recent]
    scores = [score for _, _, score in recent]

    hits: list[tuple[list[str], float]] = []
    for chain in chains:
        idx = 0
        supporting_scores: list[float] = []
        for cat, score in zip(cats, scores):
            if cat == chain[idx]:
                supporting_scores.append(score)
                idx += 1
                if idx == len(chain):
                    break
        if idx == len(chain):
            avg_support = sum(supporting_scores) / len(supporting_scores)
            chain_strength = _soft_threshold(avg_support, 0.26)
            hits.append((chain, round(chain_strength, 4)))

    if not hits:
        return 0.0, []

    score = _noisy_or([h[1] for h in hits])
    return round(score, 4), sorted(hits, key=lambda x: x[1], reverse=True)


def _policy_violations(
    neural_scores: dict[str, float],
    adversarial_score: float,
    compound_score: float,
    temporal_score: float,
    chain_score: float,
) -> tuple[list[str], float]:

    violated: list[str] = []

    jailbreak = neural_scores.get("jailbreak", 0.0)
    pii = neural_scores.get("pii_extraction", 0.0)
    system_leak = neural_scores.get("system_prompt_leak", 0.0)
    code_exploit = neural_scores.get("code_exploit", 0.0)

    if _soft_threshold(jailbreak, 0.52) > 0.72:
        violated.append("P-001: No Jailbreak Tolerated")
    if _soft_threshold(pii, 0.35) > 0.60:
        violated.append("P-002: PII Extraction Guard")
    if _soft_threshold(system_leak, 0.50) > 0.72:
        violated.append("P-003: System Prompt Confidentiality")
    if _soft_threshold(code_exploit, 0.40) > 0.60:
        violated.append("P-004: Code Exploit Prevention")
    temporal_chain = _noisy_or([temporal_score, chain_score])
    if temporal_score >= 0.72 or chain_score >= 0.72 or temporal_chain >= 0.78:
        violated.append("P-005: Multi-turn Escalation")
    composite_symbolic = _noisy_or([adversarial_score, compound_score])
    critical_peak = max(
        neural_scores.get("jailbreak", 0.0),
        neural_scores.get("prompt_injection", 0.0),
        neural_scores.get("code_exploit", 0.0),
        neural_scores.get("harmful_content", 0.0),
        neural_scores.get("pii_extraction", 0.0),
        neural_scores.get("system_prompt_leak", 0.0),
    )
    if composite_symbolic >= 0.86 and critical_peak >= 0.58:
        violated.append("P-006: Composite Threat")


    violation_score = _clamp01(len(violated) / 4.0)
    return violated, round(violation_score, 4)


class ScallopReasonerLayer:


    def __init__(self):
        self._session_history: list[tuple[int, str, float]] = []
        self._current_turn = 0
        self._engine = "scallopy" if SCALLOP_AVAILABLE else "python-probabilistic"

    def reset_session(self):
        self._session_history.clear()
        self._current_turn = 0

    def reason(self, neural_scores: dict[str, float], update_session: bool = True) -> ScallopReasonerResult:
        t0 = time.perf_counter()
        result = ScallopReasonerResult(engine_used=self._engine)
        result.scallop_program = SCALLOP_PROGRAM_TEMPLATE


        neural_facts = [(cat, round(score, 4)) for cat, score in neural_scores.items()]
        result.input_facts = [ScallopFact("neural_threat", (cat, score), score) for cat, score in neural_facts]


        turn_id = self._current_turn
        if update_session:
            persisted = False
            for cat, score in sorted(neural_facts, key=lambda x: x[1], reverse=True):
                if score >= 0.18:
                    self._session_history.append((turn_id, cat, score))
                    persisted = True
            if not persisted:
                top_candidates = sorted(neural_facts, key=lambda x: x[1], reverse=True)[:2]
                for cat, score in top_candidates:
                    if score >= 0.12:
                        self._session_history.append((turn_id, cat, score))
                        persisted = True
            if not persisted and neural_facts:
                top_cat, top_score = max(neural_facts, key=lambda x: x[1])
                self._session_history.append((turn_id, top_cat, max(top_score, 0.01)))
            self._current_turn += 1

        result.session_turn_count = self._current_turn


        result.injection_score = round(neural_scores.get("prompt_injection", 0.0), 4)
        result.jailbreak_score = round(neural_scores.get("jailbreak", 0.0), 4)
        result.harmful_content_score = round(neural_scores.get("harmful_content", 0.0), 4)


        strong_channels = [
            neural_scores.get("jailbreak", 0.0),
            neural_scores.get("prompt_injection", 0.0),
            neural_scores.get("code_exploit", 0.0),
            neural_scores.get("harmful_content", 0.0),
            neural_scores.get("pii_extraction", 0.0),
            neural_scores.get("system_prompt_leak", 0.0),
        ]
        active_strong = [score for score in strong_channels if score >= 0.15]
        strong_posteriors = [_soft_threshold(score, 0.52) for score in active_strong]
        result.adversarial_score = round(_noisy_or(strong_posteriors), 4) if strong_posteriors else 0.0


        compound_score, compound_pairs = _pairwise_compound_score(neural_scores)


        temporal_score, temporal_events = _temporal_escalation_score(self._session_history)
        result.temporal_escalation_score = temporal_score


        chain_score, chain_hits = _cross_category_chain_score(self._session_history)


        multi_turn_signal = _noisy_or([temporal_score, chain_score])
        result.multi_turn_attack_detected = multi_turn_signal > 0.72


        if temporal_events:
            top_category = temporal_events[0][1]
            grouped = _group_session_by_cat(self._session_history)
            traj = [s for _, s in grouped.get(top_category, [])]
            deltas = [round(traj[i] - traj[i - 1], 4) for i in range(1, len(traj))]
            if traj:
                result.escalation_evidence = EscalationEvidence(
                    turn_ids=[t for t, _ in grouped.get(top_category, [])],
                    threat_categories=[top_category] * len(traj),
                    score_trajectory=traj,
                    delta_scores=deltas,
                    escalation_rate=round(sum(deltas) / len(deltas), 4) if deltas else 0.0,
                    peak_score=max(traj),
                )


        violated, policy_score = _policy_violations(
            neural_scores=neural_scores,
            adversarial_score=result.adversarial_score,
            compound_score=compound_score,
            temporal_score=temporal_score,
            chain_score=chain_score,
        )
        result.violated_policies = violated
        result.policy_violation_score = policy_score


        result.symbolic_risk_score = round(
            _clamp01(
                0.34 * result.adversarial_score
                + 0.20 * compound_score
                + 0.24 * temporal_score
                + 0.12 * chain_score
                + 0.10 * policy_score
            ),
            4,
        )


        result.derived_facts = []
        result.derived_facts.append(("adversarial_posterior", "global", result.adversarial_score))
        result.derived_facts.append(("compound_posterior", compound_pairs[:3], compound_score))
        result.derived_facts.append(("temporal_posterior", temporal_events[:3], temporal_score))
        result.derived_facts.append(("chain_posterior", chain_hits[:2], chain_score))
        for policy in violated:
            result.derived_facts.append(("policy_violated", policy, 1.0))


        result.rules_applied = [
            "cat_risk(c) := sigmoid(k*(neural_score(c)-t_c))",
            "compound_attack := noisy_or(pairwise_soft_conjunctions)",
            "temporal_escalation := noisy_or(sigmoid(delta_score - delta_min))",
            "attack_chain := sequence_match(social/prompt -> pii/leak -> exploit)",
            "policy_violated := posterior(policy) > policy_threshold",
            "symbolic_risk := weighted_sum(adversarial, compound, temporal, chain, policy)",
        ]

        result.processing_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "Scallop [%s]: risk=%.3f adv=%.3f temporal=%.3f chain=%.3f policies=%d [%.1fms]",
            result.engine_used,
            result.symbolic_risk_score,
            result.adversarial_score,
            temporal_score,
            chain_score,
            len(result.violated_policies),
            result.processing_time_ms,
        )
        return result
