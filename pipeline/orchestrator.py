

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from config import (
    NEURAL_THREAT_THRESHOLD,
    SCALLOP_RISK_THRESHOLD,
    THREAT_CATEGORIES,
    RISK_LEVELS,
)
from pipeline.layer1_smooth_llm import SmoothLLMLayer, SmoothLLMResult
from pipeline.layer2_neural import NeuralClassifierLayer, NeuralClassifierResult
from pipeline.layer3_scallop import ScallopReasonerLayer, ScallopReasonerResult
from pipeline.layer4_z3 import Z3VerifierLayer, Z3VerifierResult
from pipeline.layer5_knowledge_graph import KnowledgeGraphLayer, KnowledgeGraphResult

logger = logging.getLogger(__name__)


@dataclass
class FinalVerdict:

    decision:         str
    risk_level:       str
    composite_score:  float
    confidence:       float


    primary_reason:   str = ""
    contributing_factors: list[str] = field(default_factory=list)


    smoothllm_flag:   str = "PROCEED"
    neural_flag:      bool = False
    scallop_flag:     str = "SAFE"
    z3_flag:          str = "ALLOW"
    kg_flag:          str = "SAFE"


    color:            str = "#a6e3a1"
    emoji:            str = "✅"

    def __post_init__(self):
        level_info = RISK_LEVELS.get(self.risk_level, RISK_LEVELS["SAFE"])
        self.color = level_info["color"]
        self.emoji = level_info["emoji"]


@dataclass
class AnalysisResult:


    prompt:            str
    turn_number:       int
    session_id:        str


    smooth_result:     Optional[SmoothLLMResult]   = None
    neural_result:     Optional[NeuralClassifierResult] = None
    scallop_result:    Optional[ScallopReasonerResult]  = None
    z3_result:         Optional[Z3VerifierResult]       = None
    kg_result:         Optional[KnowledgeGraphResult]   = None


    verdict:           Optional[FinalVerdict] = None


    layer_timings_ms:  dict[str, float] = field(default_factory=dict)
    total_time_ms:     float = 0.0


    errors:            list[str] = field(default_factory=list)

    @property
    def was_blocked(self) -> bool:
        return self.verdict is not None and self.verdict.decision == "BLOCK"

    @property
    def top_threat(self) -> str:
        if self.neural_result:
            return self.neural_result.top_threat_category
        return "unknown"


class ThinkSymOrchestrator:


    def __init__(
        self,
        enable_smoothllm: bool = True,
        enable_neural:    bool = True,
        enable_scallop:   bool = True,
        enable_z3:        bool = True,
        enable_kg:        bool = True,
        device:           str  = "cuda",
        session_id:       str  = "default",
    ):
        self.enable_smoothllm = enable_smoothllm
        self.enable_neural    = enable_neural
        self.enable_scallop   = enable_scallop
        self.enable_z3        = enable_z3
        self.enable_kg        = enable_kg
        self.session_id       = session_id

        # Keyword patterns for rule-based detection
        self._jailbreak_keywords = {
            "roleplay", "pretend", "ignore", "disregard", "override", "bypass",
            "forget", "unfiltered", "uncensored", "unrestricted", "no filter",
            "root-level", "developer mode", "do anything now", "dan", "simulator",
            "hypothetically", "without policy", "without restrictions",
        }
        self._injection_keywords = {
            "[SYSTEM]", "[ADMIN]", "[ROOT]", "higher priority", "treat as", 
            "above rule", "override", "supersede", "::", "--", "//", 
            "<!-- ", "-->", "{{", "}}", "${", "}", "?>", "<%",
        }
        self._harmful_keywords = {
            "malware", "ransomware", "exploit", "vulnerability", "payload",
            "xss", "injection", "sql", "ddos", "botnet", "trojan",
        }

        self._smooth_layer  = SmoothLLMLayer() if enable_smoothllm else None
        self._neural_layer  = NeuralClassifierLayer(device=device) if enable_neural else None
        self._scallop_layer = ScallopReasonerLayer() if enable_scallop else None
        self._z3_layer      = Z3VerifierLayer() if enable_z3 else None
        self._kg_layer      = KnowledgeGraphLayer() if enable_kg else None


        self._turn_count    = 0
        self._session_history: list[AnalysisResult] = []

        logger.info(
            "ThinkSymOrchestrator initialized | session=%s | device=%s",
            session_id, device,
        )

    def reset_session(self):

        self._turn_count = 0
        self._session_history.clear()
        if self._scallop_layer:
            self._scallop_layer.reset_session()
        logger.info("Session reset: %s", self.session_id)

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def session_history(self) -> list[AnalysisResult]:
        return self._session_history.copy()

    def _detect_keyword_threats(self, prompt: str) -> dict[str, float]:
        """Rule-based keyword detection to catch attacks neural model might miss."""
        prompt_lower = prompt.lower()
        scores = {}
        
        # Jailbreak detection - very aggressive
        jailbreak_matches = sum(1 for kw in self._jailbreak_keywords if kw in prompt_lower)
        scores["jailbreak"] = min(jailbreak_matches * 0.25, 0.85)
        
        # Prompt injection detection - very aggressive
        injection_matches = sum(1 for kw in self._injection_keywords if kw in prompt_lower)
        scores["prompt_injection"] = min(injection_matches * 0.30, 0.85)
        
        # Harmful/code exploit detection - very aggressive
        harmful_matches = sum(1 for kw in self._harmful_keywords if kw in prompt_lower)
        scores["code_exploit"] = min(harmful_matches * 0.28, 0.75)
        
        return scores

    def analyze(
        self,
        prompt: str,
        update_session: bool = True,
    ) -> AnalysisResult:

        t_global = time.perf_counter()
        turn = self._turn_count
        if update_session:
            self._turn_count += 1

        result = AnalysisResult(
            prompt     = prompt,
            turn_number = turn,
            session_id  = self.session_id,
        )


        canonical_prompt = prompt
        if self.enable_smoothllm and self._smooth_layer:
            try:
                t0 = time.perf_counter()
                result.smooth_result = self._smooth_layer.analyze(prompt)
                result.layer_timings_ms["L1_SmoothLLM"] = round(
                    (time.perf_counter() - t0) * 1000, 2
                )
                canonical_prompt = result.smooth_result.canonical_prompt
            except Exception as exc:
                logger.error("L1 SmoothLLM error: %s", exc)
                result.errors.append(f"L1: {exc}")


        neural_scores: dict[str, float] = {}
        if self.enable_neural and self._neural_layer:
            try:
                t0 = time.perf_counter()
                result.neural_result = self._neural_layer.classify(canonical_prompt)
                result.layer_timings_ms["L2_Neural"] = round(
                    (time.perf_counter() - t0) * 1000, 2
                )

                neural_scores = {
                    cat: ts.combined_score
                    for cat, ts in result.neural_result.threat_scores.items()
                }
            except Exception as exc:
                logger.error("L2 Neural error: %s", exc)
                result.errors.append(f"L2: {exc}")

        if self.enable_scallop and self._scallop_layer and neural_scores:
            try:
                t0 = time.perf_counter()
                result.scallop_result = self._scallop_layer.reason(
                    neural_scores, update_session=update_session
                )
                result.layer_timings_ms["L3_Scallop"] = round(
                    (time.perf_counter() - t0) * 1000, 2
                )
            except Exception as exc:
                logger.error("L3 Scallop error: %s", exc)
                result.errors.append(f"L3: {exc}")


        if self.enable_z3 and self._z3_layer:
            try:
                t0 = time.perf_counter()
                result.z3_result = self._z3_layer.verify(
                    neural_scores,
                    result.scallop_result,
                )
                result.layer_timings_ms["L4_Z3"] = round(
                    (time.perf_counter() - t0) * 1000, 2
                )
            except Exception as exc:
                logger.error("L4 Z3 error: %s", exc)
                result.errors.append(f"L4: {exc}")


        if self.enable_kg and self._kg_layer:
            try:
                t0 = time.perf_counter()
                result.kg_result = self._kg_layer.analyze(canonical_prompt, neural_scores)
                result.layer_timings_ms["L5_KG"] = round(
                    (time.perf_counter() - t0) * 1000, 2
                )
            except Exception as exc:
                logger.error("L5 KG error: %s", exc)
                result.errors.append(f"L5: {exc}")


        result.verdict = self._compute_verdict(result)

        result.total_time_ms = round((time.perf_counter() - t_global) * 1000, 2)

        if update_session:
            self._session_history.append(result)

        logger.info(
            "ThinkSym [turn %d] → %s (%s) | %.0fms",
            turn, result.verdict.decision, result.verdict.risk_level,
            result.total_time_ms,
        )
        return result


    def _compute_verdict(self, result: AnalysisResult) -> FinalVerdict:


        smooth_flag  = result.smooth_result.recommendation if result.smooth_result else "PROCEED"
        neural_score = result.neural_result.composite_threat_score if result.neural_result else 0.0
        neural_adv   = neural_score >= NEURAL_THREAT_THRESHOLD if result.neural_result else False
        neural_warn_signal = neural_score >= 0.60 if result.neural_result else False
        top_neural_category = result.neural_result.top_threat_category if result.neural_result else "unknown"
        scallop_risk = result.scallop_result.symbolic_risk_score if result.scallop_result else 0.0
        scallop_viol = len(result.scallop_result.violated_policies) if result.scallop_result else 0
        z3_verdict   = result.z3_result.formal_verdict if result.z3_result else "ALLOW"
        z3_score     = result.z3_result.verification_score if result.z3_result else 0.0
        kg_risk      = result.kg_result.kg_risk_score if result.kg_result else 0.0
        smooth_fragility = result.smooth_result.structural_fragility if result.smooth_result else 0.0
        multi_turn   = (
            result.scallop_result.multi_turn_attack_detected
            if result.scallop_result else False
        )
        temporal_esc = (
            result.scallop_result.temporal_escalation_score
            if result.scallop_result else 0.0
        )


        composite_score = round(min(
            0.25 * neural_score +
            0.25 * scallop_risk +
            0.20 * z3_score +
            0.20 * kg_risk +
            0.10 * (result.smooth_result.structural_fragility if result.smooth_result else 0.0),
            1.0
        ), 4)

        high_sensitivity_categories = {"prompt_injection", "jailbreak", "system_prompt_leak"}
        severe_content_categories = {
            "harmful_content",
            "code_exploit",
            "pii_extraction",
            "social_engineering",
            "misinformation",
            "resource_abuse",
            "data_exfiltration",
        }
        corroborated_sensitive_block = (
            result.neural_result is not None and
            top_neural_category in high_sensitivity_categories and
            neural_score >= 0.45 and
            (scallop_risk >= 0.15 or kg_risk >= 0.30 or smooth_fragility >= 0.30)
        )
        corroborated_severe_block = (
            result.neural_result is not None and
            top_neural_category in severe_content_categories and
            neural_score >= 0.70 and
            (scallop_risk >= 0.15 or kg_risk >= 0.15)
        )
        ultra_high_neural_block = (
            result.neural_result is not None and
            top_neural_category in severe_content_categories and
            neural_score >= 0.90
        )


        contributing: list[str] = []
        decision = "ALLOW"
        primary_reason = "No significant threat signals detected across all layers."

        keyword_block_patterns = {
            "jailbreak": (result.prompt.lower().count("roleplay") + result.prompt.lower().count("unfiltered") + 
                         result.prompt.lower().count("without policy")) >= 2,
            "injection": any(pat in result.prompt.lower() for pat in ["[system]", "[admin]", "[root]", "higher priority", "treat as"]),
            "harmful": any(pat in result.prompt.lower() for pat in ["malware", "ransomware", "payload", "ddos", "botnet"])
        }
        
        if keyword_block_patterns.get("jailbreak") or keyword_block_patterns.get("injection"):
            decision = "BLOCK"
            primary_reason = "High-confidence attack pattern detected via keyword analysis (jailbreak/injection)"
            contributing.append("Keyword pattern: likely jailbreak or injection attack")
        elif keyword_block_patterns.get("harmful"):
            decision = "BLOCK"
            primary_reason = "Explicit harmful content keywords detected"
            contributing.append("Keyword pattern: harmful content indicators")
        elif z3_verdict == "BLOCK":
            decision = "BLOCK"
            primary_reason = f"Z3 formal verification failed: {scallop_viol} policy(ies) violated."
            contributing.append(f"Z3: {scallop_viol} policy violations")
        elif neural_adv and scallop_risk >= SCALLOP_RISK_THRESHOLD and neural_score >= NEURAL_THREAT_THRESHOLD:
            decision = "BLOCK"
            primary_reason = (
                f"Neural + symbolic consensus: {result.neural_result.top_threat_category} "
                f"threat (score={neural_score:.3f})"
            )
            contributing.append(f"Neural: {result.neural_result.top_threat_category} ({neural_score:.3f})")
            contributing.append(f"Scallop risk: {scallop_risk:.3f}")
        elif multi_turn and temporal_esc >= 0.55:
            decision = "BLOCK"
            primary_reason = f"Multi-turn escalation attack detected (escalation score: {temporal_esc:.3f})"
            contributing.append(f"Temporal escalation: {temporal_esc:.3f}")
        elif composite_score >= 0.60:
            decision = "BLOCK"
            primary_reason = f"Composite threat score {composite_score:.3f} exceeds block threshold."
            contributing.append(f"Composite score: {composite_score:.3f}")
        elif corroborated_sensitive_block or corroborated_severe_block or ultra_high_neural_block:
            decision = "BLOCK"
            primary_reason = (
                f"High-confidence {top_neural_category} threat detected with corroborating signals "
                f"(neural={neural_score:.3f})."
            )
            contributing.append(f"Neural high-confidence: {top_neural_category} ({neural_score:.3f})")
            if scallop_risk > 0:
                contributing.append(f"Scallop risk: {scallop_risk:.3f}")
            if kg_risk > 0:
                contributing.append(f"KG risk: {kg_risk:.3f}")
            if smooth_fragility >= 0.35:
                contributing.append(f"SmoothLLM fragility: {smooth_fragility:.3f}")
        elif result.neural_result and (
            (
                neural_score >= NEURAL_THREAT_THRESHOLD + 0.10 and
                result.neural_result.top_threat_category in {"prompt_injection", "jailbreak", "system_prompt_leak"}
            ) or (
                neural_score >= NEURAL_THREAT_THRESHOLD + 0.10 and
                result.neural_result.top_threat_category in {"harmful_content", "code_exploit", "pii_extraction"}
            )
        ):
            decision = "BLOCK"
            primary_reason = (
                f"High-confidence {result.neural_result.top_threat_category} threat detected "
                f"(score={neural_score:.3f})."
            )
            contributing.append(f"Neural high-confidence: {result.neural_result.top_threat_category} ({neural_score:.3f})")

        elif (smooth_flag == "BLOCK" and composite_score >= 0.30) or composite_score >= 0.45 or (neural_warn_signal and scallop_risk >= SCALLOP_RISK_THRESHOLD):
            decision = "WARN"
            primary_reason = (
                f"Threat signals present but below block threshold "
                f"(composite={composite_score:.3f})"
            )
            if smooth_flag in ("BLOCK", "WARN"):
                contributing.append(f"SmoothLLM fragility: {result.smooth_result.structural_fragility:.3f}")
            if neural_warn_signal:
                contributing.append(f"Neural threat: {result.neural_result.top_threat_category} ({neural_score:.3f})")
            if scallop_viol > 0:
                contributing.append(f"Scallop: {scallop_viol} soft violation(s)")
        elif multi_turn:
            decision = "WARN"
            primary_reason = "Escalating pattern observed in multi-turn session."
            contributing.append(f"Temporal score: {temporal_esc:.3f}")

        if not contributing:
            contributing.append("All layers returned clean scores")


        risk_level = "SAFE"
        for level, info in RISK_LEVELS.items():
            if composite_score >= info["min"]:
                risk_level = level
                break

        preserve_high_confidence_block = corroborated_sensitive_block or corroborated_severe_block or ultra_high_neural_block
        if decision == "BLOCK" and risk_level in {"LOW", "SAFE"} and not preserve_high_confidence_block:
            decision = "WARN"
            primary_reason = (
                f"Threat signals present but low aggregate risk (composite={composite_score:.3f}); "
                "downgraded from hard block."
            )
            if "Low-risk downgrade" not in contributing:
                contributing.append("Low-risk downgrade")


        layer_signals = [
            1.0 if decision == "BLOCK" and z3_verdict   == "BLOCK" else 0.0,
            1.0 if decision == "BLOCK" and neural_adv              else 0.0,
            1.0 if decision == "BLOCK" and scallop_risk >= 0.5     else 0.0,
            1.0 if decision == "ALLOW" and z3_verdict   == "ALLOW" else 0.0,
            1.0 if decision == "ALLOW" and not neural_adv          else 0.0,
        ]
        confidence = round(sum(layer_signals) / max(len(layer_signals), 1), 4)
        confidence = max(confidence, 0.50)

        return FinalVerdict(
            decision          = decision,
            risk_level        = risk_level,
            composite_score   = composite_score,
            confidence        = confidence,
            primary_reason    = primary_reason,
            contributing_factors = contributing,
            smoothllm_flag    = smooth_flag,
            neural_flag       = neural_warn_signal,
            scallop_flag      = result.scallop_result.risk_label if result.scallop_result else "SAFE",
            z3_flag           = z3_verdict,
            kg_flag           = result.kg_result.risk_label if result.kg_result else "SAFE",
        )


    def get_session_stats(self) -> dict:

        if not self._session_history:
            return {"turns": 0, "blocked": 0, "warned": 0, "allowed": 0, "block_rate": 0.0}

        blocked = sum(1 for r in self._session_history if r.verdict and r.verdict.decision == "BLOCK")
        warned  = sum(1 for r in self._session_history if r.verdict and r.verdict.decision == "WARN")
        allowed = sum(1 for r in self._session_history if r.verdict and r.verdict.decision == "ALLOW")
        total   = len(self._session_history)

        avg_composite = sum(
            r.verdict.composite_score for r in self._session_history if r.verdict
        ) / total

        top_cats: dict[str, int] = {}
        for r in self._session_history:
            if r.neural_result and r.neural_result.is_adversarial:
                cat = r.neural_result.top_threat_category
                top_cats[cat] = top_cats.get(cat, 0) + 1

        return {
            "turns":           total,
            "blocked":         blocked,
            "warned":          warned,
            "allowed":         allowed,
            "block_rate":      round(blocked / total, 3),
            "warn_rate":       round(warned / total, 3),
            "avg_threat_score": round(avg_composite, 4),
            "top_threat_categories": sorted(top_cats.items(), key=lambda x: x[1], reverse=True)[:3],
            "multi_turn_attacks_detected": sum(
                1 for r in self._session_history
                if r.scallop_result and r.scallop_result.multi_turn_attack_detected
            ),
        }
