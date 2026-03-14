from backend.api.schemas import AnalyzeResponse, LayerScore, VerdictPayload
from pipeline.orchestrator import AnalysisResult


def to_analyze_response(result: AnalysisResult) -> AnalyzeResponse:
    verdict = result.verdict
    decision = verdict.decision if verdict else "ALLOW"
    risk_level = verdict.risk_level if verdict else "SAFE"
    primary_reason = verdict.primary_reason if verdict else ""
    contributing_factors = verdict.contributing_factors if verdict else []

    if decision == "BLOCK" and risk_level in {"LOW", "SAFE"}:
        decision = "WARN"
        primary_reason = (
            f"Threat signals present but low aggregate risk (composite={verdict.composite_score:.3f}); "
            "downgraded from hard block."
            if verdict
            else "Threat signals present but low aggregate risk; downgraded from hard block."
        )
        if "Low-risk downgrade" not in contributing_factors:
            contributing_factors = [*contributing_factors, "Low-risk downgrade"]

    verdict_payload = VerdictPayload(
        decision=decision,
        risk_level=risk_level,
        composite_score=verdict.composite_score if verdict else 0.0,
        confidence=verdict.confidence if verdict else 0.0,
        primary_reason=primary_reason,
        contributing_factors=contributing_factors,
        flags={
            "smoothllm": verdict.smoothllm_flag if verdict else "PROCEED",
            "neural": verdict.neural_flag if verdict else False,
            "scallop": verdict.scallop_flag if verdict else "SAFE",
            "z3": verdict.z3_flag if verdict else "ALLOW",
            "kg": verdict.kg_flag if verdict else "SAFE",
        },
    )

    show_security_threat = decision in {"WARN", "BLOCK"} or bool(verdict_payload.flags.get("neural", False))
    displayed_top_threat = result.top_threat if show_security_threat else "none"
    neural_label = getattr(result.neural_result, "top_threat_category", "none") if result.neural_result else "none"
    if not show_security_threat:
        neural_label = "none"

    layers = [
        LayerScore(name="smoothllm", score=getattr(result.smooth_result, "structural_fragility", 0.0), label=getattr(result.smooth_result, "recommendation", "PROCEED") if result.smooth_result else "PROCEED"),
        LayerScore(name="neural", score=getattr(result.neural_result, "composite_threat_score", 0.0), label=neural_label),
        LayerScore(name="scallop", score=getattr(result.scallop_result, "symbolic_risk_score", 0.0), label=getattr(result.scallop_result, "risk_label", "SAFE") if result.scallop_result else "SAFE"),
        LayerScore(name="z3", score=getattr(result.z3_result, "verification_score", 0.0), label=getattr(result.z3_result, "formal_verdict", "ALLOW") if result.z3_result else "ALLOW"),
        LayerScore(name="knowledge_graph", score=getattr(result.kg_result, "kg_risk_score", 0.0), label=getattr(result.kg_result, "risk_label", "SAFE") if result.kg_result else "SAFE"),
    ]

    return AnalyzeResponse(
        session_id=result.session_id,
        turn_number=result.turn_number,
        prompt=result.prompt,
        verdict=verdict_payload,
        top_threat=displayed_top_threat,
        timings_ms=result.layer_timings_ms,
        total_time_ms=result.total_time_ms,
        layers=layers,
        errors=result.errors,
    )
