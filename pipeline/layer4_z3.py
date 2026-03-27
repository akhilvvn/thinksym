

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import z3

from config import Z3_POLICIES

logger = logging.getLogger(__name__)
_Z3_VERIFY_LOCK = threading.Lock()


@dataclass
class PolicyCheckResult:

    policy_id:       str
    policy_name:     str
    condition_str:   str
    variable_value:  float
    threshold:       float
    triggered:       bool
    smt_result:      str
    proof_obligation: str


@dataclass
class Z3VerifierResult:


    evidence_scores:       dict[str, float] = field(default_factory=dict)
    proposed_action:       str = "ALLOW"


    policy_results:        list[PolicyCheckResult] = field(default_factory=list)


    total_policies:        int   = 0
    triggered_policies:    int   = 0
    satisfied_policies:    int   = 0
    unsatisfied_policies:  int   = 0


    formal_verdict:        str   = "ALLOW"
    formal_proof:          str   = ""
    is_formally_safe:      bool  = True
    verification_score:    float = 0.0


    z3_version:            str   = ""
    solver_timeout_ms:     int   = 2000
    processing_time_ms:    float = 0.0

    @property
    def blocking_policies(self) -> list[str]:
        return [r.policy_name for r in self.policy_results if r.triggered]


def _build_evidence(
    neural_scores:   dict[str, float],
    scallop_scores:  dict,
) -> dict[str, float]:

    evidence = {

        "jailbreak_score":           neural_scores.get("jailbreak",           0.0),
        "pii_extraction_score":      neural_scores.get("pii_extraction",      0.0),
        "system_prompt_leak_score":  neural_scores.get("system_prompt_leak",  0.0),
        "code_exploit_score":        neural_scores.get("code_exploit",        0.0),
        "prompt_injection_score":    neural_scores.get("prompt_injection",    0.0),
        "harmful_content_score":     neural_scores.get("harmful_content",     0.0),
        "social_engineering_score":  neural_scores.get("social_engineering",  0.0),
        "misinformation_score":      neural_scores.get("misinformation",      0.0),
        "resource_abuse_score":      neural_scores.get("resource_abuse",      0.0),

        "temporal_escalation_score": getattr(scallop_scores, "temporal_escalation_score", 0.0),
        "symbolic_risk_score":       getattr(scallop_scores, "symbolic_risk_score",       0.0),

        "composite_score":           max(
            neural_scores.get("jailbreak",           0.0),
            neural_scores.get("harmful_content",     0.0),
            neural_scores.get("code_exploit",        0.0),
            getattr(scallop_scores, "symbolic_risk_score", 0.0),
        ),
    }
    return {k: round(v, 4) for k, v in evidence.items()}


def _parse_condition(
    condition_str: str,
    z3_vars: dict[str, z3.RealVal],
    evidence: dict[str, float],
) -> tuple[z3.BoolRef, float, float]:

    parts = condition_str.strip().split()

    if len(parts) != 3:
        raise ValueError(f"Cannot parse condition: {condition_str}")

    var_name, op, threshold_str = parts
    threshold = float(threshold_str)
    var = z3_vars.get(var_name)
    if var is None:
        raise ValueError(f"Unknown variable in Z3 context: {var_name}")

    if op == ">=":
        expr = var >= threshold
    elif op == ">":
        expr = var >  threshold
    elif op == "<=":
        expr = var <= threshold
    elif op == "<":
        expr = var <  threshold
    elif op == "==":
        expr = var == threshold
    else:
        raise ValueError(f"Unsupported operator: {op}")


    actual_value = evidence.get(var_name, 0.0)
    return expr, actual_value, threshold


class Z3VerifierLayer:


    def __init__(self, timeout_ms: int = 2000):
        self.timeout_ms = timeout_ms
        z3.set_param("parallel.enable", False)
        self.z3_version = z3.get_version_string()
        logger.info("Z3 SMT Solver version: %s", self.z3_version)

    def verify(
        self,
        neural_scores:  dict[str, float],
        scallop_result: object,
    ) -> Z3VerifierResult:

        t0 = time.perf_counter()


        evidence = _build_evidence(neural_scores, scallop_result)

        result = Z3VerifierResult(
            evidence_scores     = evidence,
            proposed_action     = "ALLOW",
            total_policies      = len(Z3_POLICIES),
            z3_version          = self.z3_version,
            solver_timeout_ms   = self.timeout_ms,
        )


        proof_lines: list[str] = ["Z3 Formal Verification Trace", "=" * 50, ""]
        proof_lines.append(f"Evidence Context:")
        for k, v in evidence.items():
            proof_lines.append(f"  {k} = {v:.4f}")
        proof_lines.append("")
        proof_lines.append("Policy Evaluation:")
        proof_lines.append("-" * 40)

        for policy in Z3_POLICIES:
            try:
                with _Z3_VERIFY_LOCK:
                    solver = z3.Solver()
                    solver.set("timeout", self.timeout_ms)

                    z3_vars: dict[str, z3.RealRef] = {}
                    for var_name, value in evidence.items():
                        z3_var = z3.Real(var_name)
                        z3_vars[var_name] = z3_var
                        solver.add(z3_var == z3.RealVal(str(round(value, 6))))

                    cond_expr, var_val, threshold = _parse_condition(
                        policy["condition"], z3_vars, evidence
                    )

                    solver.add(cond_expr)
                    smt_result_raw = solver.check()

                if smt_result_raw == z3.sat:
                    smt_result_str = "SAT"
                    triggered      = True
                elif smt_result_raw == z3.unsat:
                    smt_result_str = "UNSAT"
                    triggered      = False
                else:
                    smt_result_str = "UNKNOWN"
                    triggered      = False

            except Exception as exc:
                logger.warning("Z3 policy parse error for %s: %s", policy["id"], exc)

                var_name_in_cond = policy["condition"].split()[0]
                var_val = evidence.get(var_name_in_cond, 0.0)
                threshold = float(policy["condition"].split()[2])
                triggered = var_val >= threshold
                smt_result_str = "SAT" if triggered else "UNSAT"


            proof_obligation = (
                f"∃ assignment : evidence ⊨ {policy['condition']}\n"
                f"  ↳ {policy['condition'].split()[0]} = {evidence.get(policy['condition'].split()[0], 0.0):.4f} "
                f"{'≥' if triggered else '<'} {policy['condition'].split()[2]}"
            )

            policy_check = PolicyCheckResult(
                policy_id        = policy["id"],
                policy_name      = policy["name"],
                condition_str    = policy["condition"],
                variable_value   = evidence.get(policy["condition"].split()[0], 0.0),
                threshold        = float(policy["condition"].split()[2]),
                triggered        = triggered,
                smt_result       = smt_result_str,
                proof_obligation = proof_obligation,
            )
            result.policy_results.append(policy_check)

            status_icon = "🔴 TRIGGERED" if triggered else "✅ CLEAR"
            proof_lines.append(
                f"  [{policy['id']}] {policy['name']}\n"
                f"       Condition : {policy['condition']}\n"
                f"       Value     : {evidence.get(policy['condition'].split()[0], 0.0):.4f}\n"
                f"       Z3 Result : {smt_result_str} — {status_icon}\n"
            )


        triggered_count = sum(1 for r in result.policy_results if r.triggered)
        result.triggered_policies   = triggered_count
        result.satisfied_policies   = result.total_policies - triggered_count
        result.unsatisfied_policies = triggered_count


        critical_policies = {"P-001", "P-004", "P-005", "P-006"}
        critical_triggered = sum(
            1 for r in result.policy_results
            if r.triggered and r.policy_id in critical_policies
        )
        result.verification_score = round(
            0.60 * (triggered_count / max(result.total_policies, 1)) +
            0.40 * (critical_triggered / max(len(critical_policies), 1)),
            4
        )


        CRITICAL_POLICY_IDS = {"P-001", "P-004"}
        critical_triggered = any(
            r.triggered for r in result.policy_results
            if r.policy_id in CRITICAL_POLICY_IDS
        )
        non_critical_triggered = sum(
            1 for r in result.policy_results
            if r.triggered and r.policy_id not in CRITICAL_POLICY_IDS
        )
        blocking_conditions = (
            critical_triggered or
            non_critical_triggered >= 2 or
            evidence.get("composite_score", 0.0) >= 0.70 or
            evidence.get("temporal_escalation_score", 0.0) >= 0.55
        )
        result.formal_verdict  = "BLOCK" if blocking_conditions else "ALLOW"
        result.is_formally_safe = result.formal_verdict == "ALLOW"


        proof_lines.append("=" * 50)
        proof_lines.append(f"\nFORMAL VERDICT: {result.formal_verdict}")
        proof_lines.append(f"Triggered Policies: {triggered_count}/{result.total_policies}")
        proof_lines.append(f"Verification Score: {result.verification_score:.4f}")
        if not result.is_formally_safe:
            proof_lines.append("\nBlocking Policies:")
            for r in result.policy_results:
                if r.triggered:
                    proof_lines.append(f"  • [{r.policy_id}] {r.policy_name}")
        result.formal_proof = "\n".join(proof_lines)

        result.processing_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "Z3 Verifier: verdict=%s, triggered=%d/%d, score=%.3f [%.1fms]",
            result.formal_verdict, triggered_count, result.total_policies,
            result.verification_score, result.processing_time_ms,
        )
        return result
