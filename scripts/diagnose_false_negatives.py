#!/usr/bin/env python3
"""Diagnose which adversarial prompts are NOT being blocked and why."""

import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent


def run_single_prompt(prompt: str, device: str = "cuda") -> dict:
    """Analyze a single prompt and return full results."""
    cmd = [
        sys.executable,
        "-c",
        "\n".join([
            "import json, sys; sys.path.insert(0, str(__import__('pathlib').Path.cwd()))",
            "from pipeline.orchestrator import ThinkSymOrchestrator",
            f"orchestrator = ThinkSymOrchestrator(device='{device}')",
            f"result = orchestrator.analyze({repr(prompt)})",
            "output = {",
            "    'decision': result.verdict.decision,",
            "    'risk_level': result.verdict.risk_level,",
            "    'composite_score': result.verdict.composite_score,",
            "    'top_threat': result.top_threat,",
            "    'neural_score': result.neural_result.composite_threat_score if result.neural_result else 0.0,",
            "    'scallop_risk': result.scallop_result.symbolic_risk_score if result.scallop_result else 0.0,",
            "    'z3_verdict': result.z3_result.formal_verdict if result.z3_result else 'N/A',",
            "    'kg_risk': result.kg_result.kg_risk_score if result.kg_result else 0.0,",
            "    'neural_scores': {cat: ts.combined_score for cat, ts in (result.neural_result.threat_scores.items() if result.neural_result else [])}",
            "}",
            "print(json.dumps(output))",
        ])
    ]
    
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return json.loads(result.stdout.strip())
        else:
            return {"error": result.stderr}
    except Exception as e:
        return {"error": str(e)}


def main():
    # Load adversarial prompts (tier 1 = first 40)
    with open(ROOT / "data" / "sample_prompts.json") as f:
        data = json.load(f)
    
    adversarial = data["adversarial"][:40]
    
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    
    print(f"Analyzing {len(adversarial)} adversarial prompts...")
    print("=" * 100)
    
    false_negatives = []
    threat_category_stats = defaultdict(lambda: {"total": 0, "blocked": 0, "allowed": 0})
    
    for idx, item in enumerate(adversarial, 1):
        prompt = item["prompt"]
        threat_cat = item.get("attack_type", "unknown")
        
        analysis = run_single_prompt(prompt, device)
        
        if "error" in analysis:
            print(f"[{idx:02d}] ERROR: {analysis['error'][:80]}")
            continue
        
        decision = analysis["decision"]
        blocked = decision == "BLOCK"
        
        threat_category_stats[threat_cat]["total"] += 1
        if blocked:
            threat_category_stats[threat_cat]["blocked"] += 1
        else:
            threat_category_stats[threat_cat]["allowed"] += 1
        
        status = "✓ BLOCKED" if blocked else "✗ ALLOWED (FALSE NEGATIVE)"
        print(f"[{idx:02d}] {status} | {threat_cat:25s} | Neural={analysis['neural_score']:.3f} | "
              f"Scallop={analysis['scallop_risk']:.3f} | Composite={analysis['composite_score']:.3f}")
        
        if not blocked:
            false_negatives.append({
                "idx": idx,
                "threat_category": threat_cat,
                "prompt_preview": prompt[:70],
                "decision": decision,
                "neural_score": analysis["neural_score"],
                "scallop_risk": analysis["scallop_risk"],
                "composite_score": analysis["composite_score"],
                "z3_verdict": analysis["z3_verdict"],
                "kg_risk": analysis["kg_risk"],
                "top_neural_threat": analysis["top_threat"],
                "neural_scores": analysis.get("neural_scores", {}),
            })
    
    print("\n" + "=" * 100)
    print("SUMMARY BY THREAT CATEGORY:")
    print("-" * 100)
    for cat in sorted(threat_category_stats.keys()):
        stats = threat_category_stats[cat]
        block_rate = stats["blocked"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{cat:30s} | Total: {stats['total']:2d} | Blocked: {stats['blocked']:2d} | "
              f"Allowed: {stats['allowed']:2d} | Block Rate: {block_rate:5.1f}%")
    
    print("\n" + "=" * 100)
    print(f"FALSE NEGATIVES ({len(false_negatives)}/40 = {len(false_negatives)/40*100:.1f}%):")
    print("-" * 100)
    
    for fn in false_negatives:
        print(f"\n[{fn['idx']:02d}] {fn['threat_category']}")
        print(f"    Prompt: {fn['prompt_preview']}")
        print(f"    Neural Score: {fn['neural_score']:.4f}")
        print(f"    Scallop Risk: {fn['scallop_risk']:.4f}")
        print(f"    Composite Score: {fn['composite_score']:.4f}")
        print(f"    Top Neural Threat: {fn['top_neural_threat']}")
        print(f"    Z3 Verdict: {fn['z3_verdict']}")
        print(f"    KG Risk: {fn['kg_risk']:.4f}")
        print(f"    All Neural Scores: {', '.join(f'{k}={v:.3f}' for k, v in sorted(fn['neural_scores'].items(), key=lambda x: -x[1])[:3])}")
    
    # Save detailed report
    report = {
        "summary": {
            "total_adversarial": len(adversarial),
            "false_negatives": len(false_negatives),
            "block_rate": (len(adversarial) - len(false_negatives)) / len(adversarial),
        },
        "category_stats": dict(threat_category_stats),
        "false_negatives": false_negatives,
    }
    
    report_path = ROOT / "diagnose_false_negatives.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
