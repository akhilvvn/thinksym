#!/usr/bin/env python3
"""Quick diagnostic to check neural scores on first few adversarial prompts."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.orchestrator import ThinkSymOrchestrator


def main():
    with open(ROOT / "data" / "sample_prompts.json") as f:
        data = json.load(f)
    
    adversarial = data["adversarial"][:40]
    
    orchestrator = ThinkSymOrchestrator(device="cuda")
    
    print("Checking first 12 adversarial prompts:\n")
    
    for idx, item in enumerate(adversarial[:12], 1):
        prompt = item["prompt"]
        threat_type = item.get("attack_type", "unknown")
        
        result = orchestrator.analyze(prompt, update_session=False)
        
        if result.neural_result:
            neural_score = result.neural_result.composite_threat_score
            top_category = result.neural_result.top_threat_category
            decision = result.verdict.decision
            status = "✓" if decision == "BLOCK" else "✗"
            
            print(f"{status} [{idx:02d}] {threat_type:20s} | Neural: {neural_score:.3f} | "
                  f"TopCat: {top_category:20s} | Decision: {decision}")
        else:
            print(f"✗ [{idx:02d}] ERROR: Neural result is None")


if __name__ == "__main__":
    main()
