#!/usr/bin/env python3
"""Direct test of orchestrator with keyword detection."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

data_file = ROOT / "data" / "sample_prompts.json"
print(f"Loading from: {data_file}")
assert data_file.exists(), f"File not found: {data_file}"

with open(data_file) as f:
    data = json.load(f)

adversarial_prompts = data["adversarial"][:5]

print("Testing with new keyword detection logic...\n")

from pipeline.orchestrator import ThinkSymOrchestrator

orch = ThinkSymOrchestrator(device="cuda")

for idx, item in enumerate(adversarial_prompts, 1):
    prompt = item["prompt"]
    attack_type = item.get("attack_type", "unknown")
    
    # Test keyword detection directly
    keyword_scores = orch._detect_keyword_threats(prompt)
    print(f"[{idx}] {attack_type}")
    print(f"    Prompt: {prompt[:60]}...")
    print(f"    Keyword scores: {keyword_scores}")
    
    # Now test the full analysis
    result = orch.analyze(prompt, update_session=False)
    print(f"    Decision: {result.verdict.decision}")
    print(f"    Composite Score: {result.verdict.composite_score:.3f}")
    print()
