import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.main import benchmark

result = benchmark(device="cuda")
payload = {
    "total_adversarial": result.total_adversarial,
    "blocked_adversarial": result.blocked_adversarial,
    "warned_adversarial": result.warned_adversarial,
    "allowed_adversarial": result.allowed_adversarial,
    "adversarial_block_rate": result.adversarial_block_rate,
    "total_benign": result.total_benign,
    "blocked_benign": result.blocked_benign,
    "warned_benign": result.warned_benign,
    "allowed_benign": result.allowed_benign,
    "benign_false_positive_rate": result.benign_false_positive_rate,
}

out = ROOT / "bench_counts.json"
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(out)