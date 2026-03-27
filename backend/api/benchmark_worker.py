import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.orchestrator import ThinkSymOrchestrator

DATA_PATH = ROOT / "data" / "sample_prompts.json"


def run(kind: str, start: int, end: int, device: str) -> dict:
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    items = data.get(kind, [])[start:end]
    orchestrator = ThinkSymOrchestrator(device=device, session_id=f"bench-{kind}-{start}-{end}")
    orchestrator.reset_session()

    blocked = 0
    warned = 0
    allowed = 0

    for item in items:
        result = orchestrator.analyze(item.get("prompt", ""), update_session=False)
        decision = result.verdict.decision if result.verdict else "ALLOW"
        if decision == "BLOCK":
            blocked += 1
        elif decision == "WARN":
            warned += 1
        else:
            allowed += 1

    return {
        "kind": kind,
        "start": start,
        "end": end,
        "processed": len(items),
        "blocked": blocked,
        "warned": warned,
        "allowed": allowed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["adversarial", "benign"], required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    payload = run(args.kind, args.start, args.end, args.device)
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())