import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "backend" / "api" / "benchmark_worker.py"
OUT_PATH = ROOT / "bench_tier1.json"


def run_batch(kind: str, start: int, end: int, device: str) -> dict:
    cmd = [
        sys.executable,
        str(WORKER_PATH),
        "--kind",
        kind,
        "--start",
        str(start),
        "--end",
        str(end),
        "--device",
        device,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "worker failed").strip()
        raise RuntimeError(f"batch failed kind={kind} start={start} end={end}: {detail[:300]}")

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"batch returned empty output kind={kind} start={start} end={end}")

    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"batch output parse error kind={kind} start={start} end={end}: {exc}"
        ) from exc


def run_tier1(device: str = "cuda") -> dict:
    # Tier 1: 200 prompts total with 80:20 split.
    adversarial_total = 40
    benign_total = 160
    batch_size = 20

    blocked_adversarial = 0
    warned_adversarial = 0
    allowed_adversarial = 0

    for start in range(0, adversarial_total, batch_size):
        end = min(start + batch_size, adversarial_total)
        batch = run_batch("adversarial", start, end, device)
        blocked_adversarial += int(batch.get("blocked", 0))
        warned_adversarial += int(batch.get("warned", 0))
        allowed_adversarial += int(batch.get("allowed", 0))

    blocked_benign = 0
    warned_benign = 0
    allowed_benign = 0

    for start in range(0, benign_total, batch_size):
        end = min(start + batch_size, benign_total)
        batch = run_batch("benign", start, end, device)
        blocked_benign += int(batch.get("blocked", 0))
        warned_benign += int(batch.get("warned", 0))
        allowed_benign += int(batch.get("allowed", 0))

    return {
        "tier": "tier1",
        "total_adversarial": adversarial_total,
        "blocked_adversarial": blocked_adversarial,
        "warned_adversarial": warned_adversarial,
        "allowed_adversarial": allowed_adversarial,
        "adversarial_block_rate": blocked_adversarial / max(adversarial_total, 1),
        "total_benign": benign_total,
        "blocked_benign": blocked_benign,
        "warned_benign": warned_benign,
        "allowed_benign": allowed_benign,
        "benign_false_positive_rate": (blocked_benign + warned_benign) / max(benign_total, 1),
    }


def main() -> int:
    device = "cuda"
    if len(sys.argv) > 1:
        device = sys.argv[1]

    payload = run_tier1(device=device)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUT_PATH)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
