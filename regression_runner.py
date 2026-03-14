import json
import traceback

from pipeline.orchestrator import ThinkSymOrchestrator


def run():
    with open("data/sample_prompts.json", encoding="utf-8") as f:
        data = json.load(f)

    orch = ThinkSymOrchestrator(device="cuda", session_id="regression")

    adv_res = []
    ben_res = []

    for i, item in enumerate(data["adversarial"], 1):
        try:
            result = orch.analyze(item["prompt"], update_session=False)
            adv_res.append((item["id"], result.verdict.decision, result.verdict.composite_score, result.verdict.primary_reason))
            print(f"ADV {i}/{len(data['adversarial'])}: {item['id']} -> {result.verdict.decision} ({result.verdict.composite_score:.3f})", flush=True)
        except Exception as exc:
            adv_res.append((item["id"], "ERROR", 1.0, str(exc)))
            print(f"ADV {i}/{len(data['adversarial'])}: {item['id']} -> ERROR: {exc}", flush=True)

    for i, item in enumerate(data["benign"], 1):
        try:
            result = orch.analyze(item["prompt"], update_session=False)
            ben_res.append((item["id"], result.verdict.decision, result.verdict.composite_score, result.verdict.primary_reason))
            print(f"BEN {i}/{len(data['benign'])}: {item['id']} -> {result.verdict.decision} ({result.verdict.composite_score:.3f})", flush=True)
        except Exception as exc:
            ben_res.append((item["id"], "ERROR", 1.0, str(exc)))
            print(f"BEN {i}/{len(data['benign'])}: {item['id']} -> ERROR: {exc}", flush=True)

    orch.reset_session()
    mt_res = []
    for turn in data["multi_turn_attack_sequence"]["turns"]:
        try:
            result = orch.analyze(turn["prompt"], update_session=True)
            mt_res.append((turn["turn"], result.verdict.decision, result.verdict.composite_score, result.verdict.primary_reason))
            print(f"MT T{turn['turn']} -> {result.verdict.decision} ({result.verdict.composite_score:.3f})", flush=True)
        except Exception as exc:
            mt_res.append((turn["turn"], "ERROR", 1.0, str(exc)))
            print(f"MT T{turn['turn']} -> ERROR: {exc}", flush=True)

    adv_block = sum(1 for _, d, _, _ in adv_res if d == "BLOCK")
    adv_warn = sum(1 for _, d, _, _ in adv_res if d == "WARN")
    adv_allow = sum(1 for _, d, _, _ in adv_res if d == "ALLOW")
    adv_error = sum(1 for _, d, _, _ in adv_res if d == "ERROR")

    ben_allow = sum(1 for _, d, _, _ in ben_res if d == "ALLOW")
    ben_warn = sum(1 for _, d, _, _ in ben_res if d == "WARN")
    ben_block = sum(1 for _, d, _, _ in ben_res if d == "BLOCK")
    ben_error = sum(1 for _, d, _, _ in ben_res if d == "ERROR")

    print("=== REGRESSION SUMMARY ===", flush=True)
    print(f"Adversarial total={len(adv_res)} block={adv_block} warn={adv_warn} allow={adv_allow} error={adv_error} block_rate={adv_block/max(len(adv_res),1):.3f}", flush=True)
    print(f"Benign total={len(ben_res)} allow={ben_allow} warn={ben_warn} block={ben_block} error={ben_error} false_positive_rate={(ben_warn+ben_block)/max(len(ben_res),1):.3f}", flush=True)

    print("--- Adversarial misses (not BLOCK) ---", flush=True)
    for row in adv_res:
        if row[1] != "BLOCK":
            print(row[0], row[1], f"score={row[2]:.3f}", row[3][:140], flush=True)

    print("--- Benign violations (not ALLOW) ---", flush=True)
    for row in ben_res:
        if row[1] != "ALLOW":
            print(row[0], row[1], f"score={row[2]:.3f}", row[3][:140], flush=True)

    print("--- Multi-turn decisions ---", flush=True)
    for row in mt_res:
        print(f"T{row[0]}", row[1], f"score={row[2]:.3f}", row[3][:140], flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        raise
