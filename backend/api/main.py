import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BenchmarkResponse,
    ChatRequest,
    ChatResponse,
    DemoCompareResponse,
    SessionHistoryItem,
    SessionHistoryResponse,
    SessionStatsResponse,
)
from backend.api.serialization import to_analyze_response
from backend.api.session_store import clear_all, get_orchestrator, reset_session
from pipeline.llm_backend import get_llm_response


app = FastAPI(title="ThinkSym API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_prompts.json"
WORKER_PATH = Path(__file__).resolve().parent / "benchmark_worker.py"


def _run_isolated_batch(kind: str, start: int, end: int, device: str) -> dict:
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
    last_error = "worker failed"
    for _ in range(3):
        proc = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
        if proc.returncode != 0:
            last_error = (proc.stderr or proc.stdout or "worker failed").strip()
            continue

        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not lines:
            last_error = "benchmark worker returned empty output"
            continue
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            last_error = f"benchmark worker output parse error: {exc}"
            continue

    raise HTTPException(status_code=500, detail=f"benchmark worker failed: {last_error[:300]}")


def _run_benchmark_isolated(data: dict, device: str) -> BenchmarkResponse:
    adversarial = data.get("adversarial", [])
    benign = data.get("benign", [])

    batch_size = 25

    blocked_adversarial = 0
    warned_adversarial = 0
    allowed_adversarial = 0

    for start in range(0, len(adversarial), batch_size):
        end = min(start + batch_size, len(adversarial))
        batch = _run_isolated_batch("adversarial", start, end, device)
        blocked_adversarial += int(batch.get("blocked", 0))
        warned_adversarial += int(batch.get("warned", 0))
        allowed_adversarial += int(batch.get("allowed", 0))

    blocked_benign = 0
    warned_benign = 0
    allowed_benign = 0

    for start in range(0, len(benign), batch_size):
        end = min(start + batch_size, len(benign))
        batch = _run_isolated_batch("benign", start, end, device)
        blocked_benign += int(batch.get("blocked", 0))
        warned_benign += int(batch.get("warned", 0))
        allowed_benign += int(batch.get("allowed", 0))

    total_adversarial = len(adversarial)
    total_benign = len(benign)

    return BenchmarkResponse(
        total_adversarial=total_adversarial,
        blocked_adversarial=blocked_adversarial,
        warned_adversarial=warned_adversarial,
        allowed_adversarial=allowed_adversarial,
        total_benign=total_benign,
        blocked_benign=blocked_benign,
        warned_benign=warned_benign,
        allowed_benign=allowed_benign,
        adversarial_block_rate=blocked_adversarial / max(total_adversarial, 1),
        benign_false_positive_rate=(blocked_benign + warned_benign) / max(total_benign, 1),
    )


def _normalize_low_risk_block(payload: AnalyzeResponse) -> AnalyzeResponse:
    if payload.verdict.decision == "BLOCK" and payload.verdict.risk_level in {"LOW", "SAFE"}:
        payload.verdict.decision = "WARN"
        payload.verdict.primary_reason = (
            f"Threat signals present but low aggregate risk (composite={payload.verdict.composite_score:.3f}); "
            "downgraded from hard block."
        )
        if "Low-risk downgrade" not in payload.verdict.contributing_factors:
            payload.verdict.contributing_factors.append("Low-risk downgrade")
    return payload


def _execute_chat(
    request: ChatRequest,
    *,
    security_enabled: bool | None = None,
    update_session: bool | None = None,
) -> ChatResponse:
    effective_security = request.security_enabled if security_enabled is None else security_enabled
    effective_update_session = request.update_session if update_session is None else update_session

    orchestrator = get_orchestrator(request.session_id, device=request.device)
    analysis = orchestrator.analyze(request.prompt, update_session=effective_update_session)
    analysis_payload = _normalize_low_risk_block(to_analyze_response(analysis))

    should_block_generation = effective_security and analysis_payload.verdict.decision in {"WARN", "BLOCK"}

    if should_block_generation:
        return ChatResponse(
            blocked=True,
            protection_enabled=effective_security,
            analysis=analysis_payload,
            llm_text="Sorry, I can't help you with that.",
            llm_model="thinksym-guard",
            llm_backend="policy-refusal",
            llm_latency_ms=0.0,
        )

    llm = get_llm_response(
        prompt=request.prompt,
        hf_token=request.hf_token,
        model_key=request.model_key,
        max_new_tokens=request.max_new_tokens,
        ollama_model=request.ollama_model,
    )

    return ChatResponse(
        blocked=False,
        protection_enabled=effective_security,
        analysis=analysis_payload,
        llm_text=llm.text,
        llm_model=llm.model,
        llm_backend=llm.backend,
        llm_latency_ms=llm.latency_ms,
        llm_error=llm.error,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "thinksym-api"}


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    orchestrator = get_orchestrator(request.session_id, device=request.device)
    result = orchestrator.analyze(request.prompt, update_session=request.update_session)
    return _normalize_low_risk_block(to_analyze_response(result))


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return _execute_chat(request)


@app.post("/api/v1/demo/compare", response_model=DemoCompareResponse)
def demo_compare(request: ChatRequest) -> DemoCompareResponse:
    baseline = _execute_chat(request, security_enabled=False, update_session=False)
    protected = _execute_chat(request, security_enabled=True, update_session=False)
    return DemoCompareResponse(prompt=request.prompt, baseline=baseline, protected=protected)


@app.get("/api/v1/sessions/{session_id}/stats", response_model=SessionStatsResponse)
def session_stats(session_id: str) -> SessionStatsResponse:
    orchestrator = get_orchestrator(session_id)
    return SessionStatsResponse(session_id=session_id, stats=orchestrator.get_session_stats())


@app.get("/api/v1/sessions/{session_id}/history", response_model=SessionHistoryResponse)
def session_history(session_id: str) -> SessionHistoryResponse:
    orchestrator = get_orchestrator(session_id)
    history = []
    for item in orchestrator.session_history:
        if item.verdict is None:
            continue
        payload = _normalize_low_risk_block(to_analyze_response(item))
        decision = payload.verdict.decision
        risk_level = payload.verdict.risk_level
        top_threat = payload.top_threat
        if decision == "ALLOW":
            risk_level = "SAFE"
            top_threat = "none"
        history.append(
            SessionHistoryItem(
                turn_number=item.turn_number,
                prompt=item.prompt,
                decision=decision,
                risk_level=risk_level,
                composite_score=payload.verdict.composite_score,
                top_threat=top_threat,
            )
        )
    return SessionHistoryResponse(session_id=session_id, history=history)


@app.post("/api/v1/sessions/{session_id}/reset")
def session_reset(session_id: str) -> dict:
    ok = reset_session(session_id)
    return {"session_id": session_id, "reset": ok}


@app.post("/api/v1/sessions/reset-all")
def reset_all() -> dict:
    clear_all()
    return {"reset": True}


@app.get("/api/v1/benchmark", response_model=BenchmarkResponse)
def benchmark(device: str = "cuda") -> BenchmarkResponse:
    if not DATA_PATH.exists():
        raise HTTPException(status_code=500, detail="sample_prompts.json not found")

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    use_isolated = os.getenv("THINKSYM_BENCHMARK_ISOLATED", "1") != "0"
    if use_isolated:
        return _run_benchmark_isolated(data, device)

    orchestrator = get_orchestrator("benchmark", device=device)
    orchestrator.reset_session()

    adversarial = data.get("adversarial", [])
    benign = data.get("benign", [])

    blocked_adversarial = 0
    warned_adversarial = 0
    allowed_adversarial = 0

    for item in adversarial:
        result = orchestrator.analyze(item.get("prompt", ""), update_session=False)
        decision = result.verdict.decision if result.verdict else "ALLOW"
        if decision == "BLOCK":
            blocked_adversarial += 1
        elif decision == "WARN":
            warned_adversarial += 1
        else:
            allowed_adversarial += 1

    blocked_benign = 0
    warned_benign = 0
    allowed_benign = 0

    for item in benign:
        result = orchestrator.analyze(item.get("prompt", ""), update_session=False)
        decision = result.verdict.decision if result.verdict else "ALLOW"
        if decision == "BLOCK":
            blocked_benign += 1
        elif decision == "WARN":
            warned_benign += 1
        else:
            allowed_benign += 1

    total_adversarial = len(adversarial)
    total_benign = len(benign)

    return BenchmarkResponse(
        total_adversarial=total_adversarial,
        blocked_adversarial=blocked_adversarial,
        warned_adversarial=warned_adversarial,
        allowed_adversarial=allowed_adversarial,
        total_benign=total_benign,
        blocked_benign=blocked_benign,
        warned_benign=warned_benign,
        allowed_benign=allowed_benign,
        adversarial_block_rate=blocked_adversarial / max(total_adversarial, 1),
        benign_false_positive_rate=(blocked_benign + warned_benign) / max(total_benign, 1),
    )
