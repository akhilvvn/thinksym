from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    prompt: str = Field(min_length=1)
    session_id: str = Field(default="default")
    update_session: bool = True
    device: str = "cuda"


class LayerScore(BaseModel):
    name: str
    score: float
    label: str = ""


class VerdictPayload(BaseModel):
    decision: str
    risk_level: str
    composite_score: float
    confidence: float
    primary_reason: str
    contributing_factors: list[str]
    flags: dict[str, str | bool]


class AnalyzeResponse(BaseModel):
    session_id: str
    turn_number: int
    prompt: str
    verdict: VerdictPayload
    top_threat: str
    timings_ms: dict[str, float]
    total_time_ms: float
    layers: list[LayerScore]
    errors: list[str]


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    session_id: str = Field(default="default")
    update_session: bool = True
    security_enabled: bool = True
    device: str = "cuda"
    hf_token: str = ""
    model_key: str = "Qwen2.5 7B Instruct"
    max_new_tokens: int = 160
    ollama_model: str = "llama3.2:3b"


class ChatResponse(BaseModel):
    blocked: bool
    protection_enabled: bool = True
    analysis: AnalyzeResponse
    llm_text: str = ""
    llm_model: str = ""
    llm_backend: str = ""
    llm_latency_ms: float = 0.0
    llm_error: str = ""


class DemoCompareResponse(BaseModel):
    prompt: str
    baseline: ChatResponse
    protected: ChatResponse


class SessionStatsResponse(BaseModel):
    session_id: str
    stats: dict


class SessionHistoryItem(BaseModel):
    turn_number: int
    prompt: str
    decision: str
    risk_level: str
    composite_score: float
    top_threat: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    history: list[SessionHistoryItem]


class BenchmarkResponse(BaseModel):
    total_adversarial: int
    blocked_adversarial: int
    warned_adversarial: int
    allowed_adversarial: int
    total_benign: int
    blocked_benign: int
    warned_benign: int
    allowed_benign: int
    adversarial_block_rate: float
    benign_false_positive_rate: float
