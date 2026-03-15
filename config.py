

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

MODERNBERT_MODEL  = "answerdotai/ModernBERT-base"
DEVICE            = "cuda"
USE_FINETUNED_MODERNBERT = True
FINETUNED_MODERNBERT_DIR = ROOT_DIR / "models" / "modernbert-defense"


LLAMA_GUARD_MODEL = "llama-guard3"
OLLAMA_BASE_URL   = "http://localhost:11434"
LLAMA_GUARD_TIMEOUT = 30


NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "thinksym@123"
NEO4J_DATABASE = "thinksym"
NEO4J_TIMEOUT  = 5


NEURAL_THREAT_THRESHOLD     = 0.55
SCALLOP_RISK_THRESHOLD      = 0.55
Z3_POLICY_STRICTNESS        = "high"
KG_MATCH_THRESHOLD          = 0.45


SCALLOP_SIGMOID_K           = 8.0
SCALLOP_ESCALATION_DELTA    = 0.08


SMOOTHLLM_N_PERTURBATIONS   = 12
SMOOTHLLM_CHAR_BUDGET       = 0.10
PERTURBATION_MODES          = ["insert", "delete", "swap", "substitute"]


MAX_SESSION_TURNS           = 20
ESCALATION_WINDOW           = 5


SCALLOP_PROVENANCE          = "topkproofs"
SCALLOP_TOP_K               = 3


THREAT_CATEGORIES = [
    "prompt_injection",
    "jailbreak",
    "pii_extraction",
    "system_prompt_leak",
    "harmful_content",
    "code_exploit",
    "social_engineering",
    "misinformation",
    "resource_abuse",
    "data_exfiltration",
]


RISK_LEVELS = {
    "CRITICAL": {"min": 0.80, "color": "#f38ba8", "emoji": "🔴"},
    "HIGH":     {"min": 0.60, "color": "#fab387", "emoji": "🟠"},
    "MEDIUM":   {"min": 0.40, "color": "#f9e2af", "emoji": "🟡"},
    "LOW":      {"min": 0.20, "color": "#a6e3a1", "emoji": "🟢"},
    "SAFE":     {"min": 0.00, "color": "#89dceb", "emoji": "✅"},
}


APP_TITLE       = "ThinkSym"
APP_SUBTITLE    = "Neurosymbolic Defense Framework for LLM Security"
APP_VERSION     = "2.0.4"
ACCENT_COLOR    = "#89b4fa"


Z3_POLICIES = [
    {
        "id": "P-001",
        "name": "No Jailbreak Tolerated",
        "description": "Block any input classified as a jailbreak attempt regardless of confidence",
        "condition": "jailbreak_score >= 0.55",
    },
    {
        "id": "P-002",
        "name": "PII Extraction Guard",
        "description": "Block attempts to extract personally identifiable information",
        "condition": "pii_extraction_score >= 0.55",
    },
    {
        "id": "P-003",
        "name": "System Prompt Confidentiality",
        "description": "Prevent system prompt leakage attempts",
        "condition": "system_prompt_leak_score >= 0.55",
    },
    {
        "id": "P-004",
        "name": "Code Exploit Prevention",
        "description": "Block requests for malicious code generation",
        "condition": "code_exploit_score >= 0.55",
    },
    {
        "id": "P-005",
        "name": "Multi-turn Escalation",
        "description": "Block when session escalation score is detected",
        "condition": "temporal_escalation_score >= 0.55",
    },
    {
        "id": "P-006",
        "name": "Composite Threat Score",
        "description": "Block when aggregate threat score exceeds safe threshold",
        "condition": "composite_score >= 0.70",
    },
]
