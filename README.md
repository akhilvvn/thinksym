# ThinkSym

**NeuroSymbolic LLM Security Framework**

ThinkSym is a multi-layer adversarial prompt detection and policy enforcement system for Large Language Models. It combines neural threat classification, probabilistic symbolic reasoning, formal policy verification, and knowledge graph corroboration to make LLM interactions safe — without compromising benign user experience.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Layers](#pipeline-layers)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Running Individual Services](#running-individual-services)
- [Benchmark](#benchmark)
- [Threat Categories](#threat-categories)
- [Dataset & Knowledge Sources](#dataset--knowledge-sources)
- [Legacy Streamlit UI](#legacy-streamlit-ui)

---

## Overview

Most LLM security solutions rely on a single layer — a neural classifier or a static blocklist — which can be bypassed by adversarial prompt engineering or causes excessive false positives for benign users. ThinkSym takes a different approach: a **five-layer neuro-symbolic pipeline** where each layer contributes a different kind of evidence, and a calibrated orchestrator fuses the signals into a final verdict.

**Current benchmark corpus:**
- 1000 prompts total
- 200 adversarial prompts across 10 threat categories
- 800 benign prompts across diverse real-world assistant tasks

**Observed interim snapshot (earlier small-set run):**
- ~95%+ adversarial block rate
- 0% benign false positive rate
- Full per-layer observability and explainability for every decision

---

## Architecture

```
User Prompt
    │
    ▼
┌─────────────────────────────────────────────┐
│               FastAPI Backend               │
│         (REST API · Session Mgmt)           │
└────────────────────┬────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │   5-Layer Pipeline    │
         │                       │
         │  L1 · SmoothLLM       │  ← Perturbation robustness
         │  L2 · Neural          │  ← HF Transformers + Llama Guard 3
         │  L3 · Scallop         │  ← Probabilistic symbolic reasoning
         │  L4 · Z3              │  ← Formal policy verification (SMT)
         │  L5 · Knowledge Graph │  ← Neo4j / NetworkX corroboration
         │                       │
         │  Orchestrator         │  ← Fuses all signals → ALLOW/WARN/BLOCK
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   Verdict Engine      │
         │  ALLOW / WARN / BLOCK │
         │  + Risk · Score ·     │
         │    Threat · Reason    │
         └───────────┬───────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
    ┌─────▼─────┐         ┌─────▼──────┐
    │ LLM Resp  │         │  Refusal   │
    │ (ALLOW)   │         │  (BLOCK/   │
    └─────┬─────┘         │   WARN)    │
          │               └────────────┘
          ▼
┌─────────────────────────────────────────────┐
│           Next.js Dashboard                 │
│  Secure Chat · Layer Observability          │
│  Security Timeline · Benchmark Panel        │
└─────────────────────────────────────────────┘
```

---

## Pipeline Layers

### Layer 1 — SmoothLLM (Robustness)
Applies perturbation-based robustness checks to detect prompts whose adversarial intent survives input noise. Acts as an early signal for fragile/adversarial inputs.

### Layer 2 — Neural Classifier
- Primary: **Fine-tuned ModernBERT** via Hugging Face Transformers, calibrated against a benign class probability baseline to reduce false positives.
- Auxiliary: **Llama Guard 3** (via Ollama) as an additional safety signal — its per-category confidence scores are fused with ModernBERT output.
- Classifies prompts into threat categories with calibrated per-category scores.

### Layer 3 — Scallop Symbolic Reasoner
Implements probabilistic Datalog-style reasoning (Scallopy library, with Python fallback):
- Computes **compound multi-category risk** using noisy-or probability fusion.
- Detects **temporal escalation** — rising threat scores across consecutive session turns.
- Matches known **cross-category attack chains** (e.g., social engineering → prompt injection → code exploit).
- Outputs a `symbolic_risk_score` with labels: SAFE / LOW / MEDIUM / HIGH / CRITICAL.

### Layer 4 — Z3 Formal Verifier
Uses Microsoft's **Z3 SMT solver** to evaluate formal policy constraints against the aggregated evidence:
- Each policy is expressed as a threshold condition (e.g., `jailbreak_score >= 0.55`).
- Z3 solves whether the constraint is satisfiable given the evidence.
- SAT = policy triggered; UNSAT = policy clear.
- Produces a **formal proof trace** and verification score.

### Layer 5 — Knowledge Graph
Two-backend architecture:
- **Neo4j** (primary, when running locally at `bolt://127.0.0.1:7687`) using Cypher queries for threat node matching.
- **NetworkX** (in-memory fallback) built from OWASP LLM Top 10 and MITRE ATT&CK for AI JSON sources.
- Matches prompt signals against known attack taxonomy nodes and returns corroboration scores.

### Orchestrator
Fuses all five layer signals using a multi-condition logic tree:
- Corroboration-based blocks for injection/jailbreak/system-prompt-leak categories.
- Ultra-high-confidence single-layer blocks for severe categories (harmful content, code exploit, PII).
- Composite score thresholding as a final backstop.
- Final decision: `ALLOW`, `WARN`, or `BLOCK` with risk level, top threat, and primary reason.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | Python 3.11, FastAPI, Uvicorn |
| Neural Classifier | Hugging Face Transformers (ModernBERT fine-tuned) |
| Safety Signal | Meta Llama Guard 3 (via Ollama) |
| Symbolic Reasoning | Scallop / Scallopy (Python fallback) |
| Formal Verification | Microsoft Z3 SMT Solver |
| Knowledge Graph | Neo4j + Cypher / NetworkX (fallback) |
| LLM Response Backend | Hugging Face Inference API (Qwen2.5, Mistral, Phi-3, Zephyr) / Ollama |
| Frontend | TypeScript, Next.js 14, Tailwind CSS |
| Version Control | Git / GitHub |

---

## Project Structure

```
thinksym/
├── backend/
│   └── api/
│       ├── main.py          # FastAPI routes, chat handler
│       ├── schemas.py       # Pydantic request/response models
│       ├── serialization.py # Pipeline result → API response
│       └── session_store.py # Session state management
├── pipeline/
│   ├── layer1_smooth_llm.py
│   ├── layer2_neural.py     # ModernBERT + Llama Guard 3
│   ├── layer3_scallop.py    # Probabilistic symbolic reasoning
│   ├── layer4_z3.py         # Formal SMT policy verification
│   ├── layer5_knowledge_graph.py  # Neo4j / NetworkX
│   ├── llm_backend.py       # HF API + Ollama routing
│   └── orchestrator.py      # Multi-layer signal fusion
├── frontend/
│   ├── app/page.tsx         # Main dashboard UI
│   ├── components/          # MetricCard, LayerBar, VerdictPill, BenchmarkPanel
│   └── lib/                 # API client, TypeScript types
├── data/
│   ├── owasp_llm_top10.json
│   ├── mitre_attack_llm.json
│   └── sample_prompts.json
├── models/                  # Fine-tuned ModernBERT weights
├── scripts/
│   └── train_modernbert.py  # Fine-tuning script
├── config.py                # All thresholds and constants
├── dev_start.py             # Launches both backend + frontend
└── app.py                   # Legacy Streamlit UI (backwards compat)
```

---

## Quick Start

**Prerequisites:** Python 3.11+, Node.js 18+

```bash
# 1. Clone the repo
git clone https://github.com/akhilvvn/thinksym.git
cd thinksym

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install frontend dependencies
cd frontend && npm install && cd ..

# 4. Set up environment variables (see below)
cp .env.example .env

# 5. Launch both services
python dev_start.py
```

- **Backend:** `http://127.0.0.1:8002`
- **Frontend:** `http://localhost:3000`

---

## Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```env
# Hugging Face token for inference API (primary LLM backend)
HF_TOKEN=hf_your_token_here

# Neo4j connection (optional — NetworkX used as fallback)
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=thinksym
```

`HF_TOKEN` is the primary requirement. Without it, the system falls back to Ollama for LLM responses.

---

## Running Individual Services

**Backend only:**
```bash
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8002 --reload
```

**Frontend only:**
```bash
cd frontend
npm run dev
```

**Ollama (optional local LLM fallback):**
```bash
ollama serve
ollama pull llama3.2:3b
ollama pull llama-guard3
```

---

## Benchmark

The built-in benchmark evaluates the pipeline against a curated adversarial and benign prompt set:

- Click **Benchmark** in the dashboard header.
- Results show adversarial block rate, benign false positive rate, and per-verdict breakdowns.
- Target: ≥90% adversarial block rate, ≤5% false positive rate.

To run via API directly:
```bash
curl http://127.0.0.1:8002/api/v1/benchmark
```

---

## Threat Categories

ThinkSym detects and classifies the following threat categories:

| Category | Description |
|---|---|
| Prompt Injection | Hijacking LLM instructions via adversarial input |
| Jailbreak | Bypassing safety constraints via persona/role manipulation |
| Harmful Content | Requests for violence, self-harm, or illegal activity |
| Code Exploit | Generating malware, exploits, or dangerous scripts |
| PII Extraction | Extracting personally identifiable or sensitive data |
| System Prompt Leak | Extracting confidential system instructions |
| Social Engineering | Manipulating the model to assist deceptive behavior |
| Data Exfiltration | Inducing unauthorized data disclosure |
| Resource Abuse | Denial-of-service or compute abuse patterns |
| Misinformation | Generating deliberately false or misleading content |

---

## Dataset & Knowledge Sources

- **Adversarial benchmark set** — 200 curated adversarial prompts across 10 threat categories
- **Benign validation set** — 800 normal user prompts across diverse real-world tasks
- **OWASP LLM Top 10 (2025)** — threat taxonomy and keyword indicators
- **MITRE ATT&CK for AI** — tactic/technique/subtechnique graph
- **HarmBench-inspired signatures** — indicator phrases and attack patterns

---

## Legacy Streamlit UI

Kept for backwards compatibility only. Not actively maintained.

```bash
streamlit run app.py
```#   D e c e m b e r   w o r k  
 #   C h a n g e   0  
 #   C h a n g e   1  
 #   C h a n g e   2  
 #   C h a n g e   3  
 #   C h a n g e   4  
 #   C h a n g e   5  
 #   C h a n g e   6  
 #   C h a n g e   7  
 #   D e c e m b e r   u p d a t e s  
 #   J a n u a r y   w o r k  
 #   F e b r u a r y   u p d a t e s  
 #   M a r c h   w o r k  
 #   A p r i l   f i n a l i z a t i o n  
 #   I n i t i a l   s e t u p  
 #   F o u n d a t i o n  
 #   M o d u l e s  
 #   C o n f i g  
 #   D e c e m b e r   f i n i s h  
 #   J a n   s t a r t  
 #   L a y e r   w o r k  
 #   D a t a b a s e  
 #   T e s t i n g  
 #   J a n   e n d  
 