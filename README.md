# ThinkSym

Default development workflow uses the enterprise console stack:

- FastAPI backend at `http://127.0.0.1:8002`
- Next.js frontend at `http://localhost:3000`

## Quick Start (Default)

From project root:

```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
python dev_start.py
```

Optional Hugging Face fallback token (local only):

```bash
cp .env.example .env
# then set HF_TOKEN in .env
```

## Individual Services

Backend only:

```bash
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8002 --reload
```

Frontend only:

```bash
cd frontend
npm run dev
```

## Legacy Streamlit UI

The Streamlit interface is kept only for backwards compatibility and is no longer the default workflow.

```bash
streamlit run app.py
```