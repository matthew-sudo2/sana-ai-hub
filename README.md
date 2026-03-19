# Sana All May Label

Local, sequential multi-agent data pipeline (Ollama) + React dashboard.

## Repository layout

```
sana-ai-hub/
  backend/    # Python agents, Ollama orchestration, local persistence
  frontend/   # Vite + React dashboard
  docs/       # submission assets
```

## Frontend (React)

From repo root:

```bash
cd frontend
npm install
npm run dev
```

## Backend (Python)

Create a virtualenv, then from repo root:

```bash
pip install -r backend/requirements.txt
```

Create `backend/.env` from `backend/.env.example` and set at minimum:
- `SPIDER_API_KEY`

### Phase 1 (Scout)

```bash
python backend/agents/scout.py https://example.com --limit 5 --return-format markdown
```

### Phase 2 (Labeler & Artist)

```bash
python backend/agents/labeler.py backend/runs/<phase1_run>/extracted_metadata.json --model qwen2.5-coder:3b
```

### Phase 3 (Artist)

```bash
python backend/agents/artist.py backend/runs/<phase2_run>/cleaned_data.csv --model qwen2.5-coder:3b
```

### Phase 4 (Analyst)

```bash
python backend/agents/analyst.py backend/runs/<phase2_run>/cleaned_data.csv --phase3-dir backend/runs/<phase3_run> --model llama3:8b
```
