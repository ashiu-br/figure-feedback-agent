# BioRender Figure Feedback Agent

AI-powered feedback system for scientific figure analysis with multi-agent architecture using FastAPI (backend), LangGraph for orchestration, and vision-enabled content interpretation. Optional Arize tracing is supported.

## Quickstart

1) Requirements
- Python 3.10+ (Docker optional)

2) Configure environment
- Copy `backend/env_example.txt` to `backend/.env`.
- Set one LLM key: `OPENAI_API_KEY=...` or `OPENROUTER_API_KEY=...`.
- Optional: `ARIZE_SPACE_ID` and `ARIZE_API_KEY` for tracing.
- Optional: `TEST_MODE=1` for development (disables vision analysis).

3) Install dependencies
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4) Run
```bash
cd backend
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

5) Open
- Web Interface: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

Docker (optional)
```bash
docker-compose up --build
```

## Project Structure
- `backend/`: FastAPI app (`main.py`), multi-agent figure analysis system, tracing hooks.
- `frontend/index.html`: Figure upload interface with drag-and-drop, demo functionality.
- `test scripts/`: `test_figure_analysis.py` for API testing with mock scientific figures.
- Root: `start.sh`, `docker-compose.yml`, `README.md`, `figure_feedback_agent_prd.md`.

## Development Commands
- Backend (dev): `cd backend && source .venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
- Test mode (no vision API): `cd backend && source .venv/bin/activate && TEST_MODE=1 uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
- API test: `python "test scripts"/test_figure_analysis.py`

## API
- POST `/analyze-figure` → analyzes scientific figures and returns structured feedback.
  Example body:
  ```json
  {
    "image_data": "data:image/png;base64,iVBORw0KGgoAAAANS...",
    "json_structure": {"elements": [{"type": "pathway", "name": "Cell Division"}]},
    "context": "Research publication figure",
    "figure_type": "pathway diagram"
  }
  ```
- POST `/analyze-figure-upload` → file upload version (multipart/form-data)
- GET `/health` → simple status check.

## Notes on Tracing (Optional)
- If `ARIZE_SPACE_ID` and `ARIZE_API_KEY` are set, OpenInference exports spans for agents/tools/LLM calls. View at https://app.arize.com.

## Troubleshooting
- 500 errors/failed analysis: verify `OPENAI_API_KEY` or `OPENROUTER_API_KEY` in `backend/.env`.
- Vision analysis fails: ensure image is valid PNG/JPG, system falls back to JSON analysis.
- Missing dependencies: run `pip install -r requirements.txt` in activated virtual environment.
- No traces: ensure Arize credentials are set and reachable.
- Port conflicts: stop existing services on 8000 or change ports.
- Use `TEST_MODE=1` for development without vision API calls.

## Deploy on Render
- This repo includes `render.yaml`. Connect your GitHub repo in Render and deploy as a Web Service.
- Render will run: `pip install -r backend/requirements.txt` and `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Set `OPENAI_API_KEY` (or `OPENROUTER_API_KEY`) and optional Arize vars in the Render dashboard.
- For production, do not set `TEST_MODE` to enable full vision-LLM analysis capabilities.
