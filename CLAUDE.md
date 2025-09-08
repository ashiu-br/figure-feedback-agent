# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Backend Development
```bash
# ALWAYS use virtual environment first
cd backend && python3 -m venv .venv && source .venv/bin/activate

# Install dependencies (preferred: uv for speed)
pip install -r requirements.txt
# Alternative: uv pip install -r requirements.txt

# Start development server with hot reload  
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Quick startup script (starts backend, serves frontend at /)
../start.sh
```

### Testing
```bash
# Figure analysis API test (primary test suite)
python "test scripts"/test_figure_analysis.py

# Test mode for development (disables vision API calls)
TEST_MODE=1 uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Legacy tests (from trip planner - for reference)
python "test scripts"/test_api.py
python "test scripts"/synthetic_data_gen.py --base-url http://localhost:8000 --count 12
python "test scripts"/quick_test.py
```

### Docker
```bash
# Full stack with Docker Compose
docker-compose up --build
```

## Architecture Overview

This is the **BioRender Figure Feedback Agent** - an AI-powered system for analyzing and improving scientific figures using a **multi-agent architecture** with parallel execution.

### Core Components
- **Backend**: FastAPI application (`backend/main.py`) serving both API and static frontend
- **Frontend**: Single-page HTML application with Tailwind CSS (`frontend/index.html`)
- **Multi-Agent System**: 4 specialized LangGraph agents executing in parallel
- **Figure Processing**: Image and JSON structure analysis capabilities
- **Observability**: Optional Arize/OpenInference tracing

### Multi-Agent System Design

The system uses 5 specialized agents for comprehensive figure analysis:

1. **Visual Design Agent** - Color usage, layout, hierarchy, typography, spacing analysis
2. **Communication Agent** - Logical flow, information density, audience appropriateness  
3. **Scientific Agent** - Nomenclature, pathway logic, field conventions validation
4. **Content Interpretation Agent** - Plain language summaries using hybrid vision-LLM approach with JSON fallback
5. **Feedback Synthesizer Agent** - Combines all analyses into prioritized recommendations

**Execution Flow:**
```
START → [Visual Design, Communication, Scientific, Content Interpretation] (parallel) → Feedback Synthesizer → END
```

**Vision-LLM Integration:**
- Uses GPT-4o-mini vision API for image analysis when available
- Graceful fallback to JSON-based interpretation in TEST_MODE or when vision fails
- Hybrid approach ensures robust content interpretation across all scenarios

**Target Performance**: <15 seconds per figure analysis

### Key Technical Patterns

**LangGraph State Management:**
- Uses `FigureState` TypedDict for state passing between agents
- No MemorySaver/checkpointer (causes state issues)
- Fresh state initialization per request
- Compile graphs with simple `g.compile()` - no thread_id needed

**Agent Implementation:**
- Each agent uses `@tool` decorators for their specialized analysis functions
- Returns structured string responses with scores and recommendations
- Proper error handling and fallback responses
- LLM initialization supports both OpenAI and OpenRouter providers

**FastAPI Patterns:**
- CORS middleware enabled for frontend access
- Pydantic models for request/response validation (`FigureAnalysisRequest`, `FigureAnalysisResponse`)
- Main endpoints: `POST /analyze-figure` and `POST /analyze-figure-upload`
- Health check: `GET /health`
- Static frontend served at root `/`
- File upload handling with multipart/form-data support
- BASE64 image encoding for JSON API endpoint

## Environment Configuration

Required environment variables in `backend/.env`:

```bash
# LLM Provider (choose one)
OPENAI_API_KEY=your_openai_key
# OR
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=openai/gpt-4o-mini

# Optional: Development mode (disables vision analysis)
TEST_MODE=1

# Optional: Observability/Tracing
ARIZE_SPACE_ID=your_space_id
ARIZE_API_KEY=your_arize_key
```

Copy from `backend/env_example.txt` to get started.

## Critical Implementation Details

### LangGraph Graph Construction
- Build with `StateGraph(FigureState)` (updated from TripState)
- Use parallel edges from START to multiple agents
- Single convergence point at final agent before END
- **Never use MemorySaver** - causes state persistence issues
- All 4 analysis agents execute in parallel for optimal performance

### Tracing Setup
- Initialize tracing **once at module level**, not per request
- Uses OpenInference instrumentors for LangChain and LiteLLM
- Optional based on environment variable presence
- View traces at https://app.arize.com

### Performance Considerations
- Agents execute in true parallel (not sequential)
- Current setup handles ~10-15 concurrent requests
- Uses appropriate LLM models (gpt-3.5-turbo for speed vs gpt-4 for quality)

## File Structure
```
backend/
├── main.py              # FastAPI app with multi-agent system
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (copy from env_example.txt)
└── archive/            # Legacy implementations

frontend/
└── index.html          # Complete Tailwind CSS UI served by backend

test scripts/
├── test_api.py         # API testing
├── synthetic_data_gen.py # Evaluation suite
└── quick_test.py       # Development utilities

docker-compose.yml      # Container deployment
render.yaml            # Render.com deployment config
start.sh              # Development startup script
```

## Adding New Agents/Tools

### New Agent Pattern:
1. Create agent function with proper tool integration
2. Add to `FigureState` if new state fields needed  
3. Update graph construction in `build_graph()`
4. Consider parallel vs sequential execution placement
5. For vision capabilities, use hybrid approach with fallback to JSON analysis

### New Tool Pattern:
```python
@tool
def your_tool_name(query: str) -> str:
    """Tool description for LLM."""
    # Implementation
    return "formatted_response"
```

### API Integration Guidelines:
- Implement caching for expensive API calls
- Add graceful fallbacks to mock data
- Use async/await for all external calls
- Include proper error handling and rate limiting

## Common Issues

**Agent Execution Issues:**
- Verify no MemorySaver in graph compilation
- Check parallel edges in graph construction  
- Ensure fresh state per request
- Use TEST_MODE=1 for development when vision API unavailable

**Parameter Validation:**
- Empty optional fields should default to "" not None
- Use `context or ""` instead of `context if context else None`

**Vision Analysis Issues:**
- GPT-4o-mini may reject invalid/small images (expected behavior)
- System gracefully falls back to JSON analysis
- Verify image is valid PNG/JPG with reasonable dimensions

**Tracing Problems:**
- Initialize at module level only
- Verify Arize credentials in environment
- Check instrumentor version compatibility

**Performance:**
- Confirm parallel execution working (check timing logs)
- Monitor LLM API response times
- Consider model selection for speed vs quality tradeoffs
- Vision analysis adds 2-3 seconds but provides much richer interpretation