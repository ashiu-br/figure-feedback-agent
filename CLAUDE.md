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

# Vision-only server (no JSON structure required)
uvicorn main_vision_only:app --host 0.0.0.0 --port 8001 --reload

# IMPORTANT: Do NOT use TEST_MODE when OpenAI API key is available
# TEST_MODE disables real AI analysis and returns mock responses

# Manual startup (start.sh script not available)
# Start backend and frontend is served at root /
```

### Testing
```bash
# Figure analysis API test (primary test suite)
python "test scripts"/test_figure_analysis.py

# Vision-only analysis test
python test_vision_only.py

# Test mode ONLY when no API keys available (disables vision API calls)
# WARNING: Only use when OPENAI_API_KEY and OPENROUTER_API_KEY are missing
# TEST_MODE=1 uvicorn main:app --host 0.0.0.0 --port 8000 --reload

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
- **Vision-Only Backend**: Simplified FastAPI app (`backend/main_vision_only.py`) for image-only analysis
- **Frontend**: Single-page HTML application with Tailwind CSS (`frontend/index.html`)
- **Vision-Only Frontend**: Streamlined interface (`frontend/index_vision_only.html`) with drag-and-drop image upload
- **Multi-Agent System**: 5 specialized LangGraph agents executing in parallel
- **Figure Processing**: Image and JSON structure analysis capabilities (original) + pure vision analysis (vision-only)
- **Real-time Progress**: WebSocket support for live agent progress updates
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
- **Vision-Only Mode**: Pure GPT-4o vision analysis without JSON requirements

**Target Performance**: <15 seconds per figure analysis (both original and vision-only)

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
- WebSocket endpoints: `POST /analyze-figure/ws/{session_id}` and `WS /ws/{session_id}`
- Health check: `GET /health`
- Static frontend served at root `/`
- File upload handling with multipart/form-data support
- BASE64 image encoding for JSON API endpoint
- Real-time progress updates via WebSocket connections

## Environment Configuration

Required environment variables in `backend/.env`:

```bash
# LLM Provider (choose one)
OPENAI_API_KEY=your_openai_key
# OR
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=openai/gpt-4o-mini

# Optional: Development mode (disables vision analysis)
# WARNING: Do NOT use TEST_MODE when API keys are available - it returns mock data
# TEST_MODE=1

# Optional: Arize AX Observability/Tracing  
ARIZE_SPACE_ID=your_space_id
ARIZE_API_KEY=your_arize_key
ARIZE_PROJECT_NAME=biorender-figure-feedback-agent
```

Copy from `backend/env_example.txt` to get started.

## Critical Implementation Details

### LangGraph Graph Construction
- Build with `StateGraph(FigureState)` (updated from TripState)
- Use parallel edges from START to multiple agents
- Single convergence point at final agent before END
- **Never use MemorySaver** - causes state persistence issues
- All 4 analysis agents execute in parallel for optimal performance

### Arize AX Tracing Setup
- **Modern Arize AX observability** with OpenInference instrumentation
- Initialize tracing **once at module level**, not per request  
- Comprehensive LangGraph/LangChain workflow tracing with custom spans
- LLM call instrumentation via LiteLLM integration
- Optional based on environment variable presence
- Enhanced error tracking and performance metrics
- View traces at https://app.arize.com with rich agent execution context

### Performance Considerations
- Agents execute in true parallel (not sequential)
- Current setup handles ~10-15 concurrent requests
- Uses appropriate LLM models (gpt-3.5-turbo for speed vs gpt-4 for quality)

## File Structure
```
backend/
├── main.py              # FastAPI app with multi-agent system (JSON + Image)
├── main_vision_only.py  # Vision-only FastAPI app (Image only)
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Python project configuration
├── Dockerfile          # Docker container configuration
├── env_example.txt      # Environment template
├── .env                 # Environment variables (copy from env_example.txt)
└── archive/            # Legacy implementations

frontend/
├── index.html          # Complete Tailwind CSS UI with real-time progress (JSON + Image)
└── index_vision_only.html # Vision-only UI with drag-and-drop (Image only)

test scripts/
├── test_figure_analysis.py # Primary figure analysis test suite
├── test_api.py         # Legacy API testing  
├── synthetic_data_gen.py # Evaluation suite
└── quick_test.py       # Development utilities

# Vision-only testing
├── test_vision_only.py # Vision-only analysis test script

# Documentation & Specifications
├── CLAUDE.md           # Development guide (this file)
├── specs/              # Canonical specifications (source of truth)
│   ├── JSON_INTELLIGENCE_SPEC.md # JSON parsing system specification
│   ├── figure_feedback_agent_prd.md # Product requirements document
│   └── scoring_system_improvements.md # Scoring framework proposals
├── BIORENDER_MCP_SPEC.md # MCP server specification
└── README.md           # Project overview

docker-compose.yml      # Container deployment
render.yaml            # Render.com deployment config
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
- NEVER use TEST_MODE=1 when OPENAI_API_KEY or OPENROUTER_API_KEY are available
- TEST_MODE=1 only for development when NO API keys are configured
- TEST_MODE returns mock data and disables real AI analysis

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

**WebSocket Integration:**
- Real-time progress updates require WebSocket connection
- Frontend automatically connects to WebSocket for live updates
- Agent progress cards show thinking, tool usage, and completion states
- Use session-based endpoints (`/analyze-figure/ws/{session_id}`) for progress tracking

**Known JSON Parsing Limitations:**
- Current system has hardcoded assumptions about BioRender JSON structure
- May return "0 elements detected" due to incorrect key assumptions
- See `specs/JSON_INTELLIGENCE_SPEC.md` for planned fixes to dynamic JSON parsing
- Affects accuracy of visual design, communication, and scientific analysis

## Vision-Only Implementation

### Overview
The vision-only system (`main_vision_only.py` + `index_vision_only.html`) provides a simplified workflow that analyzes scientific figures using pure GPT-4o vision capabilities without requiring JSON structure files.

### Key Differences from Original System

**Backend (`main_vision_only.py`):**
- Simplified `FigureAnalysisRequest` - no `json_structure` field required
- All analysis tools (`analyze_visual_design`, `evaluate_communication_clarity`, `validate_scientific_accuracy`) use GPT-4o vision directly
- Removed JSON parsing and fallback logic
- Same multi-agent architecture and WebSocket progress tracking

**Frontend (`index_vision_only.html`):**
- Removed JSON file upload UI entirely
- Enhanced drag-and-drop image upload with visual feedback
- "Vision Analysis" branding and GPT-4o Vision messaging
- Same real-time progress tracking and results display

### When to Use Vision-Only vs Original

**Use Vision-Only When:**
- Quick analysis of any scientific figure image (PNG/JPG)
- No access to BioRender JSON structure files
- Broader compatibility with figures from any source
- Simpler workflow for users

**Use Original When:**
- Have access to BioRender JSON structure files
- Need detailed structural analysis (element counts, exact colors, etc.)
- Working specifically with BioRender-generated figures
- Want hybrid vision + JSON analysis for maximum accuracy

### Running Both Systems

```bash
# Original system (port 8000)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Vision-only system (port 8001)
uvicorn main_vision_only:app --host 0.0.0.0 --port 8001 --reload
```

**URLs:**
- Original: http://localhost:8000/ (requires JSON + Image)
- Vision-Only: http://localhost:8001/ (Image only)

### Testing Vision-Only
```bash
# Run vision-only test script
python test_vision_only.py

# Test endpoints directly
curl http://localhost:8001/health
```
