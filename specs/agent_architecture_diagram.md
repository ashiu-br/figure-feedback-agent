# BioRender Figure Feedback Agent - Architecture Diagram

## Multi-Agent System Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BIORENDER FIGURE ANALYSIS                          │
│                              Multi-Agent System                              │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: Figure Analysis Request
├── image_data (base64)
├── json_structure (BioRender JSON)
├── context (optional)
└── figure_type (optional)

                                    ┌─────────┐
                                    │  START  │
                                    └────┬────┘
                                         │
                                         ▼
                            ┌─────────────────────┐
                            │   CONTENT           │
                            │ INTERPRETATION STEP │
                            │  (Preprocessing)    │
                            └─────────────────────┘
                                         │
                                         ▼
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
        ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
        │  VISUAL DESIGN  │  │ COMMUNICATION   │  │   SCIENTIFIC    │
        │     AGENT       │  │     AGENT       │  │     AGENT       │
        │ (Content-Aware) │  │ (Content-Aware) │  │ (Content-Aware) │
        └─────────────────┘  └─────────────────┘  └─────────────────┘
                │                        │                        │
                ▼                        ▼                        ▼
        ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
        │@tool            │  │@tool            │  │@tool            │
        │analyze_visual_  │  │evaluate_comm_   │  │validate_sci_    │
        │design +         │  │clarity +        │  │accuracy +       │
        │content_summary  │  │content_summary  │  │content_summary  │
        └─────────────────┘  └─────────────────┘  └─────────────────┘

                    ┌────────────────────────────────────┐
                    │            CONVERGENCE             │
                    │     (All analyses complete)        │
                    └────────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │        FEEDBACK SYNTHESIZER         │
                    │             AGENT                   │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │@tool                                │
                    │synthesize_feedback                  │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
                            ┌─────────────┐
                            │     END     │
                            └─────────────┘

OUTPUT: Figure Analysis Response
├── visual_design_score (0-10)
├── communication_score (0-10)
├── scientific_accuracy_score (0-10)
├── overall_score (0-30)
├── content_summary (plain language)
├── feedback (comprehensive analysis)
├── recommendations (prioritized list)
└── processing_time (seconds)
```

## Agent Details

### 0. Content Interpretation Step (Preprocessing)
**Function**: `content_interpretation_step()`  
**Tool**: `@tool interpret_figure_content()`  
**Purpose**: **Critical First Step** - Understand what the figure communicates before evaluating design quality
**Focus Areas**:
- 👁️ Vision-based analysis (GPT-4o-mini) 
- 🗣️ Plain language summaries
- 🔄 Hybrid approach (vision + JSON fallback)
- 📋 Message identification for context-aware analysis

**Why Content-First?**: Agents need to know what message the figure is trying to convey before they can evaluate how well the design supports that message.

### 1. Visual Design Agent (Content-Aware)
**Function**: `visual_design_agent()`  
**Tool**: `@tool analyze_visual_design(content_summary)`  
**Focus Areas**:
- 🎨 Color palette analysis (context-specific recommendations)
- 📝 Typography hierarchy validation
- 📐 Layout and spacing assessment  
- 💡 Accessibility considerations
- 🎯 **NEW**: Design-message alignment (e.g., pathway figures need directional flow)

**Content Integration**: References `content_summary` to provide targeted recommendations based on figure type and message.

### 2. Communication Agent (Content-Aware)
**Function**: `communication_agent()`  
**Tool**: `@tool evaluate_communication_clarity(content_summary)`  
**Focus Areas**:
- 🎯 Information flow analysis
- 📊 Information density evaluation
- 👥 Audience appropriateness
- 💡 Narrative clarity
- ⚠️ **NEW**: Message-design mismatch detection

**Content Integration**: Validates that visual elements support the identified message (e.g., pathway figures should have flow indicators).

### 3. Scientific Agent (Content-Aware)
**Function**: `scientific_agent()`  
**Tool**: `@tool validate_scientific_accuracy(content_summary)`  
**Focus Areas**:
- 🔬 Nomenclature validation
- 🧬 Pathway logic verification
- 📏 Standards compliance
- ⚠️ Literature cross-reference
- 🎯 **NEW**: Content-specific accuracy checks

**Content Integration**: Applies specialized validation based on identified content type (pathway vs. mechanism vs. process).

### 5. Feedback Synthesizer Agent
**Function**: `feedback_synthesizer_agent()`  
**Tool**: `@tool synthesize_feedback()`  
**Focus Areas**:
- 📊 Score aggregation and analysis
- 🎯 Priority ranking of recommendations
- 📝 Comprehensive feedback generation
- 💡 Implementation guidance

**Thinking Steps**:
1. Analysis review
2. Priority ranking
3. Synthesis
4. Final review

## Execution Flow

### Content-First Processing (NEW ARCHITECTURE)
- **Phase 1**: Content Interpretation Step (preprocessing) - **CRITICAL FIRST**
- **Phase 2**: 3 analysis agents execute simultaneously (now content-aware)
- **Phase 3**: Results converge at Feedback Synthesizer  
- **Phase 4**: Synthesized output to END

### Why Content-First Architecture?
**Problem Solved**: Previous architecture had agents analyzing figures without understanding what they were trying to communicate. How can you judge if visual design is effective without knowing the intended message?

**Solution**: Content interpretation happens first, providing context for all subsequent analysis.

### Real-time Progress Tracking
```
WebSocket Connection (Optional)
├── Agent start notifications
├── Thinking step updates
├── Tool execution status
├── Agent completion with confidence scores
└── Final analysis completion
```

### State Management
```typescript
FigureState {
  // Input data
  image_data: string
  json_structure: Dict[str, Any]
  context?: string
  figure_type?: string
  
  // Agent outputs
  visual_analysis?: string
  communication_analysis?: string
  scientific_analysis?: string
  content_interpretation?: string
  feedback_summary?: string
  quality_scores?: Dict[str, int]
  
  // Progress tracking
  agent_progress?: Dict[str, AgentProgress]
  websocket?: WebSocket
  session_id?: string
}
```

## Key Features

### Hybrid Vision-LLM Approach
- **Primary**: GPT-4o-mini vision API for image analysis
- **Fallback**: JSON-based rule analysis
- **Development**: TEST_MODE bypasses vision calls

### Error Handling & Resilience
- Graceful degradation when APIs unavailable
- JSON fallback for vision failures
- Parameter validation and sanitization
- Comprehensive logging and tracing

### Performance Optimization
- ~15 seconds total analysis time
- Parallel agent execution
- Efficient LLM model selection
- Optional Arize tracing integration

### API Endpoints
- `POST /analyze-figure` - JSON API
- `POST /analyze-figure/upload` - File upload
- `WebSocket /ws/{session_id}` - Real-time updates
- `GET /health` - Health check
- `GET /` - Frontend interface

## Technology Stack
- **Backend**: FastAPI, Python 3.10+
- **AI Framework**: LangGraph, LangChain
- **LLM**: OpenAI GPT-4o-mini / OpenRouter
- **Vision**: GPT-4o-mini vision API
- **WebSocket**: Real-time progress updates
- **Observability**: Arize/OpenInference (optional)
