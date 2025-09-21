# AGENTS.md

## System Overview
The BioRender Figure Feedback Agent runs a content-first LangGraph workflow that evaluates scientific figures across design, communication, and scientific accuracy dimensions. Requests land in FastAPI (`backend/main.py`), where `_analyze_figure_internal` builds the graph, seeds a `FigureState`, and streams progress over WebSockets before returning a structured `FigureAnalysisResponse`.

## Workflow at a Glance
1. **Graph construction** – `build_graph()` wires the nodes `content_interpretation → {visual_design, communication, scientific} → feedback_synthesizer`, enabling the three analysis agents to execute in parallel after the shared preprocessing step.
2. **State priming** – Requests populate `FigureState` with sanitized image metadata, JSON structure, optional context, and an empty `agent_progress` map; the raw base64 image is kept in a private key to avoid logging sensitive data.
3. **Agent execution** – Each node simulates “thinking”, invokes its dedicated tool, and writes findings back into the state. The feedback synthesizer reads all intermediate results and emits scores plus markdown-formatted recommendations.
4. **Response assembly** – Post-run logic extracts bullet recommendations from the synthesized feedback, attaches quality scores, and computes total processing time.

## Agent Roster
### 0. Content Interpretation Step
- **Node**: `content_interpretation_step`
- **Tool**: `interpret_figure_content`
- **Inputs**: Base64 image, BioRender JSON, optional context/figure type.
- **Behavior**: Attempts GPT-4o-mini vision first; on failure or in `TEST_MODE`, falls back to rule-based JSON parsing that infers purpose, key terms, and flow indicators. Provides the plain-language summary every downstream agent consumes.

### 1. Visual Design Agent
- **Node**: `visual_design_agent`
- **Tool**: `analyze_visual_design`
- **Focus**: Color palette health, typography hierarchy, layout/spacing, accessibility, and content-aware suggestions (e.g., pathway color coding).
- **Notes**: Runs after the interpretation step so it can tailor recommendations to the detected figure narrative. Emits a score out of 10 inside the tool output and records confidence for UI progress cards.

### 2. Communication Agent
- **Node**: `communication_agent`
- **Tool**: `evaluate_communication_clarity`
- **Focus**: Logical flow, information density, audience fit, and narrative coherence.
- **Content awareness**: Flags message-design mismatches (e.g., pathway described without arrows) using the shared content summary. Returns a 10-point score.

### 3. Scientific Accuracy Agent
- **Node**: `scientific_agent`
- **Tool**: `validate_scientific_accuracy`
- **Focus**: Nomenclature hygiene, pathway logic, unit usage, and convention checks with lightweight heuristics. Adjusts messaging for pathway/mechanism/timeline content types and logs targeted reminders for literature validation.

### 4. Feedback Synthesizer Agent
- **Node**: `feedback_synthesizer_agent`
- **Tool**: `synthesize_feedback`
- **Focus**: Aggregates the three analyses, harmonizes their markdown, derives overall/section scores, and ranks priority actions. Emits an overall score (sum of sections) and triggers the final WebSocket completion payload.

## Shared State & Progress Tracking
- **FigureState** stores inputs, agent outputs, `quality_scores`, and the WebSocket handle. An internal `_image_data_private` key preserves base64 data without exposing it in telemetry.
- **AgentProgress** objects keep per-agent status, current step, thinking log, timestamps, and confidence. Frontend cards subscribe to these updates via `/ws/{session_id}`.
- **send_progress(...)** guards WebSocket emission and is called at every start/thinking/tool/complete transition, plus global `analysis_start` and `analysis_complete` events.

## Tool Contracts & Scoring
- **interpret_figure_content** – Hybrid vision/JSON summary generator.
- **analyze_visual_design** – Counts colors, text, shapes, and emits design recommendations with score extraction hints.
- **evaluate_communication_clarity** – Measures element density, flow indicators, and audience alignment; returns structured feedback and score.
- **validate_scientific_accuracy** – Scans text for nomenclature/unit issues and content-specific caveats.
- **synthesize_feedback** – Parses embedded section scores, sets priority bands, and formats the final report used in the API response and recommendation extraction logic.

Score parsing relies on regex searches in `feedback_synthesizer_agent`, so tool outputs must retain the `Score: X/10` pattern. Recommendations are any lines prefixed with `→`; downstream code classifies them into visual/communication/scientific buckets.

## Vision-Only Variant
`backend/main_vision_only.py` mirrors the same LangGraph topology but assumes no JSON input. Every tool calls GPT-4o-mini vision directly, supplying tailored prompts for design, communication, and accuracy analysis. The response schema gains experimental `scientific_coherence_*` fields, but the agent sequence, progress reporting, and recommendation extraction remain consistent with the hybrid pipeline.

## Observability Hooks
When Arize credentials are present, tracing is enabled for LangChain and LiteLLM components. Large base64 payloads are suppressed via the `suppress_tracing_if_needed` context manager, and, when possible, persisted image URLs are attached to spans for external visualization.

## Extending the Graph
1. Add any new state fields to `FigureState` and initialize them inside `_analyze_figure_internal`.
2. Implement the agent function with progress updates and a dedicated tool (or direct LLM call).
3. Register the node and edges inside `build_graph()` to place it before the synthesizer (parallel fan-out is typical).
4. Update the frontend if you want real-time status cards for the new agent.
5. Ensure the synthesizer (or new downstream aggregators) read any additional outputs and include them in the API response.

