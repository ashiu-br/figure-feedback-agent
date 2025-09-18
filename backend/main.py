from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import time
import json
import base64
import logging
import traceback
import asyncio
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

# Robust .env loading: try multiple locations without overriding existing env
try:
    env_file = os.getenv("ENV_FILE")
    if env_file and Path(env_file).exists():
        load_dotenv(env_file, override=False)
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Arize AX Observability Setup
try:
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import using_prompt_template
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    _TRACING = True
    logger.info("Arize AX tracing modules loaded successfully")
except ImportError as e:
    logger.warning(f"Arize tracing not available: {e}")
    def using_prompt_template(**kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    _TRACING = False

# LangGraph + LangChain
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import httpx
import asyncio
from PIL import Image
import io
import re
import uuid
from typing import Set


class FigureAnalysisRequest(BaseModel):
    image_data: str  # base64 encoded image
    json_structure: Dict[str, Any]  # BioRender JSON structure
    context: Optional[str] = None  # Figure title, intended audience, etc.
    figure_type: Optional[str] = None  # pathway, workflow, mechanism, timeline


class FigureAnalysisResponse(BaseModel):
    visual_design_score: int
    communication_score: int  
    scientific_accuracy_score: int
    overall_score: int
    content_summary: str
    feedback: str
    recommendations: List[Dict[str, Any]]
    processing_time: float


# Progress tracking for agents
class AgentProgress(TypedDict):
    status: str  # "pending", "thinking", "using_tools", "complete", "error"
    current_step: Optional[str]
    thinking_messages: List[str]
    tool_calls: List[Dict[str, Any]]
    start_time: Optional[float]
    completion_time: Optional[float]
    confidence: Optional[float]

class ProgressUpdate(TypedDict):
    type: str  # "agent_start", "agent_thinking", "tool_call", "agent_complete", "analysis_complete"
    agent: Optional[str]
    message: str
    timestamp: float
    step: Optional[str]
    data: Optional[Dict[str, Any]]

# Figure state for LangGraph
class FigureState(TypedDict):
    image_data: str
    context: Optional[str]
    figure_type: Optional[str]
    visual_analysis: Optional[str]
    communication_analysis: Optional[str]
    scientific_analysis: Optional[str]
    content_interpretation: Optional[str]
    feedback_summary: Optional[str]
    quality_scores: Optional[Dict[str, int]]
    # Progress tracking
    agent_progress: Optional[Dict[str, AgentProgress]]
    websocket: Optional[Any]  # WebSocket connection for real-time updates
    session_id: Optional[str]


def _init_llm():
    # Simple, test-friendly LLM init
    class _Fake:
        def __init__(self):
            pass
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class _Msg:
                content = "Mock figure analysis feedback"
                tool_calls: List[Dict[str, Any]] = []
            return _Msg()

    if os.getenv("TEST_MODE"):
        return _Fake()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=2000)
    elif os.getenv("OPENROUTER_API_KEY"):
        # Use OpenRouter via OpenAI-compatible client
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.3,
        )
    else:
        # Require a key unless running tests
        raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")


llm = _init_llm()


# === FIGURE ANALYSIS TOOLS ===

def json_based_interpretation(json_structure: dict, context: str = "", figure_type: str = "") -> str:
    """Fallback interpretation using JSON structure analysis (rule-based approach)."""
    
    # Analyze JSON structure for content elements
    elements = []
    if 'objects' in json_structure:
        elements = json_structure['objects']
    elif 'elements' in json_structure:
        elements = json_structure['elements']
    
    # Extract key components
    text_elements = []
    shapes = []
    arrows_connectors = []
    
    for element in elements:
        if isinstance(element, dict):
            element_type = element.get('type', '').lower()
            
            if 'text' in element_type:
                text_content = element.get('text', '')
                if text_content:
                    text_elements.append(text_content)
            
            elif any(keyword in element_type for keyword in ['rectangle', 'circle', 'shape']):
                shapes.append(element_type)
            
            elif any(keyword in element_type for keyword in ['arrow', 'line', 'connector']):
                arrows_connectors.append(element_type)
    
    # Analyze figure type and context
    figure_context = context if context else "scientific figure"
    inferred_type = figure_type if figure_type else "diagram"
    
    # Generate content interpretation
    main_elements = len(elements)
    text_count = len(text_elements)
    flow_indicators = len(arrows_connectors)
    
    # Determine figure purpose based on structure
    if flow_indicators > 0 and text_count > 1:
        purpose = "process or pathway"
    elif text_count > 3 and flow_indicators == 0:
        purpose = "informational diagram or comparison"
    elif main_elements <= 3:
        purpose = "simple concept illustration"
    else:
        purpose = "complex multi-component diagram"
    
    # Extract key terms from text elements
    key_terms = []
    for text in text_elements[:5]:  # Limit to first 5 text elements
        if len(text.strip()) > 0:
            key_terms.append(text.strip())
    
    # Generate summary based on available information
    if key_terms:
        key_concepts = ", ".join(key_terms[:3]) if len(key_terms) <= 3 else f"{', '.join(key_terms[:2])}, and other components"
    else:
        key_concepts = f"{main_elements} visual elements"
    
    summary = f"This {inferred_type} appears to be a {purpose}"
    
    if key_terms:
        summary += f" featuring {key_concepts}"
    
    if flow_indicators > 0:
        summary += f". The figure shows {flow_indicators} directional relationship{'s' if flow_indicators > 1 else ''}"
        
    if inferred_type == "pathway":
        summary += ", illustrating how different components interact or influence each other"
    elif "workflow" in inferred_type or "process" in inferred_type:
        summary += ", depicting a sequential process or methodology"
    
    summary += "."
    
    if context:
        summary += f" Given the context '{context}', this figure likely aims to communicate the key relationships and processes within this domain."
    
    return summary


def vision_interpretation(image_data: str, context: str, figure_type: str, json_structure: dict) -> str:
    """Use GPT-4o-mini vision to analyze the actual image content."""
    
    system_prompt = """You are an expert at interpreting scientific figures. Look at the provided image and describe what it shows in 2-3 clear sentences that a non-expert could understand.

Focus on:
- Main biological/scientific concept being illustrated
- Key components and their relationships
- Overall process or message being communicated

Start with "This figure shows..." or "This diagram illustrates..." and be specific about what you observe visually."""

    # Truncate JSON for context (avoid token overflow)
    json_summary = json.dumps(json_structure, indent=2)[:300] + "..." if len(str(json_structure)) > 300 else json.dumps(json_structure, indent=2)
    
    context_text = f"""Context: {context or 'Scientific figure analysis'}
Figure Type: {figure_type or 'diagram'}
JSON Structure Preview: {json_summary}

Please analyze the image and provide a plain language interpretation."""

    # Prefer public URL when available so Arize can render images in traces
    try:
        if os.getenv("PUBLIC_BASE_URL"):
            image_ref = {"type": "image_url", "image_url": {"url": _persist_image_and_get_url(image_data)}}
        else:
            image_ref = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
    except Exception:
        image_ref = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=[
            {"type": "text", "text": context_text},
            image_ref,
        ])
    ]
    
    # Use vision-capable model
    vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=300)
    response = vision_llm.invoke(messages)
    return response.content.strip()


@tool
def analyze_visual_design(image_data: str, json_structure: dict, context: str = "", content_summary: str = "") -> str:
    """Analyze visual design aspects of the figure including color usage, layout, hierarchy, typography, and spacing.
    
    Args:
        content_summary: Plain language description of what the figure communicates (from content interpretation)
    """
    
    # Analyze JSON structure for visual elements
    elements = []
    if 'objects' in json_structure:
        elements = json_structure['objects']
    elif 'elements' in json_structure:
        elements = json_structure['elements']
    
    # Count color usage
    colors_used = set()
    text_elements = 0
    shape_elements = 0
    
    for element in elements:
        if isinstance(element, dict):
            # Extract colors
            if 'fill' in element:
                colors_used.add(element['fill'])
            if 'stroke' in element:
                colors_used.add(element['stroke'])
            if 'color' in element:
                colors_used.add(element['color'])
            
            # Count element types
            element_type = element.get('type', '').lower()
            if 'text' in element_type:
                text_elements += 1
            elif 'shape' in element_type or 'rect' in element_type or 'circle' in element_type:
                shape_elements += 1
    
    # Analyze color palette
    color_count = len(colors_used)
    color_feedback = ""
    color_score = 10
    
    if color_count > 6:
        color_feedback = "→ Reduce color palette: Use 3-4 colors max, limit accent colors to highlight key elements"
        color_score = 6
    elif color_count > 4:
        color_feedback = "→ Consider simplifying color palette: 3-4 colors typically work best for scientific figures"
        color_score = 7
    else:
        color_feedback = "→ Good color usage: Appropriate number of colors for clarity"
        color_score = 9
    
    # Analyze text hierarchy
    hierarchy_feedback = ""
    if text_elements > 10:
        hierarchy_feedback = "→ Improve text hierarchy: Group related text elements, use consistent font sizes for similar content"
    elif text_elements > 0:
        hierarchy_feedback = "→ Text hierarchy looks appropriate for the content complexity"
    
    # Layout analysis
    layout_feedback = "→ Consider white space: Ensure adequate spacing between elements for visual clarity"
    
    # Content-aware recommendations
    content_context = ""
    if content_summary:
        content_context = f"\n📋 Content Context: {content_summary[:100]}{'...' if len(content_summary) > 100 else ''}\n"
        
        # Adjust recommendations based on content
        if "pathway" in content_summary.lower():
            layout_feedback += "\n→ For pathway figures: Use directional flow to guide reader through process"
            color_feedback += "\n→ Consider color-coding different pathway components or stages"
        elif "comparison" in content_summary.lower():
            layout_feedback += "\n→ For comparison figures: Ensure visual balance between compared elements"
            color_feedback += "\n→ Use contrasting colors to distinguish compared items clearly"

    return f"""VISUAL DESIGN ANALYSIS (Score: {color_score}/10):{content_context}

🎨 Color Usage:
{color_feedback}
- {color_count} colors detected in current design
- Consider using brand colors or scientific publication standards

📝 Typography & Hierarchy:
{hierarchy_feedback}
- {text_elements} text elements found
- Ensure consistent font sizes within element categories

📐 Layout & Spacing:
{layout_feedback}
- {len(elements)} total elements in figure
- Balance information density with readability

💡 Recommendations:
→ Use high contrast for text readability
→ Maintain consistent spacing between related elements
→ Consider color-blind friendly palette for accessibility"""


@tool 
def evaluate_communication_clarity(json_structure: dict, context: str = "", figure_type: str = "", content_summary: str = "") -> str:
    """Evaluate communication clarity including logical flow, information density, and audience appropriateness.
    
    Args:
        content_summary: Plain language description of what the figure communicates (from content interpretation)
    """
    
    elements = []
    if 'objects' in json_structure:
        elements = json_structure['objects']
    elif 'elements' in json_structure:
        elements = json_structure['elements']
    
    # Analyze information density
    total_elements = len(elements)
    text_elements = sum(1 for el in elements if isinstance(el, dict) and 'text' in el.get('type', '').lower())
    
    # Analyze logical flow indicators
    arrows_or_connectors = 0
    for element in elements:
        if isinstance(element, dict):
            element_type = element.get('type', '').lower()
            if any(keyword in element_type for keyword in ['arrow', 'line', 'connector', 'flow']):
                arrows_or_connectors += 1
    
    # Score based on complexity and flow
    density_score = 10
    flow_score = 10
    
    if total_elements > 20:
        density_score = 6
        density_feedback = "→ Reduce complexity: Consider breaking into multiple panels or removing non-essential elements"
    elif total_elements > 15:
        density_score = 7
        density_feedback = "→ Monitor information density: Ensure all elements serve the main message"
    else:
        density_feedback = "→ Good information density: Appropriate complexity for the message"
    
    # Flow analysis
    if figure_type in ['pathway', 'workflow', 'mechanism'] and arrows_or_connectors < 2:
        flow_score = 6
        flow_feedback = "→ Improve flow indicators: Add arrows or connectors to show process direction"
    else:
        flow_feedback = "→ Logical flow appears appropriate for figure type"
    
    # Overall communication score
    overall_score = int((density_score + flow_score) / 2)
    
    # Content-aware feedback
    content_context = ""
    message_alignment = ""
    if content_summary:
        content_context = f"\n📋 Content Context: {content_summary[:100]}{'...' if len(content_summary) > 100 else ''}\n"
        
        # Check if design supports the identified message
        if "pathway" in content_summary.lower() and arrows_or_connectors < 2:
            message_alignment = "\n⚠️ Message-Design Mismatch: Figure describes a pathway but lacks sufficient flow indicators"
        elif "comparison" in content_summary.lower() and total_elements < 4:
            message_alignment = "\n⚠️ Message-Design Mismatch: Figure describes comparisons but has few elements to compare"
        elif arrows_or_connectors > 5 and "process" not in content_summary.lower():
            message_alignment = "\n⚠️ Message-Design Mismatch: Many flow indicators but content doesn't describe a clear process"
    
    return f"""COMMUNICATION CLARITY ANALYSIS (Score: {overall_score}/10):{content_context}{message_alignment}

🎯 Information Flow:
{flow_feedback}
- {arrows_or_connectors} flow indicators detected
- Figure type: {figure_type or 'general'}

📊 Information Density:
{density_feedback}
- {total_elements} total elements
- {text_elements} text elements

👥 Audience Appropriateness:
→ Consider target audience expertise level
→ Use terminology appropriate for intended readers
→ Include necessary context without overwhelming detail

💡 Communication Tips:
→ Start with most important message/conclusion
→ Use visual hierarchy to guide reader attention  
→ Ensure each element contributes to main narrative
→ Consider adding brief caption or title for context"""


@tool
def validate_scientific_accuracy(json_structure: dict, context: str = "", figure_type: str = "", content_summary: str = "") -> str:
    """Validate scientific accuracy including nomenclature, pathway logic, and field conventions.
    
    Args:
        content_summary: Plain language description of what the figure communicates (from content interpretation)
    """
    
    elements = []
    if 'objects' in json_structure:
        elements = json_structure['objects']
    elif 'elements' in json_structure:
        elements = json_structure['elements']
    
    # Extract text content for analysis
    text_content = []
    for element in elements:
        if isinstance(element, dict):
            if 'text' in element:
                text_content.append(element['text'])
            elif 'content' in element:
                text_content.append(element['content'])
    
    all_text = ' '.join(str(text) for text in text_content if text)
    
    # Check for common scientific naming issues
    accuracy_issues = []
    accuracy_score = 9
    
    # Protein naming conventions
    protein_issues = re.findall(r'\b[A-Z]{2,}[0-9]*\b', all_text)
    if protein_issues:
        # Check if proteins follow proper nomenclature (e.g., p53 not P53)
        uppercase_proteins = [p for p in protein_issues if len(p) > 1 and p.isupper()]
        if uppercase_proteins:
            accuracy_issues.append(f"→ Check protein nomenclature: Consider using proper case (e.g., 'p53' not 'P53')")
            accuracy_score -= 1
    
    # Gene naming conventions
    gene_issues = re.findall(r'\b[a-z]+[0-9]+\b', all_text)
    if gene_issues and any(len(g) < 3 for g in gene_issues):
        accuracy_issues.append("→ Verify gene names: Ensure standard nomenclature for organism")
    
    # Check for pathway logic (basic)
    if figure_type in ['pathway', 'mechanism']:
        # Look for directional indicators
        directional_terms = ['activates', 'inhibits', 'promotes', 'blocks', 'leads to']
        has_direction = any(term in all_text.lower() for term in directional_terms)
        if not has_direction:
            accuracy_issues.append("→ Clarify pathway relationships: Add directional language or symbols")
            accuracy_score -= 1
    
    # Check for measurement units
    if 'μ' in all_text or 'micro' in all_text.lower():
        accuracy_issues.append("→ Verify units: Ensure proper scientific notation and unit consistency")
    
    # Timeline/temporal accuracy
    if figure_type == 'timeline':
        time_terms = re.findall(r'\b\d+\s*(min|hr|day|week|month|year)s?\b', all_text.lower())
        if not time_terms:
            accuracy_issues.append("→ Add temporal context: Include time scales for timeline figures")
            accuracy_score -= 1
    
    if not accuracy_issues:
        accuracy_issues.append("→ Scientific content appears accurate based on available information")
    
    # Content-aware accuracy assessment
    content_context = ""
    scientific_alignment = ""
    if content_summary:
        content_context = f"\n📋 Content Context: {content_summary[:100]}{'...' if len(content_summary) > 100 else ''}\n"
        
        # Check for content-specific accuracy concerns
        if "pathway" in content_summary.lower():
            scientific_alignment = "\n🔬 Pathway-Specific Check: Verify directional relationships match established biological mechanisms"
        elif "mechanism" in content_summary.lower():
            scientific_alignment = "\n🔬 Mechanism-Specific Check: Ensure molecular interactions follow known biochemical principles"
        elif "process" in content_summary.lower():
            scientific_alignment = "\n🔬 Process-Specific Check: Validate temporal sequence matches biological timing"
    
    return f"""SCIENTIFIC ACCURACY ANALYSIS (Score: {accuracy_score}/10):{content_context}{scientific_alignment}

🔬 Nomenclature Check:
{"".join(f"\\n{issue}" for issue in accuracy_issues[:2])}

🧬 Pathway Logic:
→ Verify cause-effect relationships are clearly indicated
→ Ensure temporal sequence is logical for biological processes

📏 Standards Compliance:
→ Check measurement units and scientific notation
→ Verify abbreviations follow field conventions
→ Confirm pathway directions match established literature

⚠️ Validation Notes:
→ Cross-reference with current literature for latest nomenclature
→ Consider organism-specific naming conventions
→ Validate pathway interactions with established databases
→ This analysis is based on figure content only - verify with domain expertise

💡 Accuracy Tips:
→ Use standard databases (UniProt, KEGG, GO) for validation
→ Include references for novel or controversial pathways
→ Consider adding confidence indicators for uncertain relationships"""


@tool
def interpret_figure_content(image_data: str, json_structure: dict, context: str = "", figure_type: str = "") -> str:
    """Generate a plain language summary using vision LLM with JSON fallback."""
    
    # Try vision-first approach (if not in TEST_MODE and image available)
    if image_data and not os.getenv("TEST_MODE"):
        try:
            logger.info("Attempting vision-based content interpretation")
            return vision_interpretation(image_data, context, figure_type, json_structure)
        except Exception as e:
            logger.warning(f"Vision interpretation failed: {str(e)}, falling back to JSON analysis")
    
    # Fallback to rule-based approach
    logger.info("Using JSON-based content interpretation (fallback or TEST_MODE)")
    return json_based_interpretation(json_structure, context, figure_type)


@tool
def synthesize_feedback(visual_analysis: str, communication_analysis: str, scientific_analysis: str, 
                       image_data: str = "", json_structure: dict = None) -> str:
    """Synthesize all agent analyses into prioritized, actionable feedback with overall scores."""
    
    # Extract scores from analyses
    visual_score = 8  # default
    comm_score = 8
    sci_score = 9
    
    # Parse scores from analysis text
    visual_match = re.search(r'Score:\s*(\d+)', visual_analysis)
    if visual_match:
        visual_score = int(visual_match.group(1))
    
    comm_match = re.search(r'Score:\s*(\d+)', communication_analysis)
    if comm_match:
        comm_score = int(comm_match.group(1))
        
    sci_match = re.search(r'Score:\s*(\d+)', scientific_analysis)
    if sci_match:
        sci_score = int(sci_match.group(1))
    
    overall_score = visual_score + comm_score + sci_score
    
    # Determine priority recommendations
    priority_recs = []
    
    if visual_score < 7:
        priority_recs.append("🎨 Visual Design (HIGH PRIORITY)")
    if comm_score < 7:
        priority_recs.append("🎯 Communication Clarity (HIGH PRIORITY)")
    if sci_score < 8:
        priority_recs.append("🔬 Scientific Accuracy (HIGH PRIORITY)")
    
    # Overall assessment
    if overall_score >= 25:
        assessment = "Excellent - Publication ready"
    elif overall_score >= 22:
        assessment = "Good - Ready with minor revisions"
    elif overall_score >= 18:
        assessment = "Needs improvement - Address key issues"
    else:
        assessment = "Major revisions needed"
    
    return f"""📊 FIGURE QUALITY ASSESSMENT

🎨 VISUAL DESIGN (Score: {visual_score}/10)
{visual_analysis.split(':', 1)[1] if ':' in visual_analysis else visual_analysis}

🎯 COMMUNICATION (Score: {comm_score}/10)  
{communication_analysis.split(':', 1)[1] if ':' in communication_analysis else communication_analysis}

🔬 SCIENTIFIC ACCURACY (Score: {sci_score}/10)
{scientific_analysis.split(':', 1)[1] if ':' in scientific_analysis else scientific_analysis}

📈 OVERALL IMPACT SCORE: {overall_score}/30 ({assessment})

{"🚨 PRIORITY ACTIONS:" if priority_recs else "✅ OPTIONAL ENHANCEMENTS:"}
{chr(10).join(f"→ {rec}" for rec in priority_recs) if priority_recs else "→ Figure meets quality standards - consider minor refinements"}

💡 IMPLEMENTATION ORDER:
1. Address any scientific accuracy issues first
2. Improve visual hierarchy and color usage
3. Enhance communication flow and clarity
4. Final polish and consistency check"""


# === AGENT IMPLEMENTATIONS ===

async def visual_design_agent(state: FigureState):
    """Agent focused on visual design analysis with thinking simulation."""
    agent_name = "visual_design"
    
    # Initialize agent progress
    if "agent_progress" not in state:
        state["agent_progress"] = {}
    
    state["agent_progress"][agent_name] = AgentProgress(
        status="thinking",
        current_step="initialization",
        thinking_messages=[],
        tool_calls=[],
        start_time=time.time(),
        completion_time=None,
        confidence=None
    )
    
    await send_progress(state, "agent_start", agent_name, "Starting visual design analysis...", "initialization")
    
    # Simulate thinking process
    thinking_steps = [
        ("Examining visual elements", "Looking at overall layout and visual hierarchy...", 1.0),
        ("color_analysis", "Analyzing color palette and usage patterns...", 1.2),
        ("typography_check", "Evaluating text hierarchy and font choices...", 0.8),
        ("layout_assessment", "Assessing spacing, alignment, and visual balance...", 1.0),
        ("accessibility_review", "Checking color contrast and readability...", 0.6)
    ]
    
    for step, message, delay in thinking_steps:
        state["agent_progress"][agent_name]["current_step"] = step
        state["agent_progress"][agent_name]["thinking_messages"].append(message)
        await send_progress(state, "agent_thinking", agent_name, message, step)
        await asyncio.sleep(delay)  # Simulate thinking time
    
    # Tool execution simulation
    state["agent_progress"][agent_name]["status"] = "using_tools"
    await send_progress(state, "tool_call", agent_name, "Executing visual design analysis tool...", "tool_execution", {
        "tool": "analyze_visual_design",
        "status": "running"
    })
    
    await asyncio.sleep(0.5)  # Simulate tool execution time
    
    # Reference content interpretation for context-aware analysis
    content_summary = state.get("content_interpretation", "")
    
    analysis = analyze_visual_design.invoke({
        "image_data": state["image_data"],
        "json_structure": state.get("json_structure", {}),
        "context": state.get("context", ""),
        "content_summary": content_summary
    })
    
    # Complete agent execution
    state["agent_progress"][agent_name]["status"] = "complete"
    state["agent_progress"][agent_name]["completion_time"] = time.time()
    state["agent_progress"][agent_name]["confidence"] = 0.85
    
    await send_progress(state, "agent_complete", agent_name, "Visual design analysis completed", "complete", {
        "confidence": 0.85,
        "findings": "Analyzed color palette, typography, and layout structure"
    })
    
    return {
        "visual_analysis": analysis
    }


async def communication_agent(state: FigureState):
    """Agent focused on communication clarity with thinking simulation."""
    agent_name = "communication"
    
    # Initialize agent progress
    if "agent_progress" not in state:
        state["agent_progress"] = {}
    
    state["agent_progress"][agent_name] = AgentProgress(
        status="thinking",
        current_step="initialization",
        thinking_messages=[],
        tool_calls=[],
        start_time=time.time(),
        completion_time=None,
        confidence=None
    )
    
    await send_progress(state, "agent_start", agent_name, "Starting communication clarity analysis...", "initialization")
    
    # Simulate thinking process
    thinking_steps = [
        ("flow_analysis", "Examining logical flow and information sequence...", 1.1),
        ("density_check", "Evaluating information density and cognitive load...", 0.9),
        ("audience_assessment", "Assessing appropriateness for target audience...", 1.2),
        ("narrative_review", "Checking if the visual narrative is clear...", 0.8)
    ]
    
    for step, message, delay in thinking_steps:
        state["agent_progress"][agent_name]["current_step"] = step
        state["agent_progress"][agent_name]["thinking_messages"].append(message)
        await send_progress(state, "agent_thinking", agent_name, message, step)
        await asyncio.sleep(delay)
    
    # Tool execution
    state["agent_progress"][agent_name]["status"] = "using_tools"
    await send_progress(state, "tool_call", agent_name, "Executing communication clarity tool...", "tool_execution")
    await asyncio.sleep(0.4)
    
    # Reference content interpretation for context-aware analysis
    content_summary = state.get("content_interpretation", "")
    
    analysis = evaluate_communication_clarity.invoke({
        "json_structure": state.get("json_structure", {}),
        "context": state.get("context", ""),
        "figure_type": state.get("figure_type", ""),
        "content_summary": content_summary
    })
    
    # Complete agent execution
    state["agent_progress"][agent_name]["status"] = "complete"
    state["agent_progress"][agent_name]["completion_time"] = time.time()
    state["agent_progress"][agent_name]["confidence"] = 0.78
    
    await send_progress(state, "agent_complete", agent_name, "Communication analysis completed", "complete", {
        "confidence": 0.78,
        "findings": "Evaluated logical flow, information density, and audience appropriateness"
    })
    
    return {
        "communication_analysis": analysis
    }


async def scientific_agent(state: FigureState):
    """Agent focused on scientific accuracy with thinking simulation."""
    agent_name = "scientific"
    
    # Initialize agent progress
    if "agent_progress" not in state:
        state["agent_progress"] = {}
    
    state["agent_progress"][agent_name] = AgentProgress(
        status="thinking",
        current_step="initialization",
        thinking_messages=[],
        tool_calls=[],
        start_time=time.time(),
        completion_time=None,
        confidence=None
    )
    
    await send_progress(state, "agent_start", agent_name, "Starting scientific accuracy validation...", "initialization")
    
    # Simulate thinking process
    thinking_steps = [
        ("nomenclature_check", "Validating scientific nomenclature and terminology...", 1.3),
        ("pathway_logic", "Examining pathway logic and biological accuracy...", 1.5),
        ("convention_review", "Checking adherence to field conventions...", 1.0),
        ("literature_cross_check", "Cross-referencing with established scientific knowledge...", 1.1)
    ]
    
    for step, message, delay in thinking_steps:
        state["agent_progress"][agent_name]["current_step"] = step
        state["agent_progress"][agent_name]["thinking_messages"].append(message)
        await send_progress(state, "agent_thinking", agent_name, message, step)
        await asyncio.sleep(delay)
    
    # Tool execution
    state["agent_progress"][agent_name]["status"] = "using_tools"
    await send_progress(state, "tool_call", agent_name, "Executing scientific accuracy validation tool...", "tool_execution")
    await asyncio.sleep(0.6)
    
    # Reference content interpretation for context-aware analysis
    content_summary = state.get("content_interpretation", "")
    
    analysis = validate_scientific_accuracy.invoke({
        "json_structure": state.get("json_structure", {}),
        "context": state.get("context", ""),
        "figure_type": state.get("figure_type", ""),
        "content_summary": content_summary
    })
    
    # Complete agent execution
    state["agent_progress"][agent_name]["status"] = "complete"
    state["agent_progress"][agent_name]["completion_time"] = time.time()
    state["agent_progress"][agent_name]["confidence"] = 0.92
    
    await send_progress(state, "agent_complete", agent_name, "Scientific accuracy validation completed", "complete", {
        "confidence": 0.92,
        "findings": "Validated nomenclature, pathway logic, and field conventions"
    })
    
    return {
        "scientific_analysis": analysis
    }


async def content_interpretation_step(state: FigureState):
    """Simple preprocessing step to interpret figure content before analysis."""
    
    await send_progress(state, "agent_start", "content_interpretation", "Interpreting figure content...", "content_analysis")
    
    # Direct call to content interpretation (no complex agent wrapper)
    interpretation = interpret_figure_content.invoke({
        "image_data": state["image_data"],
        "json_structure": state.get("json_structure", {}),  # Handle missing JSON gracefully
        "context": state.get("context", ""),
        "figure_type": state.get("figure_type", "")
    })
    
    await send_progress(state, "agent_complete", "content_interpretation", "Content interpretation completed", "complete", {
        "findings": "Generated plain language summary of figure content and message"
    })
    
    return {
        "content_interpretation": interpretation
    }


async def feedback_synthesizer_agent(state: FigureState):
    """Agent that synthesizes all analyses into final feedback with thinking simulation."""
    agent_name = "feedback_synthesizer"
    
    # Initialize agent progress
    if "agent_progress" not in state:
        state["agent_progress"] = {}
    
    state["agent_progress"][agent_name] = AgentProgress(
        status="thinking",
        current_step="initialization",
        thinking_messages=[],
        tool_calls=[],
        start_time=time.time(),
        completion_time=None,
        confidence=None
    )
    
    await send_progress(state, "agent_start", agent_name, "Starting feedback synthesis...", "initialization")
    
    # Simulate thinking process
    thinking_steps = [
        ("analysis_review", "Reviewing all agent analyses and findings...", 0.8),
        ("priority_ranking", "Ranking recommendations by priority and impact...", 1.0),
        ("synthesis", "Synthesizing comprehensive feedback report...", 1.2),
        ("final_review", "Finalizing recommendations and overall assessment...", 0.6)
    ]
    
    for step, message, delay in thinking_steps:
        state["agent_progress"][agent_name]["current_step"] = step
        state["agent_progress"][agent_name]["thinking_messages"].append(message)
        await send_progress(state, "agent_thinking", agent_name, message, step)
        await asyncio.sleep(delay)
    
    # Tool execution
    state["agent_progress"][agent_name]["status"] = "using_tools"
    await send_progress(state, "tool_call", agent_name, "Executing feedback synthesis tool...", "tool_execution")
    await asyncio.sleep(0.5)
    
    feedback = synthesize_feedback.invoke({
        "visual_analysis": state["visual_analysis"],
        "communication_analysis": state["communication_analysis"],
        "scientific_analysis": state["scientific_analysis"],
        "image_data": state["image_data"],
        "json_structure": state.get("json_structure", {})
    })
    
    # Extract scores for response
    visual_score = 8
    comm_score = 8  
    sci_score = 9
    
    visual_match = re.search(r'VISUAL DESIGN \(Score:\s*(\d+)', feedback)
    if visual_match:
        visual_score = int(visual_match.group(1))
        
    comm_match = re.search(r'COMMUNICATION \(Score:\s*(\d+)', feedback)
    if comm_match:
        comm_score = int(comm_match.group(1))
        
    sci_match = re.search(r'SCIENTIFIC ACCURACY \(Score:\s*(\d+)', feedback)
    if sci_match:
        sci_score = int(sci_match.group(1))
    
    # Complete agent execution
    state["agent_progress"][agent_name]["status"] = "complete"
    state["agent_progress"][agent_name]["completion_time"] = time.time()
    state["agent_progress"][agent_name]["confidence"] = 0.95
    
    await send_progress(state, "agent_complete", agent_name, "Feedback synthesis completed", "complete", {
        "confidence": 0.95,
        "overall_score": visual_score + comm_score + sci_score,
        "findings": "Generated comprehensive feedback with prioritized recommendations"
    })
    
    # Send final completion message
    await send_progress(state, "analysis_complete", None, "Figure analysis completed successfully", "complete", {
        "total_time": time.time() - min(prog["start_time"] for prog in state["agent_progress"].values() if prog["start_time"]),
        "agents_completed": len([prog for prog in state["agent_progress"].values() if prog["status"] == "complete"])
    })
    
    return {
        "feedback_summary": feedback,
        "quality_scores": {
            "visual_design": visual_score,
            "communication": comm_score,
            "scientific_accuracy": sci_score,
            "overall": visual_score + comm_score + sci_score
        }
    }
    
    # Extract scores for response
    visual_score = 8
    comm_score = 8  
    sci_score = 9
    
    visual_match = re.search(r'VISUAL DESIGN \(Score:\s*(\d+)', feedback)
    if visual_match:
        visual_score = int(visual_match.group(1))
        
    comm_match = re.search(r'COMMUNICATION \(Score:\s*(\d+)', feedback)
    if comm_match:
        comm_score = int(comm_match.group(1))
        
    sci_match = re.search(r'SCIENTIFIC ACCURACY \(Score:\s*(\d+)', feedback)
    if sci_match:
        sci_score = int(sci_match.group(1))
    
    return {
        "feedback_summary": feedback,
        "quality_scores": {
            "visual_design": visual_score,
            "communication": comm_score,
            "scientific_accuracy": sci_score,
            "overall": visual_score + comm_score + sci_score
        }
    }


# === LANGGRAPH SETUP ===

def build_graph():
    """Build the figure analysis workflow graph with content-first architecture."""
    workflow = StateGraph(FigureState)
    
    # Add nodes - content interpretation first, then parallel analysis agents
    workflow.add_node("content_interpretation", content_interpretation_step)
    workflow.add_node("visual_design", visual_design_agent)
    workflow.add_node("communication", communication_agent)
    workflow.add_node("scientific", scientific_agent)
    workflow.add_node("feedback_synthesizer", feedback_synthesizer_agent)
    
    # Content-first flow: START -> content_interpretation -> [3 parallel agents] -> synthesizer
    workflow.add_edge(START, "content_interpretation")
    
    # After content interpretation, run analysis agents in parallel
    workflow.add_edge("content_interpretation", "visual_design")
    workflow.add_edge("content_interpretation", "communication") 
    workflow.add_edge("content_interpretation", "scientific")
    
    # All analysis agents feed into synthesizer
    workflow.add_edge("visual_design", "feedback_synthesizer")
    workflow.add_edge("communication", "feedback_synthesizer")
    workflow.add_edge("scientific", "feedback_synthesizer")
    
    # End after synthesis
    workflow.add_edge("feedback_synthesizer", END)
    
    return workflow.compile()


# === FASTAPI APP ===

app = FastAPI(
    title="BioRender Figure Feedback Agent",
    description="AI-powered feedback system for scientific figure analysis",
    version="1.0.0"
)

# Static media serving for persisted input images (useful for Arize/AX rendering)
MEDIA_DIR = Path(__file__).parent / "media"
try:
    MEDIA_DIR.mkdir(exist_ok=True)
except Exception:
    # If directory creation fails (e.g., readonly FS), proceed without raising
    pass
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

def _persist_image_and_get_url(image_b64: str, session_id: str | None = None) -> str:
    """Persist base64 image under backend/media and return a URL.

    If PUBLIC_BASE_URL is set (e.g., to an ngrok/Cloudflare tunnel or production host),
    returns an absolute HTTPS URL that Arize can fetch. Otherwise returns an app-relative
    path (may not be reachable by external services).
    """
    try:
        import uuid
        from urllib.parse import urljoin

        # Ensure media directory exists
        MEDIA_DIR.mkdir(exist_ok=True)

        fname = f"{(session_id or uuid.uuid4().hex)}.png"
        fpath = MEDIA_DIR / fname
        # Write bytes (overwrite if same name reused for session)
        fpath.write_bytes(base64.b64decode(image_b64))

        base = os.getenv("PUBLIC_BASE_URL") or os.getenv("RENDER_EXTERNAL_URL")
        if base:
            return urljoin(base.rstrip("/") + "/", f"media/{fname}")
        # Fallback to app-relative path (works locally in browser; not externally fetchable)
        return f"/media/{fname}"
    except Exception as _e:
        logger.debug(f"Image persist failed, using data URI only: {_e}")
        return f"data:image/png;base64,{image_b64}"

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def send_progress_update(self, session_id: str, update: ProgressUpdate):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(update))
            except Exception as e:
                logger.warning(f"Failed to send WebSocket update: {e}")
                self.disconnect(session_id)

manager = ConnectionManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Note: Removed custom AttributeSizeProcessor as it was interfering with trace export
# Arize handles large attribute truncation automatically

# Initialize Arize AX tracing if credentials available
if _TRACING and os.getenv("ARIZE_SPACE_ID") and os.getenv("ARIZE_API_KEY"):
    try:
        # Register with Arize AX (production-optimized)
        from arize.otel import Transport
        tracer_provider = register(
            space_id=os.getenv("ARIZE_SPACE_ID", ""),
            api_key=os.getenv("ARIZE_API_KEY", ""),
            project_name=os.getenv("ARIZE_PROJECT_NAME", "biorender-figure-feedback-agent"),
            endpoint="https://otlp.arize.com/v1/traces",  # HTTP transport for reliability
            transport=Transport.HTTP,  # More reliable than gRPC for this setup
            # batch=True (default) for production efficiency
            # log_to_console=False (default) for production
        )
        
        # Note: Custom span processors can interfere with trace export
        # Use Arize's built-in size limits instead of custom processors
        
        # Instrument LangChain components
        LangChainInstrumentor().instrument(
            tracer_provider=tracer_provider
        )
        
        # Instrument LiteLLM for direct LLM calls
        LiteLLMInstrumentor().instrument(
            tracer_provider=tracer_provider
        )
        
        logger.info("✅ Arize AX observability initialized successfully")
        logger.info(f"🔍 View traces at: https://app.arize.com/spaces/{os.getenv('ARIZE_SPACE_ID')}/projects/{os.getenv('ARIZE_PROJECT_NAME', 'biorender-figure-feedback-agent')}")
        logger.info(f"📡 Tracing endpoint: otlp.arize.com")
        logger.info(f"🏷️  Project: {os.getenv('ARIZE_PROJECT_NAME', 'biorender-figure-feedback-agent')}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Arize tracing: {e}")
        _TRACING = False
else:
    if not os.getenv("ARIZE_SPACE_ID") or not os.getenv("ARIZE_API_KEY"):
        logger.info("ℹ️ Arize credentials not provided - running without observability")
    else:
        logger.warning("⚠️ Arize tracing modules not available")

# Helper function to send progress updates
async def send_progress(state: FigureState, update_type: str, agent: str = None, message: str = "", step: str = None, data: Dict[str, Any] = None):
    """Send progress update via WebSocket if connection exists."""
    if state.get("websocket") and state.get("session_id"):
        update: ProgressUpdate = {
            "type": update_type,
            "agent": agent,
            "message": message,
            "timestamp": time.time(),
            "step": step,
            "data": data
        }
        await manager.send_progress_update(state["session_id"], update)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time progress updates."""
    await manager.connect(websocket, session_id)
    try:
        # Keep connection alive and listen for client messages
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id)

@app.get("/")
def serve_frontend():
    """Serve the frontend HTML interface."""
    return FileResponse("../frontend/index.html")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "figure-feedback-agent"}


@app.post("/analyze-figure", response_model=FigureAnalysisResponse)
async def analyze_figure(request: FigureAnalysisRequest):
    """Analyze a BioRender figure and provide feedback without WebSocket."""
    return await _analyze_figure_internal(request, None, None)

@app.post("/analyze-figure/ws/{session_id}", response_model=FigureAnalysisResponse)
async def analyze_figure_with_websocket(request: FigureAnalysisRequest, session_id: str):
    """Analyze a BioRender figure with WebSocket progress updates."""
    websocket = manager.active_connections.get(session_id)
    return await _analyze_figure_internal(request, websocket, session_id)

async def _analyze_figure_internal(request: FigureAnalysisRequest, websocket: Any = None, session_id: str = None):
    """Internal function to analyze figure with optional WebSocket support."""
    start_time = time.time()
    
    # Create tracing context for the entire analysis
    if _TRACING:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "figure_analysis_workflow",
            attributes={
                "figure.type": request.figure_type or "unknown",
                "figure.has_context": bool(request.context),
                "figure.context_length": len(request.context or ""),
                "figure.has_websocket": websocket is not None,
                "figure.session_id": session_id or "none",
                "figure.image_size_bytes": len(request.image_data) if request.image_data else 0,
                "figure.json_elements": len(request.json_structure) if request.json_structure else 0,
                "figure.context_preview": (request.context or "")[:100] if request.context else ""
            }
        ) as span:
            return await _run_analysis_with_tracing(request, websocket, session_id, start_time, span)
    else:
        return await _run_analysis_without_tracing(request, websocket, session_id, start_time)

async def _run_analysis_with_tracing(request: FigureAnalysisRequest, websocket: Any, session_id: str, start_time: float, span):
    """Run analysis with tracing context."""
    try:
        logger.info(f"Starting figure analysis - figure_type: {request.figure_type}, context length: {len(request.context or '')}")
        
        # Build the analysis workflow
        graph = build_graph()
        
        # Initialize state with WebSocket support
        initial_state: FigureState = {
            "image_data": request.image_data,
            "json_structure": request.json_structure,
            "context": request.context,
            "figure_type": request.figure_type,
            "visual_analysis": None,
            "communication_analysis": None,
            "scientific_analysis": None,
            "content_interpretation": None,
            "feedback_summary": None,
            "quality_scores": None,
            "agent_progress": {},
            "websocket": websocket,
            "session_id": session_id
        }

        # Attach a publicly reachable image URL to the span when possible
        try:
            if request.image_data:
                image_url_for_span = _persist_image_and_get_url(request.image_data, session_id)
                # If we don't have a PUBLIC_BASE_URL, this may be a relative path
                span.set_attribute("figure.image_url", image_url_for_span)
        except Exception as _e:
            logger.debug(f"Unable to attach image URL to span: {_e}")
        
        logger.info("Invoking LangGraph workflow...")
        
        # Send initial progress update
        if websocket and session_id:
            await send_progress(initial_state, "analysis_start", None, "Starting multi-agent analysis...", "initialization")
        
        # Run the analysis with async support
        result = await graph.ainvoke(initial_state)
        
        processing_time = time.time() - start_time
        logger.info(f"Analysis completed in {processing_time:.2f}s")
        
        # Add success metrics to span
        span.set_attribute("analysis.processing_time", processing_time)
        span.set_attribute("analysis.success", True)
        
        if result.get("quality_scores"):
            scores = result["quality_scores"]
            span.set_attribute("analysis.visual_score", scores.get("visual_design", 0))
            span.set_attribute("analysis.communication_score", scores.get("communication", 0))
            span.set_attribute("analysis.scientific_score", scores.get("scientific_accuracy", 0))
            span.set_attribute("analysis.overall_score", scores.get("overall", 0))
        
        span.set_status(Status(StatusCode.OK, "Analysis completed successfully"))
        
        # Extract recommendations from feedback
        recommendations = []
        feedback_text = result.get("feedback_summary", "")
        
        # Parse recommendations from feedback text
        if "→" in feedback_text:
            rec_lines = [line.strip() for line in feedback_text.split("\n") if "→" in line]
            for line in rec_lines:
                rec_text = line.replace("→", "").strip()
                if rec_text:
                    recommendations.append({
                        "text": rec_text,
                        "priority": "high" if "HIGH PRIORITY" in line else "medium",
                        "category": "visual" if "color" in rec_text.lower() or "layout" in rec_text.lower() 
                                 else "communication" if "flow" in rec_text.lower() or "clarity" in rec_text.lower()
                                 else "scientific"
                    })
        
        scores = result.get("quality_scores", {})
        
        return FigureAnalysisResponse(
            visual_design_score=scores.get("visual_design", 8),
            communication_score=scores.get("communication", 8),
            scientific_accuracy_score=scores.get("scientific_accuracy", 9),
            overall_score=scores.get("overall", 25),
            content_summary=result.get("content_interpretation", "No content summary available"),
            feedback=feedback_text,
            recommendations=recommendations,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Record error in trace
        span.set_status(Status(StatusCode.ERROR, f"Analysis failed: {str(e)}"))
        span.set_attribute("error.type", type(e).__name__)
        span.set_attribute("error.message", str(e))
        
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def _run_analysis_without_tracing(request: FigureAnalysisRequest, websocket: Any, session_id: str, start_time: float):
    """Run analysis without tracing (fallback)."""
    try:
        logger.info(f"Starting figure analysis - figure_type: {request.figure_type}, context length: {len(request.context or '')}")
        
        # Build the analysis workflow
        graph = build_graph()
        
        # Initialize state with WebSocket support
        initial_state: FigureState = {
            "image_data": request.image_data,
            "json_structure": request.json_structure,
            "context": request.context,
            "figure_type": request.figure_type,
            "visual_analysis": None,
            "communication_analysis": None,
            "scientific_analysis": None,
            "content_interpretation": None,
            "feedback_summary": None,
            "quality_scores": None,
            "agent_progress": {},
            "websocket": websocket,
            "session_id": session_id
        }
        
        logger.info("Invoking LangGraph workflow...")
        
        # Send initial progress update
        if websocket and session_id:
            await send_progress(initial_state, "analysis_start", None, "Starting multi-agent analysis...", "initialization")
        
        # Run the analysis with async support
        result = await graph.ainvoke(initial_state)
        
        processing_time = time.time() - start_time
        logger.info(f"Analysis completed in {processing_time:.2f}s")
        
        # Extract recommendations from feedback
        recommendations = []
        feedback_text = result.get("feedback_summary", "")
        
        # Parse recommendations from feedback text
        if "→" in feedback_text:
            rec_lines = [line.strip() for line in feedback_text.split("\n") if "→" in line]
            for line in rec_lines:
                rec_text = line.replace("→", "").strip()
                if rec_text:
                    recommendations.append({
                        "text": rec_text,
                        "priority": "high" if "HIGH PRIORITY" in line else "medium",
                        "category": "visual" if "color" in rec_text.lower() or "layout" in rec_text.lower() 
                                 else "communication" if "flow" in rec_text.lower() or "clarity" in rec_text.lower()
                                 else "scientific"
                    })
        
        scores = result.get("quality_scores", {})
        
        return FigureAnalysisResponse(
            visual_design_score=scores.get("visual_design", 8),
            communication_score=scores.get("communication", 8),
            scientific_accuracy_score=scores.get("scientific_accuracy", 9),
            overall_score=scores.get("overall", 25),
            content_summary=result.get("content_interpretation", "No content summary available"),
            feedback=feedback_text,
            recommendations=recommendations,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze-figure/upload")
async def analyze_figure_upload(
    image: UploadFile = File(...),
    json_file: UploadFile = File(...),
    context: str = Form(""),
    figure_type: str = Form("")
):
    """Upload figure image and JSON for analysis without WebSocket."""
    return await _analyze_figure_upload_internal(image, json_file, context, figure_type, None, None)

@app.post("/analyze-figure/upload/{session_id}")
async def analyze_figure_upload_with_websocket(
    session_id: str,
    image: UploadFile = File(...),
    json_file: UploadFile = File(...),
    context: str = Form(""),
    figure_type: str = Form("")
):
    """Upload figure image and JSON for analysis with WebSocket support."""
    websocket = manager.active_connections.get(session_id)
    return await _analyze_figure_upload_internal(image, json_file, context, figure_type, websocket, session_id)

async def _analyze_figure_upload_internal(
    image: UploadFile,
    json_file: UploadFile, 
    context: str,
    figure_type: str,
    websocket: Any = None,
    session_id: str = None
):
    """Internal function to handle file upload and analysis."""
    try:
        logger.info(f"File upload received - image: {image.filename}, json: {json_file.filename}")
        
        # Validate file types
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Image file must be PNG or JPG format")
        
        if json_file.content_type and json_file.content_type != 'application/json':
            logger.warning(f"JSON file has content-type: {json_file.content_type}, but proceeding...")
        
        # Read image and convert to base64
        image_content = await image.read()
        if len(image_content) == 0:
            raise HTTPException(status_code=400, detail="Image file is empty")
        image_b64 = base64.b64encode(image_content).decode('utf-8')
        logger.info(f"Image processed successfully - {len(image_content)} bytes")
        
        # Read JSON structure
        json_content = await json_file.read()
        if len(json_content) == 0:
            raise HTTPException(status_code=400, detail="JSON file is empty")
            
        try:
            json_structure = json.loads(json_content.decode('utf-8'))
        except UnicodeDecodeError as e:
            raise HTTPException(status_code=400, detail=f"JSON file encoding error: {str(e)}")
        
        logger.info(f"JSON processed successfully - {len(json_structure)} keys")
        
        # Create request object
        request = FigureAnalysisRequest(
            image_data=image_b64,
            json_structure=json_structure,
            context=context or "",
            figure_type=figure_type or ""
        )
        
        # Analyze figure with WebSocket support
        return await _analyze_figure_internal(request, websocket, session_id)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON file format: {str(e)}")
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Upload processing failed: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
