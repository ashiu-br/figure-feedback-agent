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
    # 1) Explicit ENV_FILE if provided
    env_file = os.getenv("ENV_FILE")
    if env_file and Path(env_file).exists():
        load_dotenv(env_file, override=False)
    # 2) backend/.env (next to this file)
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    # 3) repo root .env (one level up from backend/)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    # 4) fallback search from CWD
    load_dotenv(find_dotenv(), override=False)
except Exception:
    # Fall back silently if dotenv not available; env vars may still be set
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
    from opentelemetry import context as otel_context
    from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
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

    # Fallback when tracing not available
    otel_context = None
    _SUPPRESS_INSTRUMENTATION_KEY = None

    _TRACING = False

# Suppress instrumentation context manager
from contextlib import contextmanager

@contextmanager
def suppress_instrumentation():
    """Suppress OpenTelemetry instrumentation for the duration of this context."""
    if not _TRACING or not otel_context or not _SUPPRESS_INSTRUMENTATION_KEY:
        # Tracing disabled or not available, just yield
        yield
        return

    # Create context with suppression enabled
    ctx = otel_context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True)
    token = otel_context.attach(ctx)

    try:
        yield
    finally:
        otel_context.detach(token)

# Global image cache for session-based storage (reduces Arize payload size)
IMAGE_CACHE = {}

# LangGraph + LangChain
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from PIL import Image
import io
import re
import uuid

# V2 optimization: Direct implementation without vision engine abstraction


class FigureAnalysisRequest(BaseModel):
    image_data: str  # base64 encoded image
    context: Optional[str] = None  # Figure title, intended audience, etc.
    figure_type: Optional[str] = None  # pathway, workflow, mechanism, timeline
    vision_version: Optional[str] = None  # v1, v2, shadow - engine version override


class FigureAnalysisResponse(BaseModel):
    visual_design_score: int
    communication_score: int
    scientific_accuracy_score: int
    overall_score: int
    content_summary: str
    feedback: str
    recommendations: List[Dict[str, Any]]
    processing_time: float
    # Scientific coherence 3-level system (HIGH/MEDIUM/LOW) and rationale
    scientific_coherence_level: Optional[str] = None
    scientific_coherence_reason: Optional[str] = None


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

# Figure state for LangGraph - Vision Only
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
    vision_version: Optional[str]  # Vision engine version override
    detailed_image_description: Optional[str]  # Rich description from single vision call for text-only agent analysis


def _init_llm():
    if os.getenv("TEST_MODE"):
        class _Fake:
            def invoke(self, messages):
                class _Msg:
                    content = "Mock figure analysis feedback"
                return _Msg()
        return _Fake()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=2000)
    elif os.getenv("OPENROUTER_API_KEY"):
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.3,
        )
    else:
        raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")


llm = _init_llm()


# === VISION-ONLY ANALYSIS TOOLS ===

@tool
def analyze_visual_design(session_id: str, context: str = "", content_summary: str = "") -> str:
    """Analyze visual design aspects of the figure using vision-only approach.

    Args:
        session_id: Session identifier to lookup image data from cache
        content_summary: Plain language description of what the figure communicates (from content interpretation)
    """

    # Lookup image data from cache
    image_data = IMAGE_CACHE.get(session_id, "")

    if not image_data or os.getenv("TEST_MODE"):
        return """VISUAL DESIGN ANALYSIS (Score: 8/10):

🎨 Color Usage:
→ Visual analysis suggests balanced color palette
- Consider using brand colors or scientific publication standards

📝 Typography & Hierarchy:
→ Text elements appear well-organized
- Ensure consistent font sizes within element categories

📐 Layout & Spacing:
→ Layout appears well-balanced
- Balance information density with readability

💡 Recommendations:
→ Use high contrast for text readability
→ Maintain consistent spacing between related elements
→ Consider color-blind friendly palette for accessibility"""
    
    try:
        system_prompt = """You are an expert in scientific figure visual design. Analyze this image and evaluate:
1. Color palette effectiveness and accessibility
2. Typography and text hierarchy clarity
3. Layout balance and spacing
4. Visual hierarchy and information organization

Provide specific, actionable feedback with a score out of 10. Focus on design principles for scientific communication."""
        
        context_text = f"Context: {context or 'Scientific figure analysis'}\n\nPlease analyze the visual design aspects and provide scored feedback."
        
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
        
        vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=500)

        response = vision_llm.invoke(messages)
        return f"VISUAL DESIGN ANALYSIS:\n\n{response.content.strip()}"
        
    except Exception as e:
        logger.warning(f"Vision analysis failed: {str(e)}, using fallback")
        return """VISUAL DESIGN ANALYSIS (Score: 7/10):

🎨 Color Usage:
→ Unable to analyze colors in detail - ensure good contrast
- Consider using brand colors or scientific publication standards

📝 Typography & Hierarchy:
→ Ensure text hierarchy supports information flow
- Use consistent font sizes within element categories

📐 Layout & Spacing:
→ Review spacing for optimal readability
- Balance information density with visual clarity

💡 Recommendations:
→ Use high contrast for text readability
→ Maintain consistent spacing between related elements
→ Consider color-blind friendly palette for accessibility"""


@tool
def evaluate_communication_clarity(session_id: str, context: str = "", figure_type: str = "", content_summary: str = "") -> str:
    """Evaluate communication clarity using vision-only approach.

    Args:
        session_id: Session identifier to lookup image data from cache
        content_summary: Plain language description of what the figure communicates (from content interpretation)
    """

    # Lookup image data from cache
    image_data = IMAGE_CACHE.get(session_id, "")

    if not image_data or os.getenv("TEST_MODE"):
        return f"""COMMUNICATION CLARITY ANALYSIS (Score: 8/10):

🎯 Information Flow:
→ Logical flow appears appropriate for figure type
- Figure type: {figure_type or 'general'}

📊 Information Density:
→ Information density appears balanced
- Review if all elements serve the main message

👥 Audience Appropriateness:
→ Consider target audience expertise level
→ Use terminology appropriate for intended readers
→ Include necessary context without overwhelming detail

💡 Communication Tips:
→ Start with most important message/conclusion
→ Use visual hierarchy to guide reader attention  
→ Ensure each element contributes to main narrative
→ Consider adding brief caption or title for context"""
    
    try:
        system_prompt = """You are an expert in scientific communication. Analyze this figure and evaluate:
1. Information flow and logical sequence
2. Information density and cognitive load
3. Clarity of main message
4. Audience appropriateness

Provide specific feedback with a score out of 10. Focus on how effectively the figure communicates its intended message."""
        
        context_text = f"Context: {context or 'Scientific figure analysis'}\nFigure Type: {figure_type or 'general'}\n\nPlease analyze the communication effectiveness."
        
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
        
        vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=500)

        response = vision_llm.invoke(messages)
        return f"COMMUNICATION CLARITY ANALYSIS:\n\n{response.content.strip()}"
        
    except Exception as e:
        logger.warning(f"Vision analysis failed: {str(e)}, using fallback")
        return """COMMUNICATION CLARITY ANALYSIS (Score: 7/10):

🎯 Information Flow:
→ Review information sequence and flow indicators
- Consider adding arrows or connectors if needed

📊 Information Density:
→ Monitor information density for target audience
- Ensure all elements serve the main message

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
def validate_scientific_accuracy(session_id: str, context: str = "", figure_type: str = "", content_summary: str = "") -> str:
    """Validate scientific accuracy using vision-only approach.

    Args:
        session_id: Session identifier to lookup image data from cache
        content_summary: Plain language description of what the figure communicates (from content interpretation)
    """

    # Lookup image data from cache
    image_data = IMAGE_CACHE.get(session_id, "")

    if not image_data or os.getenv("TEST_MODE"):
        return """SCIENTIFIC ACCURACY ANALYSIS (Score: 9/10):

🔬 Nomenclature Check:
→ Scientific content appears accurate based on available information

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
    
    try:
        system_prompt = """You are a scientific expert. Analyze this figure and evaluate:
1. Scientific nomenclature and terminology accuracy
2. Biological/scientific pathway logic and relationships
3. Adherence to field conventions
4. Units, measurements, and notation correctness

Provide specific feedback with a score out of 10. Focus on scientific validity and accuracy."""
        
        context_text = f"Context: {context or 'Scientific figure analysis'}\nFigure Type: {figure_type or 'general'}\n\nPlease analyze the scientific accuracy."
        
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
        
        vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=500)

        response = vision_llm.invoke(messages)
        return f"SCIENTIFIC ACCURACY ANALYSIS:\n\n{response.content.strip()}"
        
    except Exception as e:
        logger.warning(f"Vision analysis failed: {str(e)}, using fallback")
        return """SCIENTIFIC ACCURACY ANALYSIS (Score: 8/10):

🔬 Nomenclature Check:
→ Review scientific terminology for accuracy
→ Verify naming conventions for your field

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
def interpret_figure_content(session_id: str, context: str = "", figure_type: str = "") -> str:
    """Generate a plain language summary using vision LLM.

    Args:
        session_id: Session identifier to lookup image data from cache
    """

    # Lookup image data from cache
    image_data = IMAGE_CACHE.get(session_id, "")

    if not image_data or os.getenv("TEST_MODE"):
        return f"This figure appears to be a {figure_type or 'scientific diagram'} showing biological or scientific relationships. {context or 'The figure illustrates key concepts and processes within its domain.'}"
    
    try:
        logger.info("Attempting vision-based content interpretation")
        
        system_prompt = """You are an expert at interpreting scientific figures. Look at the provided image and describe what it shows in 2-3 clear sentences that a non-expert could understand.

Focus on:
- Main biological/scientific concept being illustrated
- Key components and their relationships
- Overall process or message being communicated

Start with "This figure shows..." or "This diagram illustrates..." and be specific about what you observe visually."""

        context_text = f"""Context: {context or 'Scientific figure analysis'}
Figure Type: {figure_type or 'diagram'}

Please analyze the image and provide a plain language interpretation."""

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
        
        vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=300)

        response = vision_llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.warning(f"Vision interpretation failed: {str(e)}, using fallback")
        return f"This figure appears to be a {figure_type or 'scientific diagram'}. While detailed analysis was not possible, it likely shows important biological or scientific relationships. {context or 'The figure communicates key concepts within its domain.'}"


@tool
def generate_actionable_recommendations(visual_analysis: str, communication_analysis: str, scientific_analysis: str, content_interpretation: str, context: str = "") -> str:
    """Generate exactly 3 specific, actionable recommendations using LLM analysis of all agent findings."""

    if os.getenv("TEST_MODE"):
        return """RECOMMENDATION 1 (Priority: HIGH - Visual Design)
Issue: Too many colors creating visual confusion
Action: Reduce color palette to 3-4 colors maximum. Use blue as primary highlight, gray for secondary elements, and black for text. Reserve bright colors only for critical elements.

RECOMMENDATION 2 (Priority: MEDIUM - Communication)
Issue: Information flow is unclear
Action: Add numbered sequence indicators (1→2→3) and directional arrows between process steps. Reorganize elements left-to-right following logical sequence.

RECOMMENDATION 3 (Priority: MEDIUM - Typography)
Issue: Inconsistent text hierarchy impacting readability
Action: Standardize all element labels to size 12pt, pathway names to 14pt bold, and main title to 18pt. Ensure minimum 4px spacing between text and element borders."""

    try:
        system_prompt = """You are an expert scientific figure consultant. Based on the provided analyses, generate exactly 3 specific, actionable recommendations that the user can immediately implement.

Each recommendation must follow this exact format:

RECOMMENDATION X (Priority: HIGH/MEDIUM/LOW - Category)
Issue: [Specific problem identified]
Action: [Detailed, implementable steps with specific values/measurements]

Requirements:
- Recommendations must be specific and actionable (e.g., "Use Arial 12pt font" not "improve typography")
- Include specific values, measurements, colors, or techniques where possible
- Priority should reflect impact on figure effectiveness
- Categories: Visual Design, Communication, Scientific Accuracy, Typography, Layout
- Focus on the most impactful changes that can be made immediately
- Draw from specific issues mentioned in the analyses
- Limit each Action to 2-3 sentences maximum

Generate exactly 3 recommendations, prioritized by impact."""

        user_prompt = f"""Content Summary: {content_interpretation}

Visual Design Analysis:
{visual_analysis}

Communication Analysis:
{communication_analysis}

Scientific Analysis:
{scientific_analysis}

Context: {context or 'Scientific figure analysis'}

Please generate 3 specific, actionable recommendations based on the most critical issues identified across all analyses."""

        llm_local = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=600)

        with using_prompt_template(
            template=system_prompt,
            variables={
                "content_interpretation": content_interpretation[:200] + "..." if len(content_interpretation) > 200 else content_interpretation,
                "visual_analysis": visual_analysis[:300] + "..." if len(visual_analysis) > 300 else visual_analysis,
                "communication_analysis": communication_analysis[:300] + "..." if len(communication_analysis) > 300 else communication_analysis,
                "scientific_analysis": scientific_analysis[:300] + "..." if len(scientific_analysis) > 300 else scientific_analysis,
                "context": context,
            },
            version="recommendations-v1.0",
        ):
            response = llm_local.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

        return response.content.strip()

    except Exception as e:
        logger.warning(f"Recommendation generation failed: {e}")
        return """RECOMMENDATION 1 (Priority: HIGH - Visual Design)
Issue: Visual elements may need optimization
Action: Review color palette for consistency and ensure high contrast between text and background. Consider using a maximum of 4 colors for clarity.

RECOMMENDATION 2 (Priority: MEDIUM - Communication)
Issue: Information flow could be enhanced
Action: Add directional indicators or arrows to show sequence. Group related elements using consistent spacing and alignment.

RECOMMENDATION 3 (Priority: MEDIUM - Scientific Accuracy)
Issue: Scientific conventions should be verified
Action: Cross-reference terminology with standard databases (UniProt, KEGG). Ensure abbreviations follow field-specific conventions and add confidence indicators where appropriate."""


@tool
def synthesize_feedback(visual_analysis: str, communication_analysis: str, scientific_analysis: str, content_interpretation: str, context: str = "") -> str:
    """Synthesize all agent analyses into prioritized, actionable feedback with overall scores and specific recommendations."""

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

    # Overall assessment
    if overall_score >= 25:
        assessment = "Excellent - Publication ready"
    elif overall_score >= 22:
        assessment = "Good - Ready with minor revisions"
    elif overall_score >= 18:
        assessment = "Needs improvement - Address key issues"
    else:
        assessment = "Major revisions needed"

    # Generate 3 specific actionable recommendations
    try:
        actionable_recommendations = generate_actionable_recommendations.invoke({
            "visual_analysis": visual_analysis,
            "communication_analysis": communication_analysis,
            "scientific_analysis": scientific_analysis,
            "content_interpretation": content_interpretation,
            "context": context
        })
    except Exception as e:
        logger.warning(f"Failed to generate actionable recommendations: {e}")
        actionable_recommendations = """RECOMMENDATION 1 (Priority: HIGH - Visual Design)
Issue: Visual elements need optimization
Action: Review and improve visual hierarchy, color usage, and spacing for better clarity.

RECOMMENDATION 2 (Priority: MEDIUM - Communication)
Issue: Information flow could be enhanced
Action: Improve logical sequence and add directional indicators where needed.

RECOMMENDATION 3 (Priority: MEDIUM - Scientific Accuracy)
Issue: Scientific conventions should be verified
Action: Cross-reference terminology and ensure adherence to field standards."""

    return f"""📊 FIGURE QUALITY ASSESSMENT

🎨 VISUAL DESIGN (Score: {visual_score}/10)
{visual_analysis.split(':', 1)[1] if ':' in visual_analysis else visual_analysis}

🎯 COMMUNICATION (Score: {comm_score}/10)
{communication_analysis.split(':', 1)[1] if ':' in communication_analysis else communication_analysis}

🔬 SCIENTIFIC ACCURACY (Score: {sci_score}/10)
{scientific_analysis.split(':', 1)[1] if ':' in scientific_analysis else scientific_analysis}

📈 OVERALL IMPACT SCORE: {overall_score}/30 ({assessment})

🎯 ACTIONABLE RECOMMENDATIONS:
{actionable_recommendations}

💡 IMPLEMENTATION PRIORITY:
1. Address HIGH priority recommendations first
2. Implement MEDIUM priority recommendations for enhancement
3. Validate changes against scientific standards
4. Review overall visual consistency and polish"""


# === AGENT IMPLEMENTATIONS ===

@tool
def evaluate_scientific_coherence(description: str, context: str = "", figure_type: str = "") -> str:
    """Evaluate whether the scientific description is conceptually coherent using 3-level system.

    Returns a structured text block including:
    - COHERENCE_LEVEL: HIGH/MEDIUM/LOW
    - Primary Issue: <short phrase if not HIGH>
    - Specific Problems: <bulleted or comma-separated list>
    - Reason: <one sentence explanation>
    """
    if not description or os.getenv("TEST_MODE"):
        return (
            "COHERENCE_LEVEL: HIGH\n"
            "Primary Issue: None\n"
            "Specific Problems: None\n"
            "Reason: Concepts align at the same level and flow logically."
        )

    system_prompt = (
        "You are a rigorous scientific reviewer focusing on conceptual coherence.\n\n"
        "Task: Evaluate whether a scientific description is coherent and logically sound using a 3-level system.\n\n"
        "Analyze for:\n"
        "1. Internal Consistency: Do the concepts flow logically from one to another?\n"
        "2. Conceptual Appropriateness: Are different scientific concepts meaningfully connected?\n"
        "3. Technical Accuracy: Are the scientific methods/concepts described correctly?\n"
        "4. Scope Alignment: Do all elements belong at the same level of analysis?\n\n"
        "Classification:\n"
        "- HIGH: Concepts flow logically, relationships are clear and justified, no significant gaps\n"
        "- MEDIUM: Generally coherent with minor gaps or unclear connections that don't undermine the overall logic\n"
        "- LOW: Major logical inconsistencies, disconnected concepts, or confusing relationships that significantly undermine scientific validity\n\n"
        "Provide assessment in this exact format:\n"
        "- COHERENCE_LEVEL: [HIGH/MEDIUM/LOW]\n"
        "- Primary Issue: [short phrase or 'None' if HIGH]\n"
        "- Specific Problems: [list main issues or 'None' if HIGH]\n"
        "- Reason: [one sentence explaining the classification]\n\n"
        "Be critical and avoid hand-waving."
    )

    user_prompt = (
        f"Description:\n{description}\n\n"
        f"Context: {context or 'Scientific figure'}\n"
        f"Figure Type: {figure_type or 'general'}"
    )

    try:
        llm_local = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=500)
        # Track template for Arize/OpenInference to separate template vs variables
        with using_prompt_template(
            template=system_prompt,
            variables={
                "description": (description[:500] + "...") if len(description) > 500 else description,
                "context": context,
                "figure_type": figure_type,
            },
            version="coherence-v2.0",
        ):
            response = llm_local.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
        return response.content.strip()
    except Exception as e:
        logger.warning(f"Coherence evaluation failed: {e}")
        return (
            "COHERENCE_LEVEL: LOW\n"
            "Primary Issue: Unable to evaluate coherence\n"
            "Specific Problems: Evaluation error\n"
            "Reason: Automatic fallback due to evaluation error."
        )


def calculate_coherence_penalty(coherence_level: str) -> int:
    """Calculate penalty points based on coherence level.

    Args:
        coherence_level: HIGH, MEDIUM, or LOW

    Returns:
        Penalty points to subtract from scientific accuracy score
    """
    coherence_level = coherence_level.upper().strip()

    penalty_map = {
        "HIGH": 0,     # No penalty
        "MEDIUM": 3,   # Moderate penalty
        "LOW": 7       # Severe penalty
    }

    return penalty_map.get(coherence_level, 7)  # Default to severe penalty for unknown levels


def extract_score_from_analysis(analysis_text: str) -> int:
    """Extract score from analysis text like 'Score: 8/10'.

    Args:
        analysis_text: Text containing score pattern

    Returns:
        Extracted score or 8 as default
    """
    score_match = re.search(r'Score:\s*(\d+)(?:/10)?', analysis_text, re.IGNORECASE)
    if score_match:
        return int(score_match.group(1))
    return 8  # Default score


def update_analysis_score(analysis_text: str, new_score: int) -> str:
    """Update the score in analysis text.

    Args:
        analysis_text: Original analysis text
        new_score: New score to replace existing score

    Returns:
        Updated analysis text with new score
    """
    # Update score in formats like "Score: 8/10" or "Score: 8"
    updated_text = re.sub(
        r'(Score:\s*)(\d+)(/10)?',
        rf'\g<1>{new_score}\g<3>',
        analysis_text,
        flags=re.IGNORECASE
    )

    # If no score pattern found, add score to beginning
    if not re.search(r'Score:\s*\d+', analysis_text, re.IGNORECASE):
        if analysis_text.startswith("SCIENTIFIC ACCURACY ANALYSIS"):
            updated_text = f"SCIENTIFIC ACCURACY ANALYSIS (Score: {new_score}/10):\n\n{analysis_text.split(':', 1)[1] if ':' in analysis_text else analysis_text}"
        else:
            updated_text = f"SCIENTIFIC ACCURACY ANALYSIS (Score: {new_score}/10):\n\n{analysis_text}"

    return updated_text

async def visual_design_agent(state: FigureState):
    """Agent focused on visual design analysis with thinking simulation."""
    agent_name = "visual_design"
    
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
    ]
    
    for step, message, delay in thinking_steps:
        state["agent_progress"][agent_name]["current_step"] = step
        state["agent_progress"][agent_name]["thinking_messages"].append(message)
        await send_progress(state, "agent_thinking", agent_name, message, step)
        await asyncio.sleep(delay)
    
    state["agent_progress"][agent_name]["status"] = "using_tools"

    # Always define content_summary first (scope fix)
    content_summary = state.get("content_interpretation", "")

    # Check if detailed description is available (V2 approach)
    detailed_description = state.get("detailed_image_description", "")

    if detailed_description:
        # V2: Use detailed description with text-only LLM call
        logger.info("Visual design: Using V2 text-only analysis with detailed description")
        await send_progress(state, "tool_call", agent_name, "Executing text-based visual design analysis (V2)", "tool_execution")
        await asyncio.sleep(0.3)

        try:
            system_prompt = """You are an expert in scientific figure visual design. Based on the comprehensive image description provided, analyze the visual design aspects and provide specific feedback.

Evaluate:
1. Color palette effectiveness and accessibility
2. Typography and text hierarchy clarity
3. Layout balance and spacing
4. Visual hierarchy and information organization

Provide specific, actionable feedback with a score out of 10. Focus on design principles for scientific communication.

Format your response as:
VISUAL DESIGN ANALYSIS (Score: X/10):

[Your detailed analysis with specific recommendations]"""

            user_prompt = f"""Context: {state.get("context", "Scientific figure analysis")}

Detailed Image Description:
{detailed_description}

Based on this comprehensive description, provide focused visual design analysis and scoring."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            text_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, max_tokens=600)
            response = text_llm.invoke(messages)
            analysis = f"VISUAL DESIGN ANALYSIS:\n\n{response.content.strip()}"

            logger.info("Visual design: V2 text-only analysis completed")

        except Exception as e:
            logger.error(f"Visual design V2 analysis failed: {str(e)}, using fallback")
            analysis = """VISUAL DESIGN ANALYSIS (Score: 7/10):

🎨 Color Usage:
→ Review color palette based on description provided
- Ensure appropriate contrast ratios for readability

📝 Typography & Hierarchy:
→ Text hierarchy appears structured based on analysis
- Maintain readable font sizes and consistent styling

📐 Layout & Spacing:
→ Layout organization seems balanced from description
- Continue focusing on professional visual presentation"""

    else:
        # V1: Fallback to individual vision tool call
        logger.info("Visual design: No detailed description found, using individual vision call (V1)")
        await send_progress(state, "tool_call", agent_name, "Executing visual design analysis tool...", "tool_execution")
        await asyncio.sleep(0.5)

        analysis = analyze_visual_design.invoke({
            "session_id": state.get("session_id", ""),
            "context": state.get("context", ""),
            "content_summary": content_summary
        })
    
    state["agent_progress"][agent_name]["status"] = "complete"
    state["agent_progress"][agent_name]["completion_time"] = time.time()
    state["agent_progress"][agent_name]["confidence"] = 0.85
    
    await send_progress(state, "agent_complete", agent_name, "Visual design analysis completed", "complete")
    
    return {"visual_analysis": analysis}


async def communication_agent(state: FigureState):
    """Agent focused on communication clarity with thinking simulation."""
    agent_name = "communication"
    
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
    
    thinking_steps = [
        ("flow_analysis", "Examining logical flow and information sequence...", 1.1),
        ("density_check", "Evaluating information density and cognitive load...", 0.9),
        ("audience_assessment", "Assessing appropriateness for target audience...", 1.2),
    ]
    
    for step, message, delay in thinking_steps:
        state["agent_progress"][agent_name]["current_step"] = step
        state["agent_progress"][agent_name]["thinking_messages"].append(message)
        await send_progress(state, "agent_thinking", agent_name, message, step)
        await asyncio.sleep(delay)
    
    state["agent_progress"][agent_name]["status"] = "using_tools"

    # Always define content_summary first (scope fix)
    content_summary = state.get("content_interpretation", "")

    # Check if detailed description is available (V2 approach)
    detailed_description = state.get("detailed_image_description", "")

    if detailed_description:
        # V2: Use detailed description with text-only LLM call
        logger.info("Communication: Using V2 text-only analysis with detailed description")
        await send_progress(state, "tool_call", agent_name, "Executing text-based communication clarity analysis (V2)", "tool_execution")
        await asyncio.sleep(0.3)

        try:
            system_prompt = """You are an expert in scientific communication. Based on the comprehensive image description provided, evaluate the communication effectiveness.

Evaluate:
1. Information flow and logical sequence
2. Information density and cognitive load
3. Clarity of main message
4. Audience appropriateness

Provide specific feedback with a score out of 10. Focus on how effectively the figure communicates its intended message.

Format your response as:
COMMUNICATION CLARITY ANALYSIS (Score: X/10):

[Your detailed analysis with specific recommendations]"""

            user_prompt = f"""Context: {state.get("context", "Scientific figure analysis")}
Figure Type: {state.get("figure_type", "general")}

Detailed Image Description:
{detailed_description}

Based on this comprehensive description, provide focused communication effectiveness analysis and scoring."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            text_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, max_tokens=600)
            response = text_llm.invoke(messages)
            analysis = f"COMMUNICATION CLARITY ANALYSIS:\n\n{response.content.strip()}"

            logger.info("Communication: V2 text-only analysis completed")

        except Exception as e:
            logger.error(f"Communication V2 analysis failed: {str(e)}, using fallback")
            analysis = f"""COMMUNICATION CLARITY ANALYSIS (Score: 7/10):

🎯 Information Flow:
→ Information sequence appears logical based on description
- Figure type: {state.get("figure_type", "general")}

📊 Information Density:
→ Complexity level seems appropriate for scientific content
- Balance maintained between detail and clarity

👥 Audience Appropriateness:
→ Content appears suitable for intended scientific audience
- Terminology and detail level seem well-matched"""

    else:
        # V1: Fallback to individual vision tool call
        logger.info("Communication: No detailed description found, using individual vision call (V1)")
        await send_progress(state, "tool_call", agent_name, "Executing communication clarity tool...", "tool_execution")
        await asyncio.sleep(0.4)

        analysis = evaluate_communication_clarity.invoke({
            "session_id": state.get("session_id", ""),
            "context": state.get("context", ""),
            "figure_type": state.get("figure_type", ""),
            "content_summary": content_summary
        })
    
    state["agent_progress"][agent_name]["status"] = "complete"
    state["agent_progress"][agent_name]["completion_time"] = time.time()
    state["agent_progress"][agent_name]["confidence"] = 0.78
    
    await send_progress(state, "agent_complete", agent_name, "Communication analysis completed", "complete")
    
    return {"communication_analysis": analysis}


async def scientific_agent(state: FigureState):
    """Agent focused on scientific accuracy with thinking simulation."""
    agent_name = "scientific"
    
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
    
    thinking_steps = [
        ("nomenclature_check", "Validating scientific nomenclature and terminology...", 1.3),
        ("pathway_logic", "Examining pathway logic and biological accuracy...", 1.5),
        ("convention_review", "Checking adherence to field conventions...", 1.0),
    ]
    
    for step, message, delay in thinking_steps:
        state["agent_progress"][agent_name]["current_step"] = step
        state["agent_progress"][agent_name]["thinking_messages"].append(message)
        await send_progress(state, "agent_thinking", agent_name, message, step)
        await asyncio.sleep(delay)
    
    state["agent_progress"][agent_name]["status"] = "using_tools"

    # Always define content_summary first (scope fix)
    content_summary = state.get("content_interpretation", "")

    # Check if detailed description is available (V2 approach)
    detailed_description = state.get("detailed_image_description", "")

    if detailed_description:
        # V2: Use detailed description with text-only LLM call
        logger.info("Scientific: Using V2 text-only analysis with detailed description")
        await send_progress(state, "tool_call", agent_name, "Executing text-based scientific accuracy analysis (V2)", "tool_execution")
        await asyncio.sleep(0.3)

        try:
            system_prompt = """You are a scientific expert. Based on the comprehensive image description provided, validate the scientific accuracy.

Evaluate:
1. Scientific nomenclature and terminology accuracy
2. Biological/scientific pathway logic and relationships
3. Adherence to field conventions
4. Units, measurements, and notation correctness

Provide specific feedback with a score out of 10. Focus on scientific validity and accuracy.

Format your response as:
SCIENTIFIC ACCURACY ANALYSIS (Score: X/10):

[Your detailed analysis with specific recommendations]"""

            user_prompt = f"""Context: {state.get("context", "Scientific figure analysis")}
Figure Type: {state.get("figure_type", "general")}

Detailed Image Description:
{detailed_description}

Based on this comprehensive description, provide focused scientific accuracy analysis and scoring."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            text_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, max_tokens=600)
            response = text_llm.invoke(messages)
            analysis = f"SCIENTIFIC ACCURACY ANALYSIS:\n\n{response.content.strip()}"

            logger.info("Scientific: V2 text-only analysis completed")

        except Exception as e:
            logger.error(f"Scientific V2 analysis failed: {str(e)}, using fallback")
            analysis = f"""SCIENTIFIC ACCURACY ANALYSIS (Score: 8/10):

🔬 Nomenclature Check:
→ Scientific terminology appears appropriate based on description
- Figure type: {state.get("figure_type", "general")}

🧬 Pathway Logic:
→ Relationships and processes seem logically structured
→ Temporal sequences appear coherent

📏 Standards Compliance:
→ Content appears to follow scientific conventions
→ Recommend verification against field-specific standards"""

    else:
        # V1: Fallback to individual vision tool call
        logger.info("Scientific: No detailed description found, using individual vision call (V1)")
        await send_progress(state, "tool_call", agent_name, "Executing scientific accuracy validation tool...", "tool_execution")
        await asyncio.sleep(0.6)

        analysis = validate_scientific_accuracy.invoke({
            "session_id": state.get("session_id", ""),
            "context": state.get("context", ""),
            "figure_type": state.get("figure_type", ""),
            "content_summary": content_summary
        })

    # Scientific conceptual coherence (based on generated content interpretation)
    coherence_block = evaluate_scientific_coherence.invoke({
        "description": content_summary,
        "context": state.get("context", ""),
        "figure_type": state.get("figure_type", ""),
    })

    # Parse coherence level and apply penalty to scientific accuracy score
    coherence_level = "HIGH"  # Default
    coherence_reason = None
    try:
        # Extract coherence level (HIGH/MEDIUM/LOW)
        level_match = re.search(r"COHERENCE_LEVEL:\s*(\w+)", coherence_block, re.IGNORECASE)
        if level_match:
            coherence_level = level_match.group(1).strip().upper()

        # Extract reason
        reason_match = re.search(r"Reason:\s*(.+)", coherence_block, re.IGNORECASE)
        if reason_match:
            coherence_reason = reason_match.group(1).strip()
        else:
            # Fallback to primary issue as reason if present
            issue_match = re.search(r"Primary\s*Issue:\s*(.+)", coherence_block, re.IGNORECASE)
            if issue_match and issue_match.group(1).strip().lower() != "none":
                coherence_reason = issue_match.group(1).strip()

        # Trim reason if overly long
        if coherence_reason and len(coherence_reason) > 240:
            coherence_reason = coherence_reason[:237] + "..."

        # Apply coherence penalty to scientific accuracy score
        original_score = extract_score_from_analysis(analysis)
        penalty = calculate_coherence_penalty(coherence_level)
        adjusted_score = max(1, original_score - penalty)  # Minimum score of 1

        if penalty > 0:
            logger.info(f"Scientific: Applied coherence penalty - {coherence_level} coherence reduces score from {original_score} to {adjusted_score} (-{penalty})")
            analysis = update_analysis_score(analysis, adjusted_score)
        else:
            logger.info(f"Scientific: No coherence penalty - {coherence_level} coherence maintains score at {original_score}")

    except Exception as e:
        logger.warning(f"Coherence penalty calculation failed: {e}")

    combined = f"{analysis}\n\nConceptual Coherence Check:\n{coherence_block}"
    
    state["agent_progress"][agent_name]["status"] = "complete"
    state["agent_progress"][agent_name]["completion_time"] = time.time()
    state["agent_progress"][agent_name]["confidence"] = 0.92
    
    await send_progress(state, "agent_complete", agent_name, "Scientific accuracy validation completed", "complete")
    
    result: Dict[str, Any] = {"scientific_analysis": combined}
    # Store coherence level as string for response (HIGH/MEDIUM/LOW)
    result["scientific_coherence_level"] = coherence_level
    if coherence_reason:
        result["scientific_coherence_reason"] = coherence_reason
    return result


async def content_interpretation_step(state: FigureState):
    """Content interpretation step - V2 uses single vision call for detailed description."""

    await send_progress(state, "agent_start", "content_interpretation", "Interpreting figure content...", "content_analysis")

    # Get configuration
    vision_version = state.get("vision_version") or os.getenv("VISION_VERSION", "v1")
    session_id = state.get("session_id", "")
    image_data = IMAGE_CACHE.get(session_id, "")

    if vision_version == "v2" and image_data and not os.getenv("TEST_MODE"):
        try:
            # V2: Single comprehensive vision call for detailed description
            logger.info("V2: Making single vision call for comprehensive detailed description")

            system_prompt = """You are an expert scientific figure analyst. Analyze this image comprehensively and provide TWO outputs in this EXACT format:

DETAILED ANALYSIS:
[Provide extensive, technical description covering ALL aspects:]

VISUAL ELEMENTS:
- Specific colors used (mention hex codes if identifiable), color palette assessment
- Typography details: font sizes, hierarchy levels, text positioning, readability issues
- Layout specifics: spacing measurements, alignment, element positioning, visual balance
- Design quality: professional polish, consistency issues, accessibility considerations

COMMUNICATION ASPECTS:
- Information flow: how information is sequenced and organized
- Information density: cognitive load assessment, complexity level analysis
- Message clarity: how well the main message is communicated
- Audience considerations: appropriateness for different expertise levels

SCIENTIFIC CONTENT:
- Terminology: specific scientific terms used, nomenclature accuracy
- Relationships: biological/scientific pathways, cause-effect relationships shown
- Process logic: temporal sequences, mechanism validity, logical flow
- Standards: adherence to field conventions, potential accuracy issues

HUMAN SUMMARY:
[Write a concise 2-3 sentence summary that describes what this figure shows and its main message in plain language that anyone can understand.]

IMPORTANT: You MUST include both the "DETAILED ANALYSIS:" and "HUMAN SUMMARY:" section headers exactly as shown above."""

            context_text = f"""Context: {state.get("context", "Scientific figure analysis")}
Figure Type: {state.get("figure_type", "general")}

Please provide comprehensive detailed analysis followed by human-readable summary."""

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

            vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=2000)
            response = vision_llm.invoke(messages)
            full_response = response.content.strip()

            # Extract detailed description and human summary
            if "HUMAN SUMMARY:" in full_response:
                parts = full_response.split("HUMAN SUMMARY:")
                detailed_description = parts[0].strip()
                human_summary = parts[1].strip() if len(parts) > 1 else "Comprehensive analysis completed."
            else:
                # Better fallback - try to extract a meaningful summary from the response
                detailed_description = full_response

                # Try to find any descriptive content about what the figure shows
                lines = full_response.split('\n')
                potential_summary = None

                for line in lines:
                    line = line.strip()
                    if any(starter in line.lower() for starter in ['this figure', 'the figure', 'this diagram', 'the diagram', 'this image', 'shows', 'illustrates', 'demonstrates', 'depicts']):
                        potential_summary = line
                        break

                if potential_summary:
                    human_summary = potential_summary
                else:
                    # Last resort - generate a basic summary from available context
                    figure_type = state.get("figure_type", "scientific diagram")
                    context_info = state.get("context", "")
                    if context_info:
                        human_summary = f"This {figure_type} presents {context_info.lower()} with scientific information and analysis components."
                    else:
                        human_summary = f"This {figure_type} shows scientific concepts and relationships for analysis and understanding."

            logger.info(f"V2: Generated detailed description ({len(detailed_description)} chars) and summary ({len(human_summary)} chars)")

            await send_progress(state, "agent_complete", "content_interpretation",
                              "V2: Single vision call completed - detailed description ready for text-only agent analysis", "complete", {
                "findings": "Generated comprehensive description for agent text-only analysis",
                "vision_calls": 1,
                "detailed_description_length": len(detailed_description)
            })

            return {
                "content_interpretation": human_summary,
                "detailed_image_description": detailed_description
            }

        except Exception as e:
            logger.error(f"V2: Comprehensive vision call failed: {str(e)}, falling back to V1 approach")

    # V1 or fallback: Use original individual interpretation call
    logger.info(f"Using V1 approach for content interpretation (version: {vision_version})")

    interpretation = interpret_figure_content.invoke({
        "session_id": session_id,
        "context": state.get("context", ""),
        "figure_type": state.get("figure_type", "")
    })

    await send_progress(state, "agent_complete", "content_interpretation",
                      "Content interpretation completed (V1 approach)", "complete", {
        "findings": "Used individual vision call for content interpretation"
    })

    return {
        "content_interpretation": interpretation
    }


async def feedback_synthesizer_agent(state: FigureState):
    """Agent that synthesizes all analyses into final feedback."""
    agent_name = "feedback_synthesizer"
    
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
    
    thinking_steps = [
        ("analysis_review", "Reviewing all agent analyses and findings...", 0.8),
        ("priority_ranking", "Ranking recommendations by priority and impact...", 1.0),
        ("synthesis", "Synthesizing comprehensive feedback report...", 1.2),
    ]
    
    for step, message, delay in thinking_steps:
        state["agent_progress"][agent_name]["current_step"] = step
        state["agent_progress"][agent_name]["thinking_messages"].append(message)
        await send_progress(state, "agent_thinking", agent_name, message, step)
        await asyncio.sleep(delay)
    
    state["agent_progress"][agent_name]["status"] = "using_tools"
    await send_progress(state, "tool_call", agent_name, "Executing feedback synthesis tool...", "tool_execution")
    await asyncio.sleep(0.5)
    
    feedback = synthesize_feedback.invoke({
        "visual_analysis": state["visual_analysis"],
        "communication_analysis": state["communication_analysis"],
        "scientific_analysis": state["scientific_analysis"],
        "content_interpretation": state.get("content_interpretation", ""),
        "context": state.get("context", "")
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
    
    state["agent_progress"][agent_name]["status"] = "complete"
    state["agent_progress"][agent_name]["completion_time"] = time.time()
    state["agent_progress"][agent_name]["confidence"] = 0.95
    
    await send_progress(state, "agent_complete", agent_name, "Feedback synthesis completed", "complete")
    await send_progress(state, "analysis_complete", None, "Figure analysis completed successfully", "complete")
    
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
    title="BioRender Figure Feedback Agent - Vision Only",
    description="AI-powered feedback system for scientific figure analysis using vision models",
    version="1.0.0"
)

# Static media serving for persisted input images
MEDIA_DIR = Path(__file__).parent / "media"
try:
    MEDIA_DIR.mkdir(exist_ok=True)
except Exception:
    pass
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

def _persist_image_and_get_url(image_b64: str, session_id: str | None = None) -> str:
    """Persist base64 image under backend/media and return a URL.

    Uses PUBLIC_BASE_URL when present to form an absolute URL for external fetchers.
    Falls back to an app-relative path if not set.
    """
    try:
        import uuid
        from urllib.parse import urljoin

        MEDIA_DIR.mkdir(exist_ok=True)
        fname = f"{(session_id or uuid.uuid4().hex)}.png"
        fpath = MEDIA_DIR / fname
        fpath.write_bytes(base64.b64decode(image_b64))

        base = os.getenv("PUBLIC_BASE_URL") or os.getenv("RENDER_EXTERNAL_URL")
        if base:
            return urljoin(base.rstrip("/") + "/", f"media/{fname}")
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
            project_name=os.getenv("ARIZE_PROJECT_NAME", "figure-feedback-agent"),
            endpoint="https://otlp.arize.com/v1/traces",  # HTTP transport for reliability
            transport=Transport.HTTP,  # More reliable than gRPC for this setup
            log_to_console=True  # This was key for the working 8:27 PM trace!
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
        
        logger.info("✅ Arize AX observability initialized successfully (Vision-Only)")
        # Ensure the project name printed matches the one used for registration
        proj = os.getenv('ARIZE_PROJECT_NAME', 'figure-feedback-agent')
        space = os.getenv('ARIZE_SPACE_ID')
        logger.info(f"🔍 View traces at: https://app.arize.com/spaces/{space}/projects/{proj}")
        logger.info("📡 Tracing endpoint: otlp.arize.com")
        logger.info(f"🏷️  Project: {proj}")
        
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
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id)


@app.get("/")
def serve_frontend():
    """Serve the vision-only frontend HTML interface."""
    return FileResponse("../frontend/index_vision_only.html")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "figure-feedback-agent-vision-only"}


@app.post("/analyze-figure", response_model=FigureAnalysisResponse)
async def analyze_figure(request: FigureAnalysisRequest):
    """Analyze a figure using vision-only approach without WebSocket."""
    return await _analyze_figure_internal(request, None, None)


@app.post("/analyze-figure/ws/{session_id}", response_model=FigureAnalysisResponse)
async def analyze_figure_with_websocket(request: FigureAnalysisRequest, session_id: str):
    """Analyze a figure using vision-only approach with WebSocket progress updates."""
    websocket = manager.active_connections.get(session_id)
    return await _analyze_figure_internal(request, websocket, session_id)


async def _analyze_figure_internal(request: FigureAnalysisRequest, websocket: Any = None, session_id: Optional[str] = None):
    """Internal function to analyze figure with optional WebSocket support."""
    start_time = time.time()

    try:
        logger.info(f"Starting vision-only figure analysis - figure_type: {request.figure_type}, context length: {len(request.context or '')}")

        # Generate session_id if not provided (for non-WebSocket requests)
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Generated session_id for analysis: {session_id}")

        graph = build_graph()

        # Initialize state with WebSocket support - Vision Only
        initial_state: FigureState = {
            "image_data": f"[CACHED:{session_id}]",  # Reference instead of 400KB+ base64 payload
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
            "session_id": session_id,  # Now always a valid string
            "vision_version": request.vision_version,  # Vision engine version override
            "detailed_image_description": None
        }

        # Store image data in cache to avoid sending large base64 payloads to Arize
        # session_id is now guaranteed to be a valid string
        IMAGE_CACHE[session_id] = request.image_data
        
        logger.info("Invoking LangGraph workflow...")
        
        if websocket and session_id:
            await send_progress(initial_state, "analysis_start", None, "Starting multi-agent analysis...", "initialization")
        
        result = await graph.ainvoke(initial_state)
        
        processing_time = time.time() - start_time
        logger.info(f"Analysis completed in {processing_time:.2f}s")
        
        # Proactively flush spans so they appear promptly in Arize
        try:
            if _TRACING:
                provider = trace.get_tracer_provider()
                flush_result = getattr(provider, "force_flush", lambda: None)()
                logger.info("Tracing: force_flush invoked after request")
        except Exception as _e:
            logger.debug(f"Tracing flush skipped: {_e}")
        
        # Extract structured recommendations from feedback
        recommendations = []
        feedback_text = result.get("feedback_summary", "")

        # Parse new structured recommendation format
        if "RECOMMENDATION" in feedback_text:
            # Split by RECOMMENDATION blocks
            rec_blocks = re.split(r'RECOMMENDATION \d+', feedback_text)

            for block in rec_blocks[1:]:  # Skip first empty block
                if not block.strip():
                    continue

                # Extract priority and category from first line
                first_line_match = re.search(r'\(Priority:\s*(HIGH|MEDIUM|LOW)\s*-\s*([^)]+)\)', block)
                priority = "medium"  # default
                category = "general"  # default

                if first_line_match:
                    priority = first_line_match.group(1).lower()
                    category = first_line_match.group(2).strip().lower()
                    # Normalize category names
                    if "visual" in category or "design" in category:
                        category = "visual"
                    elif "communication" in category:
                        category = "communication"
                    elif "scientific" in category:
                        category = "scientific"
                    elif "typography" in category:
                        category = "visual"  # Typography is visual design
                    elif "layout" in category:
                        category = "visual"  # Layout is visual design

                # Extract issue and action
                issue_match = re.search(r'Issue:\s*(.+?)(?=\nAction:|$)', block, re.DOTALL)
                action_match = re.search(r'Action:\s*(.+?)(?=\n\n|\nRECOMMENDATION|$)', block, re.DOTALL)

                issue = issue_match.group(1).strip() if issue_match else "Issue not specified"
                action = action_match.group(1).strip() if action_match else "Action not specified"

                # Clean up text (remove extra whitespace/newlines)
                issue = ' '.join(issue.split())
                action = ' '.join(action.split())

                recommendations.append({
                    "text": f"{issue} - {action}",
                    "priority": priority,
                    "category": category,
                    "issue": issue,
                    "action": action
                })

        # Fallback to old format if new format not found
        elif "→" in feedback_text:
            rec_lines = [line.strip() for line in feedback_text.split("\n") if "→" in line]
            for line in rec_lines:
                rec_text = line.replace("→", "").strip()
                if rec_text:
                    recommendations.append({
                        "text": rec_text,
                        "priority": "high" if "HIGH PRIORITY" in line else "medium",
                        "category": "visual" if "color" in rec_text.lower() or "layout" in rec_text.lower()
                                 else "communication" if "flow" in rec_text.lower() or "clarity" in rec_text.lower()
                                 else "scientific",
                        "issue": rec_text,
                        "action": rec_text
                    })
        
        scores = result.get("quality_scores", {})
        coherence_level = result.get("scientific_coherence_level")
        coherence_reason = result.get("scientific_coherence_reason")

        return FigureAnalysisResponse(
            visual_design_score=scores.get("visual_design", 8),
            communication_score=scores.get("communication", 8),
            scientific_accuracy_score=scores.get("scientific_accuracy", 9),
            overall_score=scores.get("overall", 25),
            content_summary=result.get("content_interpretation", "No content summary available"),
            feedback=feedback_text,
            recommendations=recommendations,
            processing_time=processing_time,
            scientific_coherence_level=coherence_level,
            scientific_coherence_reason=coherence_reason
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze-figure/upload")
async def analyze_figure_upload(
    image: UploadFile = File(...),
    context: str = Form(""),
    figure_type: str = Form("")
):
    """Upload figure image for vision-only analysis without WebSocket."""
    return await _analyze_figure_upload_internal(image, context, figure_type, None, None)


@app.post("/analyze-figure/upload/{session_id}")
async def analyze_figure_upload_with_websocket(
    session_id: str,
    image: UploadFile = File(...),
    context: str = Form(""),
    figure_type: str = Form("")
):
    """Upload figure image for vision-only analysis with WebSocket support."""
    websocket = manager.active_connections.get(session_id)
    return await _analyze_figure_upload_internal(image, context, figure_type, websocket, session_id)


async def _analyze_figure_upload_internal(
    image: UploadFile,
    context: str,
    figure_type: str,
    websocket: Any = None,
    session_id: Optional[str] = None
):
    """Internal function to handle file upload and analysis - Vision Only."""
    try:
        logger.info(f"Vision-only file upload received - image: {image.filename}")
        
        # Validate file types
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Image file must be PNG or JPG format")
        
        # Read image and convert to base64
        image_content = await image.read()
        if len(image_content) == 0:
            raise HTTPException(status_code=400, detail="Image file is empty")
        image_b64 = base64.b64encode(image_content).decode('utf-8')
        logger.info(f"Image processed successfully - {len(image_content)} bytes")
        
        # Create request object - Vision Only
        request = FigureAnalysisRequest(
            image_data=image_b64,
            context=context or "",
            figure_type=figure_type or ""
        )
        
        # Analyze figure with WebSocket support
        return await _analyze_figure_internal(request, websocket, session_id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload processing failed: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
