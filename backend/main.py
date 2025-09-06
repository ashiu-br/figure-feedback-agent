from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import time
import json
import base64
import logging
import traceback
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Minimal observability via Arize/OpenInference (optional)
try:
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import using_prompt_template
    _TRACING = True
except Exception:
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


# Figure state for LangGraph
class FigureState(TypedDict):
    image_data: str
    json_structure: Dict[str, Any]
    context: Optional[str]
    figure_type: Optional[str]
    visual_analysis: Optional[str]
    communication_analysis: Optional[str]
    scientific_analysis: Optional[str]
    content_interpretation: Optional[str]
    feedback_summary: Optional[str]
    quality_scores: Optional[Dict[str, int]]


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

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=[
            {"type": "text", "text": context_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
        ])
    ]
    
    # Use vision-capable model
    vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=300)
    response = vision_llm.invoke(messages)
    return response.content.strip()


@tool
def analyze_visual_design(image_data: str, json_structure: dict, context: str = "") -> str:
    """Analyze visual design aspects of the figure including color usage, layout, hierarchy, typography, and spacing."""
    
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
    
    return f"""VISUAL DESIGN ANALYSIS (Score: {color_score}/10):

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
def evaluate_communication_clarity(json_structure: dict, context: str = "", figure_type: str = "") -> str:
    """Evaluate communication clarity including logical flow, information density, and audience appropriateness."""
    
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
    
    return f"""COMMUNICATION CLARITY ANALYSIS (Score: {overall_score}/10):

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
def validate_scientific_accuracy(json_structure: dict, context: str = "", figure_type: str = "") -> str:
    """Validate scientific accuracy including nomenclature, pathway logic, and field conventions."""
    
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
    
    return f"""SCIENTIFIC ACCURACY ANALYSIS (Score: {accuracy_score}/10):

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

def visual_design_agent(state: FigureState):
    """Agent focused on visual design analysis."""
    analysis = analyze_visual_design.invoke({
        "image_data": state["image_data"],
        "json_structure": state["json_structure"],
        "context": state.get("context", "")
    })
    
    return {
        "visual_analysis": analysis
    }


def communication_agent(state: FigureState):
    """Agent focused on communication clarity."""
    analysis = evaluate_communication_clarity.invoke({
        "json_structure": state["json_structure"],
        "context": state.get("context", ""),
        "figure_type": state.get("figure_type", "")
    })
    
    return {
        "communication_analysis": analysis
    }


def scientific_agent(state: FigureState):
    """Agent focused on scientific accuracy."""
    analysis = validate_scientific_accuracy.invoke({
        "json_structure": state["json_structure"],
        "context": state.get("context", ""),
        "figure_type": state.get("figure_type", "")
    })
    
    return {
        "scientific_analysis": analysis
    }


def content_interpretation_agent(state: FigureState):
    """Agent focused on interpreting figure content and generating plain language summary."""
    interpretation = interpret_figure_content.invoke({
        "image_data": state["image_data"],
        "json_structure": state["json_structure"],
        "context": state.get("context", ""),
        "figure_type": state.get("figure_type", "")
    })
    
    return {
        "content_interpretation": interpretation
    }


def feedback_synthesizer_agent(state: FigureState):
    """Agent that synthesizes all analyses into final feedback."""
    feedback = synthesize_feedback.invoke({
        "visual_analysis": state["visual_analysis"],
        "communication_analysis": state["communication_analysis"],
        "scientific_analysis": state["scientific_analysis"],
        "image_data": state["image_data"],
        "json_structure": state["json_structure"]
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
    """Build the figure analysis workflow graph."""
    workflow = StateGraph(FigureState)
    
    # Add nodes
    workflow.add_node("visual_design", visual_design_agent)
    workflow.add_node("communication", communication_agent)
    workflow.add_node("scientific", scientific_agent)
    workflow.add_node("content_interpretation", content_interpretation_agent)
    workflow.add_node("feedback_synthesizer", feedback_synthesizer_agent)
    
    # Parallel execution: START -> [visual, communication, scientific, content_interpretation]
    workflow.add_edge(START, "visual_design")
    workflow.add_edge(START, "communication")
    workflow.add_edge(START, "scientific")
    workflow.add_edge(START, "content_interpretation")
    
    # All analyses feed into synthesizer
    workflow.add_edge("visual_design", "feedback_synthesizer")
    workflow.add_edge("communication", "feedback_synthesizer")
    workflow.add_edge("scientific", "feedback_synthesizer")
    workflow.add_edge("content_interpretation", "feedback_synthesizer")
    
    # End after synthesis
    workflow.add_edge("feedback_synthesizer", END)
    
    return workflow.compile()


# === FASTAPI APP ===

app = FastAPI(
    title="BioRender Figure Feedback Agent",
    description="AI-powered feedback system for scientific figure analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize tracing if available
if _TRACING and os.getenv("ARIZE_SPACE_ID") and os.getenv("ARIZE_API_KEY"):
    register(
        space_id=os.getenv("ARIZE_SPACE_ID", ""),
        api_key=os.getenv("ARIZE_API_KEY", ""),
    )
    LangChainInstrumentor().instrument()
    LiteLLMInstrumentor().instrument()


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
    """Analyze a BioRender figure and provide feedback."""
    start_time = time.time()
    
    try:
        logger.info(f"Starting figure analysis - figure_type: {request.figure_type}, context length: {len(request.context or '')}")
        
        # Build the analysis workflow
        graph = build_graph()
        
        # Initialize state
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
            "quality_scores": None
        }
        
        logger.info("Invoking LangGraph workflow...")
        # Run the analysis
        result = graph.invoke(initial_state)
        
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
    """Upload figure image and JSON for analysis."""
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
        
        # Analyze figure
        return await analyze_figure(request)
        
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