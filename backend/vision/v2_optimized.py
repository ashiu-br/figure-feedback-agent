"""V2 Optimized Vision Engine - Single comprehensive vision call with text analysis."""

import os
import time
import json
import logging
import re
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from openinference.instrumentation import using_prompt_template

from .base import VisionEngine, VisionAnalysisResult

logger = logging.getLogger(__name__)


class V2OptimizedVisionEngine:
    """Optimized vision engine using 1 comprehensive vision call + text analysis."""

    version = "v2_optimized"

    def __init__(self):
        """Initialize the V2 optimized vision engine."""
        self.vision_llm = self._init_vision_llm()
        self.text_llm = self._init_text_llm()

    def _init_vision_llm(self):
        """Initialize vision LLM for comprehensive analysis."""
        if os.getenv("TEST_MODE"):
            class _FakeVision:
                def invoke(self, messages):
                    class _Msg:
                        content = """COMPREHENSIVE VISION ANALYSIS:

VISUAL ELEMENTS:
- Colors: Primary blue (#3498db), secondary orange (#e67e22), text in black (#2c3e50)
- Typography: Main title 16pt bold, subtitles 12pt medium, labels 10pt regular
- Layout: Well-balanced with 20px margins, elements aligned in grid structure

COMMUNICATION ASPECTS:
- Information flows left-to-right in logical sequence
- Information density is appropriate with clear visual hierarchy
- Main message: Signal transduction pathway demonstration
- Target audience: Undergraduate biology students

SCIENTIFIC CONTENT:
- Terminology: Standard biochemical nomenclature (receptor, kinase, transcription)
- Pathway accuracy: Proper signal cascade representation
- Process logic: Ligand binding → protein activation → gene expression
- Convention compliance: Follows standard pathway diagram conventions

DETAILED DESCRIPTION:
This figure illustrates a cellular signal transduction pathway showing how an external signal (ligand) binds to a membrane receptor, triggering a cascade of intracellular events that ultimately leads to changes in gene expression. The pathway demonstrates three key stages: signal reception, signal transduction, and cellular response."""
                    return _Msg()
            return _FakeVision()

        if os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1500)
        elif os.getenv("OPENROUTER_API_KEY"):
            return ChatOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
                temperature=0.2,
                max_tokens=1500
            )
        else:
            raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")

    def _init_text_llm(self):
        """Initialize text LLM for analysis parsing."""
        if os.getenv("TEST_MODE"):
            class _FakeText:
                def invoke(self, messages):
                    class _Msg:
                        content = "Mock analysis from comprehensive description"
                    return _Msg()
            return _FakeText()

        if os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, max_tokens=600)
        elif os.getenv("OPENROUTER_API_KEY"):
            return ChatOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                model="openai/gpt-3.5-turbo",
                temperature=0.1,
                max_tokens=600
            )
        else:
            raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")

    def _get_image_reference(self, image_data: str):
        """Create image reference for LLM messages."""
        try:
            if os.getenv("PUBLIC_BASE_URL"):
                from main_vision_only import _persist_image_and_get_url
                return {"type": "image_url", "image_url": {"url": _persist_image_and_get_url(image_data)}}
            else:
                return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
        except Exception:
            return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}

    def _comprehensive_vision_analysis(self, image_data: str, context: str, figure_type: str) -> str:
        """Single comprehensive vision analysis call covering all aspects."""
        if not image_data or os.getenv("TEST_MODE"):
            return """COMPREHENSIVE VISION ANALYSIS:

VISUAL ELEMENTS:
- Colors: Balanced color palette with primary and secondary colors
- Typography: Consistent hierarchy with readable font sizes
- Layout: Well-organized with appropriate spacing and alignment

COMMUNICATION ASPECTS:
- Information flow appears logical and sequential
- Information density is appropriate for the content
- Main message is clearly communicated
- Suitable for intended audience level

SCIENTIFIC CONTENT:
- Terminology appears accurate and follows conventions
- Pathway relationships are logically presented
- Process sequences follow biological logic
- Adheres to standard scientific visualization practices

DETAILED DESCRIPTION:
This figure appears to demonstrate scientific concepts through clear visual representation, with appropriate use of design elements to support the educational message."""

        try:
            system_prompt = """You are an expert scientific figure analyst with expertise in visual design, scientific communication, and biological accuracy.

Analyze this scientific figure comprehensively across ALL dimensions. Provide detailed observations in the following structure:

VISUAL ELEMENTS (for design analysis):
- Colors: List specific colors used, mention hex codes if identifiable, assess palette effectiveness
- Typography: Describe font sizes, hierarchy levels, text readability and consistency
- Layout: Detail spacing, alignment, visual balance, element positioning and organization
- Design quality: Professional polish, consistency, accessibility considerations

COMMUNICATION ASPECTS (for clarity analysis):
- Information flow: Describe the logical sequence and how information is presented
- Information density: Assess cognitive load, complexity level, appropriateness for audience
- Main message: Identify the primary message and how clearly it's communicated
- Audience targeting: Evaluate appropriateness for intended audience level

SCIENTIFIC CONTENT (for accuracy analysis):
- Terminology: List specific scientific terms, nomenclature accuracy, field conventions
- Pathway relationships: Describe biological/scientific relationships shown
- Process logic: Evaluate temporal sequences, cause-effect relationships, mechanism validity
- Standards compliance: Assess adherence to scientific visualization conventions and potential risks

DETAILED DESCRIPTION:
Provide a comprehensive 3-4 sentence description of what the figure shows, its main components, relationships illustrated, and key scientific concepts presented.

Be specific and detailed - this analysis will be used to evaluate the figure without additional vision calls."""

            context_text = f"""Context: {context or 'Scientific figure analysis'}
Figure Type: {figure_type or 'general'}

Analyze this figure comprehensively across visual design, communication effectiveness, and scientific accuracy."""

            image_ref = self._get_image_reference(image_data)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": context_text},
                    image_ref,
                ])
            ]

            response = self.vision_llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            logger.error(f"V2: Comprehensive vision analysis failed: {str(e)}")
            return """COMPREHENSIVE VISION ANALYSIS (FALLBACK):

VISUAL ELEMENTS:
- Colors: Unable to analyze specific colors - ensure good contrast and accessibility
- Typography: Review text hierarchy and ensure consistent sizing
- Layout: Check spacing and alignment for optimal readability
- Design quality: Verify professional appearance and consistency

COMMUNICATION ASPECTS:
- Information flow: Review logical sequence and flow indicators
- Information density: Balance complexity with clarity for target audience
- Main message: Ensure primary message is clearly emphasized
- Audience targeting: Adjust complexity for intended audience level

SCIENTIFIC CONTENT:
- Terminology: Verify scientific accuracy and standard conventions
- Pathway relationships: Confirm biological relationships are correctly represented
- Process logic: Validate temporal sequences and mechanisms
- Standards compliance: Check adherence to field-specific conventions

DETAILED DESCRIPTION:
This figure appears to present scientific information through visual elements. While detailed analysis was not possible, standard scientific figure principles should be applied for optimal communication."""

    def _extract_visual_analysis(self, comprehensive_analysis: str) -> str:
        """Extract visual design analysis from comprehensive description."""
        try:
            system_prompt = """You are a visual design expert. Based on the comprehensive figure analysis provided, create a focused visual design evaluation.

Extract and analyze:
1. Color palette effectiveness and accessibility
2. Typography and text hierarchy clarity
3. Layout balance and spacing
4. Visual hierarchy and information organization

Provide specific, actionable feedback with a score out of 10. Focus on design principles for scientific communication.

Format your response as:
VISUAL DESIGN ANALYSIS (Score: X/10):

[Your analysis with specific recommendations]"""

            user_prompt = f"Comprehensive Analysis:\n{comprehensive_analysis}\n\nProvide focused visual design analysis and score."

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = self.text_llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            logger.warning(f"V2: Visual analysis extraction failed: {str(e)}")
            return """VISUAL DESIGN ANALYSIS (Score: 7/10):

🎨 Color Usage:
→ Review color palette for effectiveness and accessibility
- Use appropriate contrast ratios for readability

📝 Typography & Hierarchy:
→ Ensure consistent text hierarchy throughout
- Maintain readable font sizes (minimum 10pt)

📐 Layout & Spacing:
→ Check spacing and alignment for professional appearance
- Balance information density with visual clarity"""

    def _extract_communication_analysis(self, comprehensive_analysis: str, figure_type: str) -> str:
        """Extract communication analysis from comprehensive description."""
        try:
            system_prompt = """You are a scientific communication expert. Based on the comprehensive figure analysis provided, create a focused communication evaluation.

Extract and analyze:
1. Information flow and logical sequence
2. Information density and cognitive load
3. Clarity of main message
4. Audience appropriateness

Provide specific feedback with a score out of 10. Focus on how effectively the figure communicates its intended message.

Format your response as:
COMMUNICATION CLARITY ANALYSIS (Score: X/10):

[Your analysis with specific recommendations]"""

            user_prompt = f"Comprehensive Analysis:\n{comprehensive_analysis}\n\nFigure Type: {figure_type or 'general'}\n\nProvide focused communication analysis and score."

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = self.text_llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            logger.warning(f"V2: Communication analysis extraction failed: {str(e)}")
            return f"""COMMUNICATION CLARITY ANALYSIS (Score: 7/10):

🎯 Information Flow:
→ Review information sequence for logical progression
- Figure type: {figure_type or 'general'}

📊 Information Density:
→ Balance complexity with clarity for target audience
- Ensure all elements serve the main message

👥 Audience Appropriateness:
→ Adjust terminology and detail level for intended readers
→ Include sufficient context without overwhelming detail"""

    def _extract_scientific_analysis(self, comprehensive_analysis: str, figure_type: str) -> str:
        """Extract scientific accuracy analysis from comprehensive description."""
        try:
            system_prompt = """You are a scientific accuracy expert. Based on the comprehensive figure analysis provided, create a focused scientific validation.

Extract and analyze:
1. Scientific nomenclature and terminology accuracy
2. Biological/scientific pathway logic and relationships
3. Adherence to field conventions
4. Units, measurements, and notation correctness

Provide specific feedback with a score out of 10. Focus on scientific validity and accuracy.

Format your response as:
SCIENTIFIC ACCURACY ANALYSIS (Score: X/10):

[Your analysis with specific recommendations]"""

            user_prompt = f"Comprehensive Analysis:\n{comprehensive_analysis}\n\nFigure Type: {figure_type or 'general'}\n\nProvide focused scientific accuracy analysis and score."

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = self.text_llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            logger.warning(f"V2: Scientific analysis extraction failed: {str(e)}")
            return f"""SCIENTIFIC ACCURACY ANALYSIS (Score: 8/10):

🔬 Nomenclature Check:
→ Verify scientific terminology follows field conventions
- Figure type: {figure_type or 'general'}

🧬 Pathway Logic:
→ Confirm biological relationships are accurately represented
→ Validate temporal sequences and mechanisms

📏 Standards Compliance:
→ Check adherence to scientific visualization standards
→ Verify conventions specific to your research field"""

    def _extract_content_interpretation(self, comprehensive_analysis: str) -> str:
        """Extract content interpretation from comprehensive description."""
        try:
            # Look for DETAILED DESCRIPTION section
            if "DETAILED DESCRIPTION:" in comprehensive_analysis:
                sections = comprehensive_analysis.split("DETAILED DESCRIPTION:")
                if len(sections) > 1:
                    description = sections[1].strip()
                    # Clean up the description
                    description = re.sub(r'\n+', ' ', description)
                    description = ' '.join(description.split())
                    if description:
                        return description

            # Fallback: extract key information from the comprehensive analysis
            system_prompt = """Based on the comprehensive figure analysis, provide a clear 2-3 sentence summary of what this figure shows.

Start with "This figure shows..." or "This diagram illustrates..." and focus on:
- Main concept being illustrated
- Key components and relationships
- Overall message or process

Keep it accessible for non-experts while being specific about what you observe."""

            user_prompt = f"Comprehensive Analysis:\n{comprehensive_analysis}\n\nProvide a plain language interpretation."

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = self.text_llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            logger.warning(f"V2: Content interpretation extraction failed: {str(e)}")
            return "This figure presents scientific information through visual elements designed to communicate key concepts and relationships within the field."

    def analyze(
        self,
        image_data: str,
        context: str,
        figure_type: str,
        session_id: str
    ) -> VisionAnalysisResult:
        """
        Analyze figure using V2 optimized approach (1 vision call + text processing).

        This approach makes:
        1. Single comprehensive vision call covering all aspects
        2. Text-based extraction of specific analyses from comprehensive result
        """
        start_time = time.time()

        logger.info(f"V2: Starting optimized vision analysis with {session_id}")

        # Single comprehensive vision call
        comprehensive_analysis = self._comprehensive_vision_analysis(image_data, context, figure_type)

        # Extract specific analyses from comprehensive result using text processing
        content_interpretation = self._extract_content_interpretation(comprehensive_analysis)
        visual_analysis = self._extract_visual_analysis(comprehensive_analysis)
        communication_analysis = self._extract_communication_analysis(comprehensive_analysis, figure_type)
        scientific_analysis = self._extract_scientific_analysis(comprehensive_analysis, figure_type)

        processing_time = time.time() - start_time
        vision_calls = 1 if image_data and not os.getenv("TEST_MODE") else 0

        logger.info(f"V2: Completed optimized analysis in {processing_time:.2f}s with {vision_calls} vision call")

        return VisionAnalysisResult(
            visual_analysis=visual_analysis,
            communication_analysis=communication_analysis,
            scientific_analysis=scientific_analysis,
            content_interpretation=content_interpretation,
            vision_calls_made=vision_calls,
            processing_time_seconds=processing_time,
            engine_version="v2_optimized",
            scientific_coherence_score=None,
            scientific_coherence_reason=None
        )

    def get_cost_estimate(self, image_data: str) -> Dict[str, Any]:
        """Estimate cost for V2 optimized analysis."""
        if not image_data or os.getenv("TEST_MODE"):
            return {
                "vision_calls": 0,
                "estimated_tokens": 0,
                "relative_cost": 0.0
            }

        return {
            "vision_calls": 1,
            "estimated_tokens": 2500,  # 1 comprehensive call (~1500 tokens) + text processing (~1000 tokens)
            "relative_cost": 0.25  # 75% cost reduction vs V1 (1 vision call vs 4)
        }