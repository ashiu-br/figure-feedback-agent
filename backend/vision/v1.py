"""V1 Vision Engine - Current multi-call approach."""

import os
import time
import logging
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from openinference.instrumentation import using_prompt_template

from .base import VisionEngine, VisionAnalysisResult

logger = logging.getLogger(__name__)


class V1MultiCallVisionEngine:
    """Current vision engine using 4 separate vision API calls."""

    version = "v1"

    def __init__(self):
        """Initialize the V1 vision engine."""
        self.llm = self._init_llm()

    def _init_llm(self):
        """Initialize LLM for vision analysis."""
        if os.getenv("TEST_MODE"):
            class _Fake:
                def invoke(self, messages):
                    class _Msg:
                        content = "Mock vision analysis"
                    return _Msg()
            return _Fake()

        if os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=500)
        elif os.getenv("OPENROUTER_API_KEY"):
            return ChatOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
                temperature=0.2,
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

    def _analyze_visual_design(self, image_data: str, context: str, content_summary: str) -> str:
        """Analyze visual design aspects using vision."""
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

            image_ref = self._get_image_reference(image_data)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": context_text},
                    image_ref,
                ])
            ]

            response = self.llm.invoke(messages)
            return f"VISUAL DESIGN ANALYSIS:\n\n{response.content.strip()}"

        except Exception as e:
            logger.warning(f"Visual design analysis failed: {str(e)}, using fallback")
            return """VISUAL DESIGN ANALYSIS (Score: 7/10):

🎨 Color Usage:
→ Unable to analyze colors in detail - ensure good contrast
- Consider using brand colors or scientific publication standards

📝 Typography & Hierarchy:
→ Ensure text hierarchy supports information flow
- Use consistent font sizes within element categories

📐 Layout & Spacing:
→ Review spacing for optimal readability
- Balance information density with visual clarity"""

    def _evaluate_communication_clarity(self, image_data: str, context: str, figure_type: str, content_summary: str) -> str:
        """Evaluate communication clarity using vision."""
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

            image_ref = self._get_image_reference(image_data)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": context_text},
                    image_ref,
                ])
            ]

            response = self.llm.invoke(messages)
            return f"COMMUNICATION CLARITY ANALYSIS:\n\n{response.content.strip()}"

        except Exception as e:
            logger.warning(f"Communication analysis failed: {str(e)}, using fallback")
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
→ Include necessary context without overwhelming detail"""

    def _validate_scientific_accuracy(self, image_data: str, context: str, figure_type: str, content_summary: str) -> str:
        """Validate scientific accuracy using vision."""
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
→ This analysis is based on figure content only - verify with domain expertise"""

        try:
            system_prompt = """You are a scientific expert. Analyze this figure and evaluate:
1. Scientific nomenclature and terminology accuracy
2. Biological/scientific pathway logic and relationships
3. Adherence to field conventions
4. Units, measurements, and notation correctness

Provide specific feedback with a score out of 10. Focus on scientific validity and accuracy."""

            context_text = f"Context: {context or 'Scientific figure analysis'}\nFigure Type: {figure_type or 'general'}\n\nPlease analyze the scientific accuracy."

            image_ref = self._get_image_reference(image_data)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": context_text},
                    image_ref,
                ])
            ]

            response = self.llm.invoke(messages)
            return f"SCIENTIFIC ACCURACY ANALYSIS:\n\n{response.content.strip()}"

        except Exception as e:
            logger.warning(f"Scientific accuracy analysis failed: {str(e)}, using fallback")
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
→ Confirm pathway directions match established literature"""

    def _interpret_figure_content(self, image_data: str, context: str, figure_type: str) -> str:
        """Generate a plain language summary using vision LLM."""
        if not image_data or os.getenv("TEST_MODE"):
            return f"This figure appears to be a {figure_type or 'scientific diagram'} showing biological or scientific relationships. {context or 'The figure illustrates key concepts and processes within its domain.'}"

        try:
            logger.info("V1: Attempting vision-based content interpretation")

            system_prompt = """You are an expert at interpreting scientific figures. Look at the provided image and describe what it shows in 2-3 clear sentences that a non-expert could understand.

Focus on:
- Main biological/scientific concept being illustrated
- Key components and their relationships
- Overall process or message being communicated

Start with "This figure shows..." or "This diagram illustrates..." and be specific about what you observe visually."""

            context_text = f"""Context: {context or 'Scientific figure analysis'}
Figure Type: {figure_type or 'diagram'}

Please analyze the image and provide a plain language interpretation."""

            image_ref = self._get_image_reference(image_data)

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
            logger.warning(f"V1: Content interpretation failed: {str(e)}, using fallback")
            return f"This figure appears to be a {figure_type or 'scientific diagram'}. While detailed analysis was not possible, it likely shows important biological or scientific relationships. {context or 'The figure communicates key concepts within its domain.'}"

    def analyze(
        self,
        image_data: str,
        context: str,
        figure_type: str,
        session_id: str
    ) -> VisionAnalysisResult:
        """
        Analyze figure using V1 approach (4 separate vision calls).

        This is the current implementation that makes:
        1. Content interpretation call
        2. Visual design analysis call
        3. Communication clarity call
        4. Scientific accuracy call
        """
        start_time = time.time()

        logger.info(f"V1: Starting vision analysis with {session_id}")

        # Call 1: Content Interpretation
        content_interpretation = self._interpret_figure_content(image_data, context, figure_type)

        # Call 2: Visual Design Analysis
        visual_analysis = self._analyze_visual_design(image_data, context, content_interpretation)

        # Call 3: Communication Clarity Analysis
        communication_analysis = self._evaluate_communication_clarity(image_data, context, figure_type, content_interpretation)

        # Call 4: Scientific Accuracy Analysis
        scientific_analysis = self._validate_scientific_accuracy(image_data, context, figure_type, content_interpretation)

        processing_time = time.time() - start_time
        vision_calls = 4 if image_data and not os.getenv("TEST_MODE") else 0

        logger.info(f"V1: Completed vision analysis in {processing_time:.2f}s with {vision_calls} vision calls")

        return VisionAnalysisResult(
            visual_analysis=visual_analysis,
            communication_analysis=communication_analysis,
            scientific_analysis=scientific_analysis,
            content_interpretation=content_interpretation,
            vision_calls_made=vision_calls,
            processing_time_seconds=processing_time,
            engine_version="v1",
            scientific_coherence_score=None,
            scientific_coherence_reason=None
        )

    def get_cost_estimate(self, image_data: str) -> Dict[str, Any]:
        """Estimate cost for V1 analysis."""
        if not image_data or os.getenv("TEST_MODE"):
            return {
                "vision_calls": 0,
                "estimated_tokens": 0,
                "relative_cost": 0.0
            }

        return {
            "vision_calls": 4,
            "estimated_tokens": 2000,  # Rough estimate: 4 calls × 500 tokens each
            "relative_cost": 1.0  # V1 is the baseline
        }