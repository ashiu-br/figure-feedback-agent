"""Vision engine abstraction for shadow mode testing."""

from typing import Protocol, Dict, Any, Optional
from typing_extensions import TypedDict


class VisionAnalysisResult(TypedDict):
    """Standardized result from vision analysis engines."""
    visual_analysis: str
    communication_analysis: str
    scientific_analysis: str
    content_interpretation: str
    vision_calls_made: int
    processing_time_seconds: float
    engine_version: str
    # Optional fields for enhanced engines
    scientific_coherence_score: Optional[int]
    scientific_coherence_reason: Optional[str]


class VisionEngine(Protocol):
    """Protocol for vision analysis engines."""

    version: str

    def analyze(
        self,
        image_data: str,
        context: str,
        figure_type: str,
        session_id: str
    ) -> VisionAnalysisResult:
        """
        Analyze figure using vision capabilities.

        Args:
            image_data: Base64 encoded image data
            context: User-provided context about the figure
            figure_type: Type of figure (pathway, workflow, etc.)
            session_id: Session identifier for caching

        Returns:
            VisionAnalysisResult with all analysis components
        """
        ...

    def get_cost_estimate(self, image_data: str) -> Dict[str, Any]:
        """
        Estimate cost for analyzing this image.

        Returns:
            Dict with cost information:
            - vision_calls: number of vision API calls
            - estimated_tokens: estimated token usage
            - relative_cost: relative cost compared to baseline
        """
        ...