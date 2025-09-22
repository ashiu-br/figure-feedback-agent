"""Vision engine factory and version selection."""

import os
import logging
from typing import Optional
from .base import VisionEngine, VisionAnalysisResult

logger = logging.getLogger(__name__)


def get_vision_engine(request_override: Optional[str] = None) -> VisionEngine:
    """
    Get vision engine based on version configuration.

    Args:
        request_override: Optional version override from request

    Returns:
        VisionEngine instance

    Supported versions:
        - v1: Current approach (4 separate vision calls)
        - v2: Optimized approach (1 comprehensive vision call)
        - shadow: Run both v1 and v2, return v1, log v2 for comparison
    """
    version = request_override or os.getenv("VISION_VERSION", "v1")

    logger.info(f"Loading vision engine version: {version}")

    try:
        if version == "v2":
            from .v2_optimized import V2OptimizedVisionEngine
            return V2OptimizedVisionEngine()
        elif version == "shadow":
            from .shadow import ShadowVisionEngine
            return ShadowVisionEngine()
        else:  # Default to v1
            from .v1 import V1MultiCallVisionEngine
            return V1MultiCallVisionEngine()

    except ImportError as e:
        logger.error(f"Failed to load vision engine {version}: {e}")
        logger.info("Falling back to v1 engine")
        from .v1 import V1MultiCallVisionEngine
        return V1MultiCallVisionEngine()


__all__ = ["VisionEngine", "VisionAnalysisResult", "get_vision_engine"]