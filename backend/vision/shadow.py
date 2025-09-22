"""Shadow Mode Vision Engine - Runs both V1 and V2 for comparison."""

import time
import logging
from typing import Dict, Any

from .base import VisionEngine, VisionAnalysisResult
from .v1 import V1MultiCallVisionEngine
from .v2_optimized import V2OptimizedVisionEngine

logger = logging.getLogger(__name__)


class ShadowVisionEngine:
    """Shadow mode engine that runs both V1 and V2, returns V1 results, logs V2 for comparison."""

    version = "shadow"

    def __init__(self):
        """Initialize both V1 and V2 engines."""
        self.v1_engine = V1MultiCallVisionEngine()
        self.v2_engine = V2OptimizedVisionEngine()

    def analyze(
        self,
        image_data: str,
        context: str,
        figure_type: str,
        session_id: str
    ) -> VisionAnalysisResult:
        """
        Run both V1 and V2 engines in shadow mode.

        Returns V1 results to maintain compatibility.
        Logs V2 results and comparison metrics for analysis.
        """
        shadow_start_time = time.time()

        logger.info(f"Shadow: Running parallel V1 and V2 analysis for {session_id}")

        # Run both engines
        try:
            # Run V1 (current production approach)
            v1_start = time.time()
            v1_result = self.v1_engine.analyze(image_data, context, figure_type, session_id)
            v1_duration = time.time() - v1_start

            # Run V2 (optimized approach)
            v2_start = time.time()
            v2_result = self.v2_engine.analyze(image_data, context, figure_type, session_id)
            v2_duration = time.time() - v2_start

            shadow_duration = time.time() - shadow_start_time

            # Calculate comparison metrics
            cost_savings = ((v1_result["vision_calls_made"] - v2_result["vision_calls_made"]) /
                           max(v1_result["vision_calls_made"], 1)) * 100

            speed_improvement = ((v1_duration - v2_duration) / max(v1_duration, 0.001)) * 100

            # Log shadow mode comparison (this will go to Arize traces)
            logger.info(f"Shadow Mode Comparison for {session_id}:")
            logger.info(f"  V1: {v1_result['vision_calls_made']} calls, {v1_duration:.2f}s")
            logger.info(f"  V2: {v2_result['vision_calls_made']} calls, {v2_duration:.2f}s")
            logger.info(f"  Cost Savings: {cost_savings:.1f}%")
            logger.info(f"  Speed Impact: {speed_improvement:+.1f}%")
            logger.info(f"  Shadow Overhead: {shadow_duration:.2f}s total")

            # Store comparison data for potential Arize instrumentation
            self._comparison_metrics = {
                "v1_calls": v1_result["vision_calls_made"],
                "v2_calls": v2_result["vision_calls_made"],
                "v1_duration": v1_duration,
                "v2_duration": v2_duration,
                "cost_savings_percent": cost_savings,
                "speed_improvement_percent": speed_improvement,
                "shadow_overhead": shadow_duration,
                "v1_engine_version": v1_result["engine_version"],
                "v2_engine_version": v2_result["engine_version"]
            }

            # Log V2 results for comparison (truncated for readability)
            logger.info(f"V2 Results Summary (for comparison):")
            logger.info(f"  Content: {v2_result['content_interpretation'][:100]}...")
            logger.info(f"  Visual Score: {self._extract_score(v2_result['visual_analysis'])}")
            logger.info(f"  Comm Score: {self._extract_score(v2_result['communication_analysis'])}")
            logger.info(f"  Sci Score: {self._extract_score(v2_result['scientific_analysis'])}")

            # Return V1 result to maintain production compatibility
            # But add shadow mode metadata
            v1_result_with_shadow = v1_result.copy()
            v1_result_with_shadow["engine_version"] = "shadow(returning_v1)"

            return v1_result_with_shadow

        except Exception as e:
            logger.error(f"Shadow mode failed: {str(e)}")
            logger.info("Falling back to V1 only")

            # Fallback to V1 only if shadow mode fails
            result = self.v1_engine.analyze(image_data, context, figure_type, session_id)
            result["engine_version"] = "shadow(fallback_v1)"
            return result

    def _extract_score(self, analysis_text: str) -> str:
        """Extract score from analysis text for comparison logging."""
        import re
        score_match = re.search(r'Score:\s*(\d+)', analysis_text)
        return score_match.group(1) + "/10" if score_match else "N/A"

    def get_comparison_metrics(self) -> Dict[str, Any]:
        """Get the latest comparison metrics from shadow mode analysis."""
        return getattr(self, '_comparison_metrics', {})

    def get_cost_estimate(self, image_data: str) -> Dict[str, Any]:
        """Return V1 cost estimate since that's what we're returning to users."""
        return self.v1_engine.get_cost_estimate(image_data)