import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PlannerFeedbackLoop:
    """
    Stateful adaptive calibration feedback loop (Feature 8).
    Maintains historic metrics, calculates query cost-to-latency ratios,
    and applies exponential smoothing adjustments to future predictions.
    """
    def __init__(self, learning_rate: float = 0.2, initial_ratio: float = 0.0001):
        self.learning_rate = learning_rate
        # cost-to-latency calibration multiplier
        self.calibration_ratio = initial_ratio
        self.query_count = 0
        self.history: list[Dict[str, Any]] = []

    def record_execution(self, estimated_cost: float, actual_latency_ms: float, strategy_name: str):
        """
        Record observed query performance to update future predictions.
        """
        if estimated_cost <= 0 or estimated_cost == float('inf'):
            return

        self.query_count += 1
        observed_ratio = actual_latency_ms / estimated_cost
        
        # Apply exponential moving average: new = alpha * observed + (1 - alpha) * old
        old_ratio = self.calibration_ratio
        self.calibration_ratio = (self.learning_rate * observed_ratio) + ((1.0 - self.learning_rate) * old_ratio)
        
        log_entry = {
            "query_index": self.query_count,
            "strategy": strategy_name,
            "estimated_cost": estimated_cost,
            "actual_latency_ms": actual_latency_ms,
            "observed_ratio": observed_ratio,
            "calibrated_ratio": self.calibration_ratio
        }
        self.history.append(log_entry)
        
        logger.debug(
            f"Planner Calibration Loop: Recorded query {self.query_count} ({strategy_name}). "
            f"Observed ratio: {observed_ratio:.6f}, New calibration ratio: {self.calibration_ratio:.6f}"
        )

    def calibrate_latency(self, cost: float) -> float:
        """
        Predict latency in milliseconds using the calibrated cost multiplier.
        """
        if cost == float('inf'):
            return float('inf')
        return cost * self.calibration_ratio

    def reset(self):
        """Reset calibration loop metrics."""
        self.calibration_ratio = 0.0001
        self.query_count = 0
        self.history.clear()
