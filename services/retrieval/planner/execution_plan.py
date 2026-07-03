import uuid
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from services.retrieval.planner.strategy_registry import RetrievalStrategy
from services.retrieval.planner.cost.cost_estimate import CostEstimate

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ExecutionPlan:
    """
    Immutable execution plan compiled by the Query Optimizer.
    Contains decision trace, statistics consulted, and cost models.
    """
    strategy_name: str
    chosen_strategy: RetrievalStrategy
    cost_estimate: CostEstimate
    statistics_used: Dict[str, Any]
    decision_trace: List[str]
    candidate_plans: Dict[str, Any]
    reasoning: str
    planner_version: str = "1.0.0"

    def execute(
        self, 
        db: Session, 
        collection_id: uuid.UUID, 
        query_vector: list[float], 
        k: int, 
        allowed_ids: Optional[set[uuid.UUID]] = None
    ) -> list[dict]:
        """
        Execute the query plan under fault-tolerant wrappers (Feature 12).
        If HNSW strategy fails, falls back automatically to Exact search.
        """
        try:
            logger.info(f"Executing Query Optimizer plan: {self.strategy_name}")
            return self.chosen_strategy.execute(db, collection_id, query_vector, k, allowed_ids)
        except Exception as e:
            logger.error(
                f"Query planning execution failed for strategy {self.strategy_name}: {e}. "
                f"Executing fault-tolerant fallbacks to EXACT scan."
            )
            
            # Fallback (Feature 12)
            from services.retrieval.planner.strategies.exact_strategy import ExactRetrievalStrategy
            fallback_strategy = ExactRetrievalStrategy()
            return fallback_strategy.execute(db, collection_id, query_vector, k, allowed_ids)

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution plan properties to a serializable dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "cost_estimate": self.cost_estimate.to_dict(),
            "statistics_used": self.statistics_used,
            "decision_trace": self.decision_trace,
            "candidate_plans": self.candidate_plans,
            "reasoning": self.reasoning,
            "planner_version": self.planner_version
        }

    def to_json(self) -> str:
        """Convert execution plan properties to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        """Render a Markdown summary statistics report for explain operations."""
        trace_str = "\n".join([f"{idx+1}. {step}" for idx, step in enumerate(self.decision_trace)])
        
        lines = [
            "# 📋 Query Execution Plan Details",
            f"* **Chosen Strategy**: `{self.strategy_name}`",
            f"* **Reasoning**: *{self.reasoning}*",
            "",
            "## ⏱️ Decision Trace logs",
            trace_str,
            "",
            "## 📊 Statistics Consulted",
            f"- **Collection Size**: `{self.statistics_used.get('collection_size')}`",
            f"- **Sealed Segments**: `{self.statistics_used.get('sealed_segments')}`",
            f"- **Growing Segments**: `{self.statistics_used.get('growing_segments')}`",
            f"- **Graph nodes**: `{self.statistics_used.get('graph_nodes')}`",
            "",
            self.cost_estimate.to_markdown()
        ]
        return "\n".join(lines)
