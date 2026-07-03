import uuid
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from services.retrieval.planner.strategy_registry import StrategyRegistry
from services.retrieval.planner.statistics import StatisticsCatalog
from services.retrieval.planner.cost.cost_estimator import CostEstimator
from services.retrieval.planner.execution_plan import ExecutionPlan
from services.retrieval.planner.selectivity_estimator import MetadataSelectivityEstimator
from services.retrieval.planner.feedback import PlannerFeedbackLoop

logger = logging.getLogger(__name__)

class QueryPlanner:
    """
    Core Cost-Based Query Optimizer.
    Evaluates statistics, estimates resource costs, chooses the optimal strategy,
    and returns a query plan.
    """
    def __init__(
        self,
        registry: StrategyRegistry,
        statistics_catalog: StatisticsCatalog,
        cost_estimator: CostEstimator,
        selectivity_estimator: Optional[MetadataSelectivityEstimator] = None,
        feedback_loop: Optional[PlannerFeedbackLoop] = None
    ):
        self.registry = registry
        self.statistics_catalog = statistics_catalog
        self.cost_estimator = cost_estimator
        self.selectivity_estimator = selectivity_estimator
        self.feedback_loop = feedback_loop or PlannerFeedbackLoop()

    def plan(
        self,
        collection_id: uuid.UUID,
        k: int,
        filters: Optional[dict] = None,
        mode: str = "BALANCED"
    ) -> ExecutionPlan:
        """
        Evaluate and compile the lowest-cost execution plan for the search.
        """
        trace: List[str] = []
        candidate_plans: Dict[str, Any] = {}

        # 0. Estimate filter selectivity (Feature 7)
        selectivity = 1.0
        if self.selectivity_estimator and filters:
            trace.append(f"Resolving metadata selectivity for filters: {filters}")
            try:
                selectivity = self.selectivity_estimator.estimate_selectivity(collection_id, filters)
                trace.append(f"Selectivity estimate: {selectivity:.2%}")
            except Exception as e:
                trace.append(f"Selectivity estimation failed: {e}. Defaulting to 1.0.")
                logger.error(f"Selectivity estimator failure: {e}")

        # 1. Fetch catalog statistics (Feature 3)
        trace.append("Retrieving statistics from StatisticsCatalog.")
        try:
            stats = self.statistics_catalog.get_statistics(collection_id)
            trace.append(
                f"Collection statistics parsed. Size: {stats.collection_size}, "
                f"Dimension: {stats.dimension}, Sealed: {stats.sealed_segments}, "
                f"Growing: {stats.growing_segments}, Graph nodes: {stats.graph_nodes}."
            )
        except Exception as e:
            trace.append(f"Statistics retrieval failed: {e}. Defaulting to backup metrics.")
            logger.error(f"Optimizer statistics catalog failure: {e}")
            from services.retrieval.planner.statistics import CollectionStatistics
            stats = CollectionStatistics()
            stats.collection_id = collection_id
            stats.collection_size = 1000  # Conservative fallback assumptions

        # 2. Evaluate Exact Retrieval Strategy Cost (Feature 4)
        trace.append("Evaluating EXACT strategy plan cost.")
        exact_est = self.cost_estimator.estimate_cost(stats, k, selectivity, mode, "EXACT")
        candidate_plans["EXACT"] = exact_est.to_dict()
        trace.append(f"EXACT cost estimate: {exact_est.total_cost:.2f} relative units.")

        # 3. Evaluate HNSW Retrieval Strategy Cost
        trace.append("Evaluating HNSW strategy plan cost.")
        is_hnsw_viable = stats.graph_nodes > 0
        
        if is_hnsw_viable:
            hnsw_est = self.cost_estimator.estimate_cost(stats, k, selectivity, mode, "HNSW")
            candidate_plans["HNSW"] = hnsw_est.to_dict()
            trace.append(f"HNSW cost estimate: {hnsw_est.total_cost:.2f} relative units.")
        else:
            trace.append("HNSW strategy is NOT viable: no sealed graph index nodes found.")
            # Set infinite cost to prevent selection
            from services.retrieval.planner.cost.cost_estimate import CostEstimate
            hnsw_est = CostEstimate(
                cpu_cost=float('inf'), graph_cost=float('inf'), io_cost=float('inf'),
                memory_cost=float('inf'), cache_cost=float('inf'), metadata_cost=float('inf'),
                ranking_cost=float('inf'), serialization_cost=float('inf'), total_cost=float('inf'),
                confidence_score=0.0, assumptions={"reason": "No graph index built."}
            )
            candidate_plans["HNSW"] = hnsw_est.to_dict()

        # 4. Selection Optimization (Feature 6 Decision Trace)
        chosen_strategy_name = "EXACT"
        chosen_cost = exact_est

        if is_hnsw_viable:
            if hnsw_est.total_cost < exact_est.total_cost:
                chosen_strategy_name = "HNSW"
                chosen_cost = hnsw_est
                trace.append(
                    f"Selected HNSW strategy. Relative cost ({hnsw_est.total_cost:.2f}) "
                    f"is lower than EXACT cost ({exact_est.total_cost:.2f})."
                )
            else:
                trace.append(
                    f"Selected EXACT strategy. Relative cost ({exact_est.total_cost:.2f}) "
                    f"is lower than HNSW cost ({hnsw_est.total_cost:.2f})."
                )
        else:
            trace.append("Defaulted to EXACT strategy due to HNSW index absence.")

        # Retrieve Strategy implementation from registry (Feature 2)
        try:
            chosen_strategy = self.registry.get_strategy(chosen_strategy_name)
        except Exception as e:
            trace.append(f"Strategy registry error for {chosen_strategy_name}: {e}. Fallback to EXACT.")
            chosen_strategy_name = "EXACT"
            chosen_strategy = self.registry.get_strategy("EXACT")
            chosen_cost = exact_est

        # Calibrate latency prediction (Feature 8)
        if self.feedback_loop:
            predicted_latency = self.feedback_loop.calibrate_latency(chosen_cost.total_cost)
            chosen_cost.assumptions["predicted_latency_ms"] = predicted_latency
            trace.append(f"Calibrated latency prediction: {predicted_latency:.3f} ms.")

        # Build reasoning explanation (Feature 5 / 6)
        reasoning = (
            f"Chosen strategy is {chosen_strategy_name} based on lower relative cost weight "
            f"({chosen_cost.total_cost:.2f} total cost units)."
        )

        return ExecutionPlan(
            strategy_name=chosen_strategy_name,
            chosen_strategy=chosen_strategy,
            cost_estimate=chosen_cost,
            statistics_used=stats.to_dict(),
            decision_trace=trace,
            candidate_plans=candidate_plans,
            reasoning=reasoning
        )
