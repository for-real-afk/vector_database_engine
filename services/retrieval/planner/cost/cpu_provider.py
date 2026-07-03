import logging
from services.retrieval.planner.cost.base_provider import CostProvider
from services.retrieval.planner.statistics import CollectionStatistics

logger = logging.getLogger(__name__)

class CPUCostProvider(CostProvider):
    """
    Estimates relative CPU cost units for vector comparison operations,
    traversal checks, and search algorithms.
    """
    def __init__(self, cost_per_distance_comp: float = 1.0, traversal_step_cost: float = 0.5):
        self.cost_per_distance_comp = cost_per_distance_comp
        self.traversal_step_cost = traversal_step_cost

    def calculate_cost(
        self, 
        stats: CollectionStatistics, 
        k: int, 
        selectivity: float = 1.0, 
        mode: str = "BALANCED",
        strategy: str = "EXACT"
    ) -> float:
        """
        Estimate relative CPU cost weights based on strategy type.
        """
        mode_upper = mode.upper()
        # Mode weights multiplier (Feature 11)
        mode_multiplier = 1.0
        if mode_upper == "LOWEST_LATENCY":
            mode_multiplier = 1.5
        elif mode_upper == "LOWEST_MEMORY":
            mode_multiplier = 0.8

        strategy_upper = strategy.upper()

        if strategy_upper == "EXACT":
            # Exact scan is O(N) comparisons
            active_size = max(1, int(stats.collection_size * selectivity))
            base_cpu = active_size * self.cost_per_distance_comp
            return base_cpu * mode_multiplier

        elif strategy_upper == "HNSW":
            if stats.graph_nodes == 0:
                return float('inf')

            # HNSW estimate (layers * M + ef_search)
            ef_search = 32.0 # default search depth
            expected_visited = (stats.graph_layers * stats.average_degree) + ef_search
            dist_computations = expected_visited * stats.average_degree
            
            base_cpu = dist_computations * self.cost_per_distance_comp
            
            # Selectivity penalty: if metadata filter matches are sparse, HNSW must traverse
            # more nodes to collect the K matching candidates.
            if selectivity < 1.0:
                penalty = max(1.0, 1.0 / max(0.01, selectivity))
                base_cpu *= penalty
                
            return base_cpu * mode_multiplier

        else:
            logger.warning(f"Unknown strategy code: {strategy}. Defaulting CPU cost to 0.0.")
            return 0.0
