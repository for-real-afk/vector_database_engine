import logging
from typing import Dict, Optional
from services.retrieval.planner.cost.cost_estimate import CostEstimate
from services.retrieval.planner.cost.cpu_provider import CPUCostProvider
from services.retrieval.planner.statistics import CollectionStatistics

logger = logging.getLogger(__name__)

class CostEstimator:
    """
    Coordinates modular Cost Providers to generate a strongly typed, 
    immutable CostEstimate.
    """
    def __init__(self, cpu_provider: CPUCostProvider = None):
        self.cpu_provider = cpu_provider or CPUCostProvider()

    def estimate_cost(
        self,
        stats: CollectionStatistics,
        k: int,
        selectivity: float = 1.0,
        mode: str = "BALANCED",
        strategy: str = "EXACT"
    ) -> CostEstimate:
        """
        Calculate and combine cost scores across all resource dimensions 
        to build a CostEstimate.
        """
        # Calculate CPU cost
        cpu_cost = self.cpu_provider.calculate_cost(stats, k, selectivity, mode, strategy)
        
        # Setup defaults for other resource dimensions (to be populated in subsequent steps)
        graph_cost = 0.0
        io_cost = 0.0
        memory_cost = 0.0
        cache_cost = 0.0
        metadata_cost = 0.0
        ranking_cost = 0.0
        serialization_cost = 0.0
        
        # Calculate total cost
        total_cost = cpu_cost
        
        # Confidence score estimation based on statistics completeness (Feature 5)
        confidence = 1.0
        assumptions = {
            "strategy": strategy,
            "mode": mode,
            "selectivity": selectivity,
            "k": k,
            "collection_size": stats.collection_size
        }
        
        if stats.collection_size == 0:
            confidence = 0.3
            assumptions["warning"] = "Missing stats: collection size is zero."
        elif stats.sealed_segments > 0 and stats.graph_nodes == 0 and strategy.upper() == "HNSW":
            confidence = 0.5
            assumptions["warning"] = "Unbuilt HNSW graph on sealed segments."
            
        return CostEstimate(
            cpu_cost=cpu_cost,
            graph_cost=graph_cost,
            io_cost=io_cost,
            memory_cost=memory_cost,
            cache_cost=cache_cost,
            metadata_cost=metadata_cost,
            ranking_cost=ranking_cost,
            serialization_cost=serialization_cost,
            total_cost=total_cost,
            confidence_score=confidence,
            assumptions=assumptions
        )
