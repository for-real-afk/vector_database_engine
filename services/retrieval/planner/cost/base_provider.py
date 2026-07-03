from abc import ABC, abstractmethod
from services.retrieval.planner.statistics import CollectionStatistics

class CostProvider(ABC):
    """
    Abstract Interface for modular cost estimation providers.
    Allows estimating specific resource footprint metrics independently.
    """
    @abstractmethod
    def calculate_cost(
        self, 
        stats: CollectionStatistics, 
        k: int, 
        selectivity: float = 1.0, 
        mode: str = "BALANCED"
    ) -> float:
        """
        Calculate relative cost weight value for this provider.
        """
        pass
