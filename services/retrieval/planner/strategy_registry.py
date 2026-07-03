from abc import ABC, abstractmethod
import uuid
import logging
from typing import Dict, List, Optional
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class RetrievalStrategy(ABC):
    """
    Abstract Base Class for retrieval strategies.
    Decouples specific search implementations from the query planning core.
    """
    @abstractmethod
    def execute(
        self, 
        db: Session, 
        collection_id: uuid.UUID, 
        query_vector: list[float], 
        k: int, 
        allowed_ids: Optional[set[uuid.UUID]] = None
    ) -> list[dict]:
        """Execute the search query."""
        pass

    @abstractmethod
    def estimate_cost(
        self, 
        stats, 
        k: int, 
        filters: Optional[dict] = None
    ) -> dict:
        """
        Calculate cost projections for CPU, Memory, and Disk I/O.
        Returns a dict: {'cpu': float, 'io': float, 'memory': float, 'total_cost': float}
        """
        pass


class StrategyRegistry:
    """
    Dynamic registry to inject and manage query strategies.
    """
    def __init__(self):
        self._strategies: Dict[str, RetrievalStrategy] = {}

    def register(self, name: str, strategy: RetrievalStrategy):
        """Register a new strategy."""
        norm_name = name.upper()
        if norm_name in self._strategies:
            logger.warning(f"Overwriting already registered strategy: {name}")
        self._strategies[norm_name] = strategy
        logger.info(f"Registered strategy: {norm_name}")

    def get_strategy(self, name: str) -> RetrievalStrategy:
        """Retrieve target strategy by name."""
        norm_name = name.upper()
        if norm_name not in self._strategies:
            raise KeyError(f"Retrieval strategy {name} is not registered.")
        return self._strategies[norm_name]

    def list_strategies(self) -> List[str]:
        """List all currently registered strategy keys."""
        return list(self._strategies.keys())
