import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass(frozen=True)
class CostEstimate:
    """
    Typed, immutable database execution cost estimation snapshot.
    Uses relative hardware-independent cost units rather than millisecond predictions.
    """
    cpu_cost: float
    graph_cost: float
    io_cost: float
    memory_cost: float
    cache_cost: float
    metadata_cost: float
    ranking_cost: float
    serialization_cost: float
    total_cost: float
    confidence_score: float
    assumptions: Dict[str, Any] = field(default_factory=dict)
    planner_version: str = "1.0.0"
    statistics_version: str = "1.0.0"

    def __lt__(self, other: "CostEstimate") -> bool:
        if not isinstance(other, CostEstimate):
            return NotImplemented
        return self.total_cost < other.total_cost

    def __le__(self, other: "CostEstimate") -> bool:
        if not isinstance(other, CostEstimate):
            return NotImplemented
        return self.total_cost <= other.total_cost

    def __gt__(self, other: "CostEstimate") -> bool:
        if not isinstance(other, CostEstimate):
            return NotImplemented
        return self.total_cost > other.total_cost

    def __ge__(self, other: "CostEstimate") -> bool:
        if not isinstance(other, CostEstimate):
            return NotImplemented
        return self.total_cost >= other.total_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert cost estimate properties to a serializable dictionary."""
        return {
            "cpu_cost": self.cpu_cost,
            "graph_cost": self.graph_cost,
            "io_cost": self.io_cost,
            "memory_cost": self.memory_cost,
            "cache_cost": self.cache_cost,
            "metadata_cost": self.metadata_cost,
            "ranking_cost": self.ranking_cost,
            "serialization_cost": self.serialization_cost,
            "total_cost": self.total_cost,
            "confidence_score": self.confidence_score,
            "assumptions": self.assumptions,
            "planner_version": self.planner_version,
            "statistics_version": self.statistics_version
        }

    def to_json(self) -> str:
        """Convert cost estimate properties to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        """Render a Markdown summary statistics report for explain operations."""
        breakdown = self.get_breakdown()
        lines = [
            "### 📊 Cost Estimate Report",
            f"* **Total Cost units**: `{self.total_cost:,.2f}`",
            f"* **Confidence Score**: `{self.confidence_score:.2%}`",
            "",
            "| Metric | Relative Cost | Percentage |",
            "| :--- | :---: | :---: |",
            f"| CPU | {self.cpu_cost:,.2f} | {breakdown['cpu']:.1%} |",
            f"| Disk I/O | {self.io_cost:,.2f} | {breakdown['io']:.1%} |",
            f"| HNSW Graph | {self.graph_cost:,.2f} | {breakdown['graph']:.1%} |",
            f"| RAM Memory | {self.memory_cost:,.2f} | {breakdown['memory']:.1%} |",
            f"| Cache | {self.cache_cost:,.2f} | {breakdown['cache']:.1%} |",
            f"| Metadata | {self.metadata_cost:,.2f} | {breakdown['metadata']:.1%} |",
            f"| Ranking | {self.ranking_cost:,.2f} | {breakdown['ranking']:.1%} |",
            f"| Serialization | {self.serialization_cost:,.2f} | {breakdown['serialization']:.1%} |",
            "",
            "#### Planner Assumptions",
        ]
        for k, v in self.assumptions.items():
            lines.append(f"- **{k}**: `{v}`")
        return "\n".join(lines)

    def get_breakdown(self) -> Dict[str, float]:
        """Calculate the percentage share of each cost component."""
        total = self.total_cost if self.total_cost > 0 else 1.0
        return {
            "cpu": self.cpu_cost / total,
            "io": self.io_cost / total,
            "graph": self.graph_cost / total,
            "memory": self.memory_cost / total,
            "cache": self.cache_cost / total,
            "metadata": self.metadata_cost / total,
            "ranking": self.ranking_cost / total,
            "serialization": self.serialization_cost / total
        }
