import numpy as np

class DistanceMetric:
    """
    Base class / utility wrapper for vector distance computations.
    """
    @staticmethod
    def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute Cosine Similarity between two vectors: (u . v) / (||u|| ||v||)
        """
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            return 0.0
        return float(np.dot(u, v) / (norm_u * norm_v))

    @staticmethod
    def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute Cosine Distance: 1.0 - Cosine Similarity
        """
        return 1.0 - DistanceMetric.cosine_similarity(u, v)

    @staticmethod
    def l2_distance(u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute L2 (Euclidean) Distance: sqrt(sum((u_i - v_i)^2))
        """
        return float(np.linalg.norm(u - v))

    @staticmethod
    def dot_product(u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute Dot Product: sum(u_i * v_i)
        """
        return float(np.dot(u, v))

    @staticmethod
    def manhattan_distance(u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute Manhattan (L1) Distance: sum(|u_i - v_i|)
        """
        return float(np.sum(np.abs(u - v)))
