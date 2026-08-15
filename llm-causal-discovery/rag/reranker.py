from typing import Dict, List, Tuple
from sentence_transformers import CrossEncoder

from ..config.settings import ModelConfig, RetrievedDocument


class MedicalReranker:

    def __init__(self, config: ModelConfig) -> None:
        self.model = CrossEncoder(config.reranker_model)

    def rerank(
            self,
            query: str,
            candidates: List[Tuple[str, float, Dict]],
            top_k: int = 3,
    ) -> List[RetrievedDocument]:
        pool = candidates[:15]
        if not pool:
            return []

        pairs = [(query, c[2]["text"]) for c in pool]
        scores = self.model.predict(pairs)

        reranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)
        return [
            RetrievedDocument(
                text=doc[2]["text"],
                metadata={k: v for k, v in doc[2].items() if k != "text"},
                score=float(score),
            )
            for doc, score in reranked[:top_k]
        ]