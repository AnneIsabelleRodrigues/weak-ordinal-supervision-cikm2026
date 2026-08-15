from typing import Dict, List, Tuple

class ReciprocalRankFusion:

    @staticmethod
    def fuse(
            dense_results: List,
            sparse_results: List,
            k: int = 60,
            alpha: float = 0.6,
    ) -> List[Tuple[str, float, Dict]]:
        scores: Dict[str, float] = {}
        payloads: Dict[str, Dict] = {}

        for rank, hit in enumerate(dense_results):
            scores[hit.id] = scores.get(hit.id, 0.0) + alpha / (k + rank)
            payloads[hit.id] = hit.payload

        for rank, hit in enumerate(sparse_results):
            scores[hit.id] = scores.get(hit.id, 0.0) + (1 - alpha) / (k + rank)
            payloads[hit.id] = hit.payload

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, payloads[doc_id]) for doc_id, score in ranked]