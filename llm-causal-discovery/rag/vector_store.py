from typing import Dict, List
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from ..config.settings import QdrantConfig, MedicalChunk

from utils import text_to_uuid


class QdrantVectorStore:

    def __init__(self, config: QdrantConfig, dense_dim: int = 384) -> None:
        self.config = config
        self.dense_dim = dense_dim
        self.client = self._build_client()
        self._ensure_collection()

    def _build_client(self) -> QdrantClient:
        if not self.config.url:
            raise ValueError("QdrantConfig.url must be set.")
        return QdrantClient(
            url=self.config.url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            prefer_grpc=self.config.prefer_grpc,
        )

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.config.collection_name in existing:
            return

        self.client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config={"dense": VectorParams(size=self.dense_dim, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

    def index_chunks(self, chunks: List[MedicalChunk], batch_size: int = 100) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            points = [
                PointStruct(
                    id=text_to_uuid(chunk.id),
                    vector={
                        "dense": chunk.dense_vector,
                        "sparse": SparseVector(
                            indices=chunk.sparse_vector["indices"],
                            values=chunk.sparse_vector["values"],
                        ),
                    },
                    payload={"text": chunk.text, "original_id": chunk.id, **chunk.metadata},
                )
                for chunk in batch
            ]
            self.client.upsert(collection_name=self.config.collection_name, points=points)

    def hybrid_search(
            self,
            query_dense: List[float],
            query_sparse: Dict,
            limit: int = 30,
    ) -> Dict[str, List]:

        dense_response = self.client.query_points(
            collection_name=self.config.collection_name,
            query=query_dense,
            using="dense",
            limit=limit,
            with_payload=True,
        )

        sparse_response = self.client.query_points(
            collection_name=self.config.collection_name,
            query=SparseVector(
                indices=query_sparse["indices"],
                values=query_sparse["values"],
            ),
            using="sparse",
            limit=limit,
            with_payload=True,
        )

        return {
            "dense": dense_response.points,
            "sparse": sparse_response.points,
        }

    def get_stats(self) -> Dict:
        info = self.client.get_collection(self.config.collection_name)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }