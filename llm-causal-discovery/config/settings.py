from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config.env import COLLECTION_NAME, QDRANT_CLUSTER_URL, QDRANT_API_KEY


@dataclass
class EmbeddingConfig:
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_dim: int = 384
    sparse_max_features: int = 500
    batch_size: int = 32


@dataclass
class QdrantConfig:
    collection_name: str = COLLECTION_NAME
    url: Optional[str] = QDRANT_CLUSTER_URL
    api_key: Optional[str] = QDRANT_API_KEY
    timeout: int = 120
    prefer_grpc: bool = False


@dataclass
class ModelConfig:
    llm_model: str = "google/medgemma-27b-text-it"
    use_quantization: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_new_tokens: int = 4096


@dataclass
class ChunkingConfig:
    chunk_size: int = 1024
    overlap: int = 150

    def __post_init__(self) -> None:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")


@dataclass
class RAGConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    use_reranker: bool = True
    use_llm: bool = True


@dataclass
class MedicalChunk:
    id: str
    text: str
    metadata: Dict
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict] = None


@dataclass
class RetrievedDocument:
    text: str
    metadata: Dict
    score: float