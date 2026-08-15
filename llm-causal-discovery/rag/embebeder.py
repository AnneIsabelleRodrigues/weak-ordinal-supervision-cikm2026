from typing import Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config.settings import EmbeddingConfig

from utils import _free_cuda


class HybridEmbedder:

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dense_encoder = SentenceTransformer(config.dense_model, device=device)
        self._vectorizer: Optional[TfidfVectorizer] = None


    def encode_dense(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.dense_encoder.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        _free_cuda()
        return embeddings.tolist()

    def fit_sparse(self, corpus: List[str]) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=self.config.sparse_max_features,
            ngram_range=(1, 2),
            min_df=1,
        )
        self._vectorizer.fit(corpus)

    def encode_sparse(self, text: str) -> Dict:
        if self._vectorizer is None:
            self.fit_sparse([text])

        vec = self._vectorizer.transform([text]).toarray()[0]
        indices = vec.nonzero()[0].tolist()
        values = vec[indices].tolist()
        return {"indices": indices, "values": values}