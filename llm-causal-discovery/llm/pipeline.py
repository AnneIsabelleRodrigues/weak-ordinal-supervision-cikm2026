import gc
from pathlib import Path
from typing import Dict, List, Optional

from ..config.settings import RAGConfig, MedicalChunk, RetrievedDocument

from llm.generator import MedGemmaGenerator
from utils import _free_cuda
from rag.doc_processor import DocumentProcessor
from rag.embebeder import HybridEmbedder
from rag.rank_fusion import ReciprocalRankFusion
from rag.reranker import MedicalReranker
from rag.vector_store import QdrantVectorStore


class MedicalRAGPipeline:

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.doc_processor = DocumentProcessor(config.chunking)
        self.embedder = HybridEmbedder(config.embedding)
        self.vector_store = QdrantVectorStore(config.qdrant, dense_dim=config.embedding.dense_dim)
        self.rrf = ReciprocalRankFusion()
        self.reranker = MedicalReranker(config.model) if config.use_reranker else None
        self.generator = MedGemmaGenerator(config.model) if config.use_llm else None


    def index_document(self, pdf_path: str, metadata: Optional[Dict] = None) -> None:

        markdown = self.doc_processor.extract_markdown(pdf_path)
        text_chunks = self.doc_processor.chunk(markdown)

        self.embedder.fit_sparse(text_chunks)

        dense_vecs = self.embedder.encode_dense(text_chunks)
        stem = Path(pdf_path).stem

        medical_chunks: List[MedicalChunk] = []
        for i, (text, dense_vec) in enumerate(zip(text_chunks, dense_vecs)):
            sparse_vec = self.embedder.encode_sparse(text)
            medical_chunks.append(
                MedicalChunk(
                    id=f"{stem}_{i}",
                    text=text,
                    dense_vector=dense_vec,
                    sparse_vector=sparse_vec,
                    metadata={"source": Path(pdf_path).name, "chunk_id": i, **(metadata or {})},
                )
            )

        self.vector_store.index_chunks(medical_chunks)
        gc.collect()
        _free_cuda()

    def index_documents(self, documents: List[Dict]) -> Dict:

        success: List[str] = []
        failed: List[Dict] = []
        total = len(documents)

        for i, doc in enumerate(documents, start=1):
            path = doc.get("path", "")
            metadata = doc.get("metadata") or {}


            if not Path(path).exists():
                msg = f"File not found: {path}"
                failed.append({"path": path, "error": msg})
                continue

            try:
                self.index_document(path, metadata)
                success.append(path)
            except Exception as exc:
                msg = str(exc)
                failed.append({"path": path, "error": msg})

        return {"success": success, "failed": failed}

    def _retrieve_and_rerank(
            self,
            system_instruction: str,
            query: str,
            top_k: int,
    ) -> List[RetrievedDocument]:

        combined = system_instruction + query

        query_dense = self.embedder.encode_dense([combined])[0]
        query_sparse = self.embedder.encode_sparse(combined)

        raw = self.vector_store.hybrid_search(
            query_dense=query_dense,
            query_sparse=query_sparse,
            limit=30,
        )

        fused = self.rrf.fuse(raw["dense"], raw["sparse"], alpha=0.6)

        if not fused:
            return []

        if self.reranker:
            return self.reranker.rerank(query=combined, candidates=fused, top_k=top_k)

        return [
            RetrievedDocument(
                text=doc[2]["text"],
                metadata={k: v for k, v in doc[2].items() if k != "text"},
                score=doc[1],
            )
            for doc in fused[:top_k]
        ]

    def retrieve(
            self,
            system_instruction: str,
            query: str,
            top_k: int = 2,
    ) -> Dict:

        docs = self._retrieve_and_rerank(system_instruction, query, top_k)
        return {
            "context": docs,
            "confidence": docs[0].score if docs else 0.0,
        }

    def query(
            self,
            system_instruction: str,
            query: str,
            top_k: int = 2,
            use_generation: Optional[bool] = None,
    ) -> Dict:

        try:

            if use_generation is None:
                use_generation = self.config.use_llm and self.generator is not None

            docs = self._retrieve_and_rerank(system_instruction, query, top_k)

            answer: Optional[str] = None
            if use_generation and self.generator and docs:
                answer = self.generator.generate(
                    system_instruction=system_instruction,
                    query=query,
                    context_docs=docs,
                    max_new_tokens=self.config.model.max_new_tokens,
                )

            return {
                "answer": answer,
                "context": docs,
                "confidence": docs[0].score if docs else 0.0,
            }

        except Exception as e:
            import traceback

            print("Erro na função query:")
            print(str(e))
            traceback.print_exc()

            return {
                "answer": None,
                "context": [],
                "confidence": 0.0,
                "error": str(e),
            }


    def get_stats(self) -> Dict:
        return self.vector_store.get_stats()