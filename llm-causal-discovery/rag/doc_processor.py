from pathlib import Path
from typing import List

import tiktoken
from docling.document_converter import DocumentConverter

from ..config.settings import ChunkingConfig


class DocumentProcessor:

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config
        self.converter = DocumentConverter()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def extract_markdown(self, pdf_path: str) -> str:
        doc = self.converter.convert(pdf_path)
        markdown = doc.document.export_to_markdown()

        output_path = Path(pdf_path).with_suffix(".md")
        output_path.write_text(markdown, encoding="utf-8")
        return markdown

    def chunk(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        step = self.config.chunk_size - self.config.overlap
        chunks: List[str] = []

        for start in range(0, len(tokens), step):
            window = tokens[start: start + self.config.chunk_size]
            if not window:
                break
            chunks.append(self.tokenizer.decode(window))

        return chunks