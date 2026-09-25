import uuid
from typing import List, Dict, Any


class TextChunker:
    """Chunks documents into semantic pieces with overlap and metadata."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_pages(self, pages_data: List[Dict[str, Any]], doc_id: str = None) -> List[Dict[str, Any]]:
        document_id = doc_id or f"doc_{uuid.uuid4().hex[:8]}"
        all_chunks = []
        chunk_counter = 1

        for page_info in pages_data:
            text = page_info.get("text", "")
            page_num = page_info.get("page", 1)
            filename = page_info.get("filename", "unknown")

            # Simple clean sliding window chunking
            start = 0
            text_len = len(text)

            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                chunk_text = text[start:end].strip()

                if chunk_text:
                    all_chunks.append({
                        "chunk_id": f"{document_id}_c{chunk_counter}",
                        "document_id": document_id,
                        "filename": filename,
                        "page": page_num,
                        "content": chunk_text
                    })
                    chunk_counter += 1

                if end == text_len:
                    break
                start += (self.chunk_size - self.chunk_overlap)

        return all_chunks


text_chunker = TextChunker()
