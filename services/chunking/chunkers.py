import re
import logging
from typing import Protocol
import numpy as np
from embeddings.providers import EmbeddingProvider

logger = logging.getLogger(__name__)

class Chunker(Protocol):
    """
    Protocol defining the interface for all text chunking strategies.
    """
    def chunk(self, text: str) -> list[str]:
        """Split a source text string into a list of smaller text chunks."""
        ...


class FixedSizeChunker:
    """
    Simple chunker that splits text strictly by character length thresholds.
    """
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]


class SlidingWindowChunker:
    """
    Splits text into chunks of a given character size with a sliding overlap.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("Overlap size must be smaller than chunk size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += step
            
        return chunks


class RecursiveCharacterChunker:
    """
    Recursively splits text using a list of priority separators (like double newlines,
    single newlines, spaces, etc.) to keep chunk sizes within a target threshold.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, separators: list[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def chunk(self, text: str) -> list[str]:
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Fallback if no separators remain: split strictly at chunk_size
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator = separators[0]
        next_separators = separators[1:]
        
        # Split text by the current separator
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        final_chunks = []
        current_chunk = []
        current_len = 0

        for part in splits:
            part_len = len(part)
            # If a single part is larger than chunk_size, recursively split it with the next separators
            if part_len > self.chunk_size:
                # Flush the current buffer first
                if current_chunk:
                    final_chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                sub_chunks = self._split_text(part, next_separators)
                final_chunks.extend(sub_chunks)
                continue

            # Check if adding the next part exceeds chunk size
            if current_len + part_len + (len(separator) if current_chunk else 0) <= self.chunk_size:
                current_chunk.append(part)
                current_len += part_len + (len(separator) if len(current_chunk) > 1 else 0)
            else:
                if current_chunk:
                    final_chunks.append(separator.join(current_chunk))
                
                # Keep some overlap from the end of the previous chunk if possible
                overlap_chunk = []
                overlap_len = 0
                # Take parts from the end of the current chunk to satisfy overlap
                for prev_part in reversed(current_chunk):
                    if overlap_len + len(prev_part) <= self.chunk_overlap:
                        overlap_chunk.insert(0, prev_part)
                        overlap_len += len(prev_part) + len(separator)
                    else:
                        break
                
                current_chunk = overlap_chunk + [part]
                current_len = sum(len(p) for p in current_chunk) + len(separator) * (len(current_chunk) - 1)

        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

        return final_chunks


class SemanticChunker:
    """
    Splits text by identifying boundaries where the semantic meaning changes significantly.
    Uses sentence tokenization and cosine similarity between sentence embedding vectors.
    """
    def __init__(self, embedding_provider: EmbeddingProvider, similarity_threshold: float = 0.75):
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Split text into sentences using simple regex
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return [sentences[0]]

        # Generate embeddings for all sentences in one batch
        embeddings = self.embedding_provider.embed_batch(sentences)
        
        # Calculate cosine similarity between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            v1 = np.array(embeddings[i])
            v2 = np.array(embeddings[i+1])
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(v1, v2) / (norm1 * norm2)
            else:
                sim = 0.0
            similarities.append(sim)

        chunks = []
        current_chunk_sentences = [sentences[0]]

        for i, sim in enumerate(similarities):
            # If similarity drops below threshold, split and start a new chunk
            if sim < self.similarity_threshold:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentences[i+1]]
            else:
                current_chunk_sentences.append(sentences[i+1])

        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks
