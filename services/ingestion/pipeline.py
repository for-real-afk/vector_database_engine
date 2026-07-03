import os
from typing import Tuple
import uuid
import hashlib
import logging
from sqlalchemy import func
from sqlalchemy.orm import Session
from models.database_models import Document, Chunk, Embedding, Metadata, AuditLog, Collection
from services.chunking.chunkers import Chunker
from embeddings.providers import EmbeddingProvider
from storage.segments.writer import BinarySegmentSerializer
from core.config import settings

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Orchestrates the document ingestion pipeline.
    Parses, chunks, embeds, records transactions in PostgreSQL, 
    and writes to physical binary segment caches on disk.
    """
    def __init__(self, db: Session, embedding_provider: EmbeddingProvider, chunker: Chunker):
        self.db = db
        self.embedding_provider = embedding_provider
        self.chunker = chunker
        # Ensure segments directory exists
        os.makedirs(os.path.join(settings.STORAGE_ROOT, "segments"), exist_ok=True)

    def ingest_document(
        self, 
        collection_id: uuid.UUID, 
        title: str, 
        text_content: str, 
        metadata_dict: dict = None,
        user_id: uuid.UUID = None
    ) -> Document:
        if not text_content:
            raise ValueError("Document text content cannot be empty.")

        # 1. Fetch collection to verify existence and dimensions
        collection = self.db.query(Collection).filter(Collection.id == collection_id).first()
        if not collection:
            raise ValueError(f"Collection with ID {collection_id} not found.")

        # 2. Check document content hash to prevent duplicate ingestion (Idempotency check)
        content_hash = hashlib.sha256(text_content.encode('utf-8')).hexdigest()
        existing_doc = self.db.query(Document).filter(
            Document.collection_id == collection_id,
            Document.content_hash == content_hash
        ).first()

        if existing_doc:
            if existing_doc.status == "completed":
                logger.info(f"Document already ingested. Returning existing ID: {existing_doc.id}")
                return existing_doc
            else:
                # If failed or pending, we will re-attempt ingestion: delete old incomplete chunks
                self.db.delete(existing_doc)
                self.db.commit()

        # 3. Create document record in "pending" status (relational lock)
        doc = Document(
            id=uuid.uuid4(),
            collection_id=collection_id,
            title=title,
            content_hash=content_hash,
            status="pending"
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)

        try:
            # 4. Chunk document
            chunks_text = self.chunker.chunk(text_content)
            if not chunks_text:
                raise ValueError("No text chunks generated from document.")

            # 5. Generate embeddings in a batch to minimize API overhead
            embeddings_list = self.embedding_provider.embed_batch(chunks_text)

            # 6. Determine target segment (Load existing active segment or create new UUID)
            segment_id, existing_records = self._get_or_create_active_segment(collection_id)
            
            # Map index coordinates
            start_index = len(existing_records)
            new_records = []
            
            # 7. Create database records in memory
            db_chunks = []
            db_embeddings = []
            db_metadata = []

            for idx, (chunk_text, vector) in enumerate(zip(chunks_text, embeddings_list)):
                chunk_id = uuid.uuid4()
                # A. Chunk Model
                chunk_obj = Chunk(
                    id=chunk_id,
                    document_id=doc.id,
                    text_content=chunk_text,
                    chunk_index=idx,
                    token_count=len(chunk_text.split()) # simple word split approximation
                )
                db_chunks.append(chunk_obj)

                # B. Embedding Model
                emb_id = uuid.uuid4()
                emb_obj = Embedding(
                    id=emb_id,
                    chunk_id=chunk_id,
                    segment_id=segment_id,
                    vector_idx=start_index + idx,
                    vector_data=vector
                )
                db_embeddings.append(emb_obj)

                # Assemble for binary file serialization
                new_records.append({
                    'id': emb_id,
                    'vector': vector,
                    'payload': {
                        'chunk_id': str(chunk_id),
                        'document_id': str(doc.id),
                        'text': chunk_text
                    },
                    'status': 1 # Active
                })

                # C. Associated Metadata Model (if provided)
                if metadata_dict:
                    for key, val in metadata_dict.items():
                        meta_obj = Metadata(
                            id=uuid.uuid4(),
                            document_id=doc.id,
                            chunk_id=chunk_id,
                            key=key,
                            value=val
                        )
                        db_metadata.append(meta_obj)

            # 8. Write to Binary Segment File on Disk
            combined_records = existing_records + new_records
            segment_bytes = BinarySegmentSerializer.serialize(
                segment_id, 
                combined_records, 
                collection.dimension
            )
            
            segment_path = os.path.join(settings.STORAGE_ROOT, "segments", f"{segment_id}.bin")
            with open(segment_path, "wb") as f:
                f.write(segment_bytes)

            # If the segment is now full, seal it by building and snapshotting HNSW graph
            if len(combined_records) >= settings.MAX_VECTORS_PER_SEGMENT:
                logger.info(f"Segment {segment_id} reached capacity ({len(combined_records)}). Sealing segment and building HNSW index.")
                from index.graph.hnsw import HNSWIndex
                from index.serialization.persistence import HNSWIndexManager
                
                hnsw_idx = HNSWIndex(
                    dimension=collection.dimension, 
                    metric=collection.metric,
                    M=16,
                    M0=32,
                    ef_construction=64
                )
                segment_mappings = {}
                for idx_pos, record in enumerate(combined_records):
                    hnsw_idx.insert(record['id'], record['vector'])
                    segment_mappings[record['id']] = (segment_id, idx_pos)
                
                snapshot_dir = os.path.join(settings.STORAGE_ROOT, "snapshots", str(segment_id))
                HNSWIndexManager.snapshot(snapshot_dir, hnsw_idx, segment_mappings)

            # 9. Write records to database in a transaction
            for c in db_chunks:
                self.db.add(c)
            for e in db_embeddings:
                self.db.add(e)
            for m in db_metadata:
                self.db.add(m)

            # Update status to completed
            doc.status = "completed"
            
            # Log audit
            audit = AuditLog(
                id=uuid.uuid4(),
                user_id=user_id,
                action="INGEST",
                target_id=doc.id,
                details={
                    "title": title,
                    "chunks_count": len(chunks_text),
                    "segment_id": str(segment_id)
                }
            )
            self.db.add(audit)
            
            self.db.commit()
            logger.info(f"Successfully completed ingestion for document: {title} (Chunks: {len(chunks_text)})")
            
        except Exception as e:
            self.db.rollback()
            # Mark document as failed
            doc.status = "failed"
            self.db.commit()
            logger.error(f"Ingestion pipeline failed for document '{title}': {e}")
            # If a new segment file was created but transaction failed, clean it up
            raise e

        self.db.refresh(doc)
        return doc

    def _get_or_create_active_segment(self, collection_id: uuid.UUID) -> Tuple[uuid.UUID, list[dict]]:
        """
        Find an active segment for the collection that has space available,
        or create a new segment ID and return an empty list of records.
        """
        # We query the database to find all segments linked to this collection
        active_segments = self.db.query(Embedding.segment_id, func.count(Embedding.id))\
            .join(Chunk)\
            .join(Document)\
            .filter(Document.collection_id == collection_id)\
            .filter(Embedding.segment_id != None)\
            .group_by(Embedding.segment_id)\
            .all()

        for seg_id, count in active_segments:
            # If segment is not full yet, load existing records
            if count < settings.MAX_VECTORS_PER_SEGMENT:
                segment_path = os.path.join(settings.STORAGE_ROOT, "segments", f"{seg_id}.bin")
                if os.path.exists(segment_path):
                    try:
                        with open(segment_path, "rb") as f:
                            data = f.read()
                        # Unpack existing records
                        # Retrieve dimension from collection
                        col = self.db.query(Collection).filter(Collection.id == collection_id).first()
                        _, records = BinarySegmentSerializer.deserialize(data, col.dimension)
                        return seg_id, records
                    except Exception as e:
                        logger.warning(f"Failed to read existing segment file {seg_id}. Creating new segment: {e}")

        # Fallback: create a new segment
        return uuid.uuid4(), []
