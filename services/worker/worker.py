import time
import uuid
import logging
import traceback
import threading
from typing import Optional
from sqlalchemy.orm import Session, sessionmaker

from models.database_models import TaskLog, Collection, Embedding, Chunk, Document
from services.ingestion.pipeline import IngestionPipeline
from services.chunking.chunkers import FixedSizeChunker
from embeddings.providers import MockEmbeddingProvider
from storage.segments.compactor import SegmentCompactor
from storage.segments.writer import BinarySegmentSerializer
from index.graph.hnsw import HNSWIndex
from index.serialization.persistence import HNSWIndexManager
from core.config import settings

logger = logging.getLogger(__name__)

class BackgroundWorker:
    """
    Durable, database-backed background task worker.
    Runs in a separate thread, consuming pending jobs and logging execution outcomes.
    """
    def __init__(self, session_factory: sessionmaker, cache_manager=None):
        self.session_factory = session_factory
        self.cache_manager = cache_manager
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Start the background worker thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Background worker started.")

    def stop(self):
        """Signal the worker to stop and wait for the thread to exit."""
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("Background worker stopped.")

    def recover_interrupted_tasks(self):
        """
        Identify tasks left in 'processing' status (e.g. from a server crash)
        and reset them back to 'pending' for retry execution.
        """
        db = self.session_factory()
        try:
            stuck_tasks = db.query(TaskLog).filter(TaskLog.status == "processing").all()
            if stuck_tasks:
                logger.info(f"Recovering {len(stuck_tasks)} stuck tasks left in 'processing' state.")
                for task in stuck_tasks:
                    task.status = "pending"
                    task.error_message = "Recovered after unexpected shutdown/crash."
                db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Task recovery failed: {e}")
        finally:
            db.close()

    def submit_task(self, task_type: str, payload: dict) -> uuid.UUID:
        """Helper to create and write a new pending task log record."""
        db = self.session_factory()
        try:
            task = TaskLog(
                id=uuid.uuid4(),
                task_type=task_type,
                status="pending",
                payload=payload
            )
            db.add(task)
            db.commit()
            logger.info(f"Submitted task {task.id} (Type: {task_type})")
            return task.id
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to submit task: {e}")
            raise e
        finally:
            db.close()

    def _run_loop(self):
        # 1. Recover crash logs once on startup
        self.recover_interrupted_tasks()

        while self.running:
            db = self.session_factory()
            task_id = None
            try:
                # 2. Poll for the oldest pending task
                # We lock the record inside a single short transaction
                task = db.query(TaskLog)\
                    .filter(TaskLog.status == "pending")\
                    .order_by(TaskLog.created_at.asc())\
                    .first()

                if task:
                    task_id = task.id
                    task.status = "processing"
                    db.commit()
                    logger.info(f"Locked task {task_id} for execution.")
                else:
                    db.close()
                    time.sleep(0.5)
                    continue
            except Exception as e:
                db.rollback()
                db.close()
                logger.error(f"Error polling tasks: {e}")
                time.sleep(1.0)
                continue

            # 3. Execute the locked task
            task_session = self.session_factory()
            try:
                task_record = task_session.query(TaskLog).filter(TaskLog.id == task_id).first()
                self._dispatch_task(task_session, task_record)
                
                # Success
                task_record.status = "completed"
                task_session.commit()
                logger.info(f"Task {task_id} execution completed successfully.")
            except Exception as e:
                task_session.rollback()
                # Log traceback on failure
                tb_str = traceback.format_exc()
                logger.error(f"Task {task_id} failed: {e}\n{tb_str}")
                
                # Update status in a fresh transaction to ensure log record updates
                error_session = self.session_factory()
                try:
                    failed_task = error_session.query(TaskLog).filter(TaskLog.id == task_id).first()
                    if failed_task:
                        failed_task.status = "failed"
                        failed_task.error_message = f"{e}\n{tb_str}"
                        error_session.commit()
                except Exception as ex:
                    error_session.rollback()
                    logger.error(f"Failed to write error logs to task: {ex}")
                finally:
                    error_session.close()
            finally:
                task_session.close()

    def _dispatch_task(self, db: Session, task: TaskLog):
        """Invoke corresponding executors based on task type."""
        task_type = task.task_type
        payload = task.payload or {}

        if task_type == "BATCH_INGEST":
            collection_id = uuid.UUID(payload["collection_id"])
            title = payload["title"]
            text_content = payload["text_content"]
            metadata_dict = payload.get("metadata_dict")
            user_id_str = payload.get("user_id")
            user_id = uuid.UUID(user_id_str) if user_id_str else None

            # Setup defaults for mock pipeline
            provider = MockEmbeddingProvider(dimension=payload.get("dimension", 1536))
            chunker = FixedSizeChunker(chunk_size=payload.get("chunk_size", 200))
            pipeline = IngestionPipeline(db, provider, chunker)
            pipeline.ingest_document(collection_id, title, text_content, metadata_dict, user_id)

        elif task_type == "COMPACT_SEGMENTS":
            collection_id = uuid.UUID(payload["collection_id"])
            compactor = SegmentCompactor(db, self.cache_manager)
            compactor.compact(collection_id)

        elif task_type == "BUILD_HNSW":
            collection_id = uuid.UUID(payload["collection_id"])
            segment_id = uuid.UUID(payload["segment_id"])
            dimension = payload["dimension"]
            metric = payload.get("metric", "Cosine")

            # Load vectors from segment file
            segment_path = os.path.join(settings.STORAGE_ROOT, "segments", f"{segment_id}.bin")
            if not os.path.exists(segment_path):
                raise FileNotFoundError(f"Segment file {segment_path} not found.")

            with open(segment_path, "rb") as f:
                data = f.read()
            _, records = BinarySegmentSerializer.deserialize(data, dimension)

            # Build HNSW index graph
            hnsw_idx = HNSWIndex(dimension=dimension, metric=metric)
            segment_mappings = {}
            for idx, record in enumerate(records):
                hnsw_idx.insert(record['id'], record['vector'])
                segment_mappings[record['id']] = (segment_id, idx)

            # Save snapshot
            snapshot_dir = os.path.join(settings.STORAGE_ROOT, "snapshots", str(segment_id))
            HNSWIndexManager.snapshot(snapshot_dir, hnsw_idx, segment_mappings)

        else:
            raise ValueError(f"Unknown task type: {task_type}")
