import os
import uuid
import time
import tempfile
import pytest
from sqlalchemy.orm import sessionmaker

from models.database_models import Collection, TaskLog, Document, Embedding
from services.worker.worker import BackgroundWorker
from core.config import settings

def test_worker_polling_and_execution_success(db_engine):
    session_factory = sessionmaker(bind=db_engine)
    collection_id = uuid.uuid4()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.STORAGE_ROOT = temp_dir
        
        # 1. Setup Collection in a short-lived transaction
        session = session_factory()
        col = Collection(
            id=collection_id,
            name="worker_col",
            namespace="default",
            dimension=16,
            metric="Cosine"
        )
        session.add(col)
        session.commit()
        session.close()
        
        # 2. Instantiate worker
        worker = BackgroundWorker(session_factory)
        
        # 3. Submit valid document ingestion task
        payload = {
            "collection_id": str(collection_id),
            "title": "Async Ingest",
            "text_content": "Asynchronous background ingestion is working.",
            "dimension": 16,
            "chunk_size": 100
        }
        task_id = worker.submit_task("BATCH_INGEST", payload)
        
        # Verify status is pending initially
        session = session_factory()
        task = session.query(TaskLog).filter(TaskLog.id == task_id).first()
        assert task.status == "pending"
        session.close()
        
        # 4. Start worker daemon
        worker.start()
        
        # Poll for completion (up to 3 seconds)
        completed = False
        for _ in range(30):
            time.sleep(0.1)
            check_session = session_factory()
            check_task = check_session.query(TaskLog).filter(TaskLog.id == task_id).first()
            if check_task.status == "completed":
                completed = True
                check_session.close()
                break
            check_session.close()
            
        worker.stop()
        
        assert completed is True
        
        # 5. Verify document was ingested successfully
        session = session_factory()
        doc = session.query(Document).filter(Document.title == "Async Ingest").first()
        assert doc is not None
        assert doc.status == "completed"
        session.close()

def test_worker_error_recording(db_engine):
    session_factory = sessionmaker(bind=db_engine)
    worker = BackgroundWorker(session_factory)
    
    # Submit task with missing fields (will raise ValueError/KeyError)
    payload = {
        "collection_id": str(uuid.uuid4()),
        # missing title and text_content
    }
    task_id = worker.submit_task("BATCH_INGEST", payload)
    
    worker.start()
    
    failed = False
    for _ in range(30):
        time.sleep(0.1)
        check_session = session_factory()
        check_task = check_session.query(TaskLog).filter(TaskLog.id == task_id).first()
        if check_task.status == "failed":
            failed = True
            assert "KeyError" in check_task.error_message or "ValueError" in check_task.error_message
            check_session.close()
            break
        check_session.close()
        
    worker.stop()
    assert failed is True

def test_worker_crash_recovery(db_engine):
    session_factory = sessionmaker(bind=db_engine)
    worker = BackgroundWorker(session_factory)
    
    # Manually insert a task marked as "processing" in a short-lived transaction
    session = session_factory()
    task_id = uuid.uuid4()
    task = TaskLog(
        id=task_id,
        task_type="BATCH_INGEST",
        status="processing",
        payload={"dummy": "data"}
    )
    session.add(task)
    session.commit()
    session.close()
    
    # Run recovery
    worker.recover_interrupted_tasks()
    
    # Assert status is successfully reset to pending
    session = session_factory()
    task_rec = session.query(TaskLog).filter(TaskLog.id == task_id).first()
    assert task_rec.status == "pending"
    assert "Recovered" in task_rec.error_message
    session.close()
