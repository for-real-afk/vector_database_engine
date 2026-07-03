import pytest
from sqlalchemy import inspect
from models.database_models import (
    User, 
    Collection, 
    Document, 
    Chunk, 
    Embedding, 
    Metadata, 
    AuditLog,
    FloatArrayType,
    JSONBType
)

def test_user_model_schema():
    """Verify columns and constraints on the User model."""
    inspector = inspect(User)
    columns = {col.name: col for col in inspector.columns}
    
    assert "id" in columns
    assert "username" in columns
    assert "hashed_password" in columns
    assert "role" in columns
    assert "is_active" in columns
    assert "created_at" in columns
    
    assert columns["username"].unique is True
    assert columns["username"].nullable is False
    assert columns["role"].default.arg == "member"

def test_collection_model_schema():
    """Verify columns and constraints on the Collection model."""
    table = Collection.__table__
    columns = {col.name: col for col in table.columns}
    
    assert "id" in columns
    assert "name" in columns
    assert "namespace" in columns
    assert "dimension" in columns
    assert "metric" in columns
    assert "created_by" in columns
    
    assert columns["name"].nullable is False
    assert columns["namespace"].nullable is False
    assert columns["dimension"].nullable is False
    
    # Check unique constraint representation in table constraints
    uq_names = [c.name for c in table.constraints]
    assert any("uq_namespace_collection_name" in name for name in uq_names if name)

def test_document_model_schema():
    """Verify columns and constraints on the Document model."""
    table = Document.__table__
    columns = {col.name: col for col in table.columns}
    
    assert "id" in columns
    assert "collection_id" in columns
    assert "title" in columns
    assert "content_hash" in columns
    assert "status" in columns
    
    # Foreign keys
    fk_list = list(table.foreign_keys)
    assert len(fk_list) == 1
    fk = fk_list[0]
    assert fk.column.table.name == "collections"

def test_chunk_model_schema():
    """Verify columns and constraints on the Chunk model."""
    table = Chunk.__table__
    columns = {col.name: col for col in table.columns}
    
    assert "id" in columns
    assert "document_id" in columns
    assert "text_content" in columns
    assert "chunk_index" in columns
    
    fk_list = list(table.foreign_keys)
    assert len(fk_list) == 1
    assert fk_list[0].column.table.name == "documents"

def test_embedding_model_schema():
    """Verify columns and constraints on the Embedding model."""
    table = Embedding.__table__
    columns = {col.name: col for col in table.columns}
    
    assert "id" in columns
    assert "chunk_id" in columns
    assert "segment_id" in columns
    assert "vector_idx" in columns
    assert "vector_data" in columns
    
    # Verify vector_data is our dialect-aware FloatArrayType
    assert isinstance(columns["vector_data"].type, FloatArrayType)
    
    fk_list = list(table.foreign_keys)
    assert len(fk_list) == 1
    assert fk_list[0].column.table.name == "chunks"

def test_metadata_model_schema():
    """Verify columns and constraints on the Metadata model."""
    table = Metadata.__table__
    columns = {col.name: col for col in table.columns}
    
    assert "id" in columns
    assert "document_id" in columns
    assert "chunk_id" in columns
    assert "key" in columns
    assert "value" in columns
    
    # Verify value is our dialect-aware JSONBType
    assert isinstance(columns["value"].type, JSONBType)

def test_audit_log_model_schema():
    """Verify columns and constraints on the AuditLog model."""
    table = AuditLog.__table__
    columns = {col.name: col for col in table.columns}
    
    assert "id" in columns
    assert "user_id" in columns
    assert "action" in columns
    assert "target_id" in columns
    assert "details" in columns
