import uuid
from datetime import datetime
from sqlalchemy import (
    Column, 
    String, 
    Integer, 
    Text, 
    DateTime, 
    ForeignKey, 
    Boolean, 
    UniqueConstraint,
    ForeignKeyConstraint,
    Float
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.orm import relationship
from core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="member")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    collections = relationship("Collection", back_populates="creator", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")


class Collection(Base):
    __tablename__ = "collections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    namespace = Column(String(100), nullable=False, index=True)
    dimension = Column(Integer, nullable=False)
    metric = Column(String(50), nullable=False, default="Cosine") # Cosine, L2, DotProduct, Manhattan
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Constraints
    __table_args__ = (
        UniqueConstraint("namespace", "name", name="uq_namespace_collection_name"),
    )

    # Relationships
    creator = relationship("User", back_populates="collections")
    documents = relationship("Document", back_populates="collection", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(UUID(as_uuid=True), ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=True)
    content_hash = Column(String(64), nullable=True, index=True)
    status = Column(String(50), nullable=False, default="pending") # pending, processing, completed, failed
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    collection = relationship("Collection", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    metadata_records = relationship("Metadata", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    text_content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks")
    embeddings = relationship("Embedding", back_populates="chunk", cascade="all, delete-orphan")
    metadata_records = relationship("Metadata", back_populates="chunk", cascade="all, delete-orphan")


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False)
    segment_id = Column(UUID(as_uuid=True), nullable=True, index=True) # References growing/sealed segment filename ID
    vector_idx = Column(Integer, nullable=True) # Byte offset or sequence idx in segment vector block
    vector_data = Column(ARRAY(Float), nullable=False) # Source of truth floats
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    chunk = relationship("Chunk", back_populates="embeddings")


class Metadata(Base):
    __tablename__ = "metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=True)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=True)
    key = Column(String(100), nullable=False, index=True)
    value = Column(JSONB, nullable=False) # Stores scalar or structured nested properties
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="metadata_records")
    chunk = relationship("Chunk", back_populates="metadata_records")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(100), nullable=False) # INGEST, SEARCH, DELETE, LOGIN, UPDATE
    target_id = Column(UUID(as_uuid=True), nullable=True) # UUID of collection/document/etc.
    details = Column(JSONB, nullable=True) # Additional contextual attributes
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="audit_logs")
