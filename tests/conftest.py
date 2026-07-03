import os
import pytest
import tempfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.database import Base

@pytest.fixture(scope="session")
def db_engine():
    # Use a file-backed SQLite database in a temp directory to allow sharing
    # data across multiple connections/threads during background worker tests.
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_engine.db")
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    
    # Explicitly import models to register them on Base.metadata
    import models.database_models
    
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    
    # Cleanup temp file and folder
    try:
        engine.dispose()
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    except Exception:
        pass

@pytest.fixture(scope="function")
def db_session(db_engine):
    connection = db_engine.connect()
    transaction = connection.begin()
    
    Session = sessionmaker(bind=connection)
    session = Session()
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()
