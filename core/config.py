import os
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database Config
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")
    POSTGRES_DB: str = Field(default="vector_engine")
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Engine Security
    JWT_SECRET_KEY: str = Field(default="antigravity_super_secret_retrieval_key_2026")
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60)

    # Ingestion & Segment parameters
    MAX_VECTORS_PER_SEGMENT: int = Field(default=50000)
    STORAGE_ROOT: str = Field(default="./storage")

    # Embeddings Provider Default
    EMBEDDING_PROVIDER: str = Field(default="gemini") # gemini | openai | huggingface
    EMBEDDING_API_KEY: str = Field(default="")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
