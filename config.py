import os


class Config:
    """Load configuration from environment variables with validation."""

    # API Keys
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "ai-news")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Embedding Model (OpenAI)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "3072"))

    # Chunking Strategy
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "5"))
    QUEUE_MAX_SIZE: int = int(os.getenv("QUEUE_MAX_SIZE", "100"))

    # Ingestion Schedule
    INGEST_INTERVAL_HOURS: int = int(os.getenv("INGEST_INTERVAL_HOURS", "4"))
    MAX_SOURCES_TO_FETCH: int = int(os.getenv("MAX_SOURCES_TO_FETCH", "50"))

    # Pinecone Metadata
    CONTEXT_WINDOW: int = 5  # Store last 5 user messages

    # Retry Policy
    MAX_RETRIES: int = 3
    INITIAL_BACKOFF_SECONDS: float = 1.0

    @staticmethod
    def validate():
        required = ["TELEGRAM_TOKEN", "PINECONE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"]
        missing = [k for k in required if not getattr(Config, k)]
        if missing:
            raise ValueError(f"Missing required env vars: {missing}")


if __name__ == "__main__":
    Config.validate()
    print("Configuration valid")
