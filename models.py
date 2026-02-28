from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class Document(BaseModel):
    """Ingested news document."""

    id: str
    title: str
    content: str
    source_url: str
    source_name: str
    published_date: Optional[datetime] = None
    chunk_index: int = 0
    total_chunks: int = 1


class SearchResult(BaseModel):
    """Result from Pinecone vector search."""

    chunk_text: str
    source_url: str
    source_name: str
    similarity_score: float
    published_date: Optional[str] = None


class UserContext(BaseModel):
    """Store user conversation context."""

    user_id: int
    message: str
    timestamp: datetime
    embedding: Optional[List[float]] = None


class ChatMessage(BaseModel):
    """A single message in the chat history."""
    user_id: int
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = datetime.now()


class RAGResponse(BaseModel):
    """Response from RAG pipeline."""

    answer: str
    sources: List[SearchResult]
    confidence: float
    timestamp: datetime


class TelegramMessage(BaseModel):
    """Telegram update payload."""

    update_id: int
    message: Optional[dict] = None
