import asyncio
import logging
from typing import List
from datetime import datetime

import requests
import google.generativeai as genai
from pinecone import Pinecone

from models import SearchResult, RAGResponse
from config import Config

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index = self.pc.Index(Config.PINECONE_INDEX)
        self.genai_model = genai.GenerativeModel("gemini-1.5-flash")
        self.user_context = {}
        logger.info("RAG Pipeline initialized")

    def _embed_sync(self, text: str) -> List[float]:
        """Call Google AI v1 embedding API directly over HTTP."""
        model_id = Config.EMBEDDING_MODEL.replace("models/", "")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:embedContent?key={Config.GEMINI_API_KEY}"
        )
        payload = {
            "model": Config.EMBEDDING_MODEL,
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_QUERY",
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]

    async def embed_with_retry(self, text: str) -> List[float]:
        for attempt in range(Config.MAX_RETRIES):
            try:
                return await asyncio.to_thread(self._embed_sync, text)
            except Exception as exc:
                if attempt == Config.MAX_RETRIES - 1:
                    logger.error("Failed to embed after retries: %s", exc)
                    raise
                backoff = Config.INITIAL_BACKOFF_SECONDS * (2**attempt)
                logger.warning("Embedding retry %d/%d after %.1fs", attempt + 1, Config.MAX_RETRIES, backoff)
                await asyncio.sleep(backoff)

    async def search_pinecone(self, query_embedding: List[float], user_id: int) -> List[SearchResult]:
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
            )

            search_results = []
            for match in results.matches:
                if match.score < 0.5:
                    continue

                metadata = match.metadata or {}
                result = SearchResult(
                    chunk_text=metadata.get("chunk_text", ""),
                    source_url=metadata.get("source_url", ""),
                    source_name=metadata.get("source_name", ""),
                    similarity_score=match.score,
                    published_date=metadata.get("published_date", ""),
                )
                search_results.append(result)

            logger.info("Found %d relevant chunks", len(search_results))
            return search_results

        except Exception as exc:
            logger.error("Pinecone search failed: %s", exc)
            return []

    async def generate_with_retry(self, prompt: str) -> str:
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = await asyncio.to_thread(self.genai_model.generate_content, prompt)
                return response.text
            except Exception as exc:
                if attempt == Config.MAX_RETRIES - 1:
                    logger.error("Generation failed after retries: %s", exc)
                    return "I'm having trouble generating a response. Please try again in a moment."

                backoff = Config.INITIAL_BACKOFF_SECONDS * (2**attempt)
                logger.warning("Generation retry %d/%d after %.1fs", attempt + 1, Config.MAX_RETRIES, backoff)
                await asyncio.sleep(backoff)

    async def answer_question(self, user_id: int, question: str) -> RAGResponse:
        try:
            query_embedding = await self.embed_with_retry(question)
            search_results = await self.search_pinecone(query_embedding, user_id)

            if not search_results:
                return RAGResponse(
                    answer=(
                        "I couldn't find relevant AI news about that topic. "
                        "Try asking about recent AI announcements or specific companies."
                    ),
                    sources=[],
                    confidence=0.0,
                    timestamp=datetime.now(),
                )

            context = "\n\n".join(
                [
                    f"Source: {r.source_name}\nDate: {r.published_date}\n{r.chunk_text}"
                    for r in search_results
                ]
            )

            user_history = self.user_context.get(user_id, [])
            history_text = "\n".join(user_history[-3:]) if user_history else ""

            prompt = (
                "You are an AI news expert. Answer the user's question based on the provided news context.\n"
                "Be concise (2-3 sentences). If the context doesn't fully answer, say so.\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"CONVERSATION HISTORY:\n{history_text}\n\n"
                f"USER QUESTION:\n{question}\n\n"
                "ANSWER:"
            )

            answer = await self.generate_with_retry(prompt)

            self.user_context.setdefault(user_id, []).append(f"Q: {question}")
            self.user_context.setdefault(user_id, []).append(f"A: {answer}")

            if len(self.user_context[user_id]) > Config.CONTEXT_WINDOW * 2:
                self.user_context[user_id] = self.user_context[user_id][-(Config.CONTEXT_WINDOW * 2) :]

            avg_similarity = sum(r.similarity_score for r in search_results) / len(search_results)

            return RAGResponse(
                answer=answer,
                sources=search_results,
                confidence=avg_similarity,
                timestamp=datetime.now(),
            )

        except Exception as exc:
            logger.error("RAG pipeline error: %s", exc)
            return RAGResponse(
                answer="Something went wrong. Please try again.",
                sources=[],
                confidence=0.0,
                timestamp=datetime.now(),
            )
