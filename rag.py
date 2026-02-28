import asyncio
import logging
import json
from typing import List
from datetime import datetime

import requests
import google.generativeai as genai
from pinecone import Pinecone

from models import SearchResult, RAGResponse, Intent
from config import Config
from database import get_chat_history, save_message

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index = self.pc.Index(Config.PINECONE_INDEX)
        
        # Using Gemini 1.5 Flash - it's stable and fast
        self.generation_model_id = "gemini-1.5-flash"
        logger.info("RAG Pipeline initialized using %s", self.generation_model_id)

    async def classify_intent(self, text: str) -> Intent:
        """Classify user intent using a fast LLM call."""
        try:
            prompt = (
                "You are an intent classifier. Classify the user's message into EXACTLY one category: "
                "GREETING, NEWS_QUERY, SUBSCRIPTION, GENERAL_CHAT, UNKNOWN.\n\n"
                f"MESSAGE: \"{text}\"\n\n"
                "CATEGORY:"
            )
            # Use direct generation for fast classification
            category = await self.generate_with_retry(prompt)
            category = category.strip().upper()
            
            for intent in Intent:
                if intent.value.upper() in category:
                    return intent
            return Intent.UNKNOWN
        except Exception as e:
            logger.error("Intent classification failed: %s", e)
            return Intent.NEWS_QUERY  # Default for safety

    def _embed_sync(self, text: str) -> List[float]:
        """Call Google AI v1beta embedding API directly over HTTP."""
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
                if match.score < 0.3:  # Lowered threshold to ensure context is found
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

    def _generate_sync(self, prompt: str) -> str:
        """Call Google AI v1 generation API directly over HTTP with the correct structure."""
        url = (
            f"https://generativelanguage.googleapis.com/v1/models/"
            f"{self.generation_model_id}:generateContent?key={Config.GEMINI_API_KEY}"
        )
        
        # Correctly structured payload according to the Gemini API spec
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        }
        
        resp = requests.post(url, json=payload, timeout=60)
        if not resp.ok:
            logger.error("Generation HTTP error %s: %s", resp.status_code, resp.text)
            resp.raise_for_status()
            
        data = resp.json()
        
        if "candidates" not in data or not data["candidates"]:
            if "promptFeedback" in data:
                logger.warning("Prompt feedback blocked: %s", data["promptFeedback"])
            return "I couldn't summarize the news from those sources."
            
        candidate = data["candidates"][0]
        if "content" not in candidate or "parts" not in candidate["content"]:
            logger.warning("No content in response candidate: %s", candidate)
            return "I couldn't summarize the news from those sources."
            
        return candidate["content"]["parts"][0]["text"]

    async def generate_with_retry(self, prompt: str) -> str:
        for attempt in range(Config.MAX_RETRIES):
            try:
                return await asyncio.to_thread(self._generate_sync, prompt)
            except Exception as exc:
                logger.error("Generation error on attempt %d: %s", attempt + 1, str(exc))
                if attempt == Config.MAX_RETRIES - 1:
                    return f"I encountered an error generating the summary. (Ref: {str(exc)[:50]})"

                backoff = Config.INITIAL_BACKOFF_SECONDS * (2**attempt)
                await asyncio.sleep(backoff)

    async def answer_question(self, user_id: int, question: str) -> RAGResponse:
        try:
            query_embedding = await self.embed_with_retry(question)
            
            # 1. INITIAL RETRIEVAL: Fetch 15 chunks (instead of 5)
            results = self.index.query(
                vector=query_embedding,
                top_k=15,
                include_metadata=True,
            )

            # 2. FILTER & FORMAT: Prepare chunks for re-ranking
            initial_chunks = []
            for match in results.matches:
                metadata = match.metadata or {}
                initial_chunks.append({
                    "id": match.id,
                    "text": metadata.get("chunk_text", ""),
                    "source": metadata.get("source_name", ""),
                    "url": metadata.get("source_url", ""),
                    "date": metadata.get("published_date", ""),
                    "score": match.score
                })

            if not initial_chunks:
                return RAGResponse(answer="I couldn't find any news about that topic.", sources=[], confidence=0.0, timestamp=datetime.now())

            # 3. LLM RE-RANKING: Use Gemini to pick the top 5 most relevant chunks
            rerank_prompt = (
                f"Question: {question}\n\n"
                "I have found 15 potential news snippets. Rate each one from 0 to 10 based on how relevant it is to the question. "
                "Output ONLY a JSON list of the top 5 indices (0-14) that best answer the question.\n\n"
                "SNIPPETS:\n"
            )
            for i, chunk in enumerate(initial_chunks):
                rerank_prompt += f"[{i}] {chunk['text'][:200]}...\n"

            rerank_resp = await self.generate_with_retry(rerank_prompt)
            
            # Parse top indices (fallback to first 5 if parsing fails)
            try:
                import re
                top_indices = [int(i) for i in re.findall(r'\d+', rerank_resp)][:5]
            except:
                top_indices = list(range(min(5, len(initial_chunks))))

            # 4. FINAL SELECTION: Pick the re-ranked top 5
            search_results = []
            for idx in top_indices:
                if idx < len(initial_chunks):
                    c = initial_chunks[idx]
                    search_results.append(SearchResult(
                        chunk_text=c["text"],
                        source_url=c["url"],
                        source_name=c["source"],
                        similarity_score=c["score"],
                        published_date=c["date"]
                    ))

            # 5. GENERATE FINAL SUMMARY (Same as before but with better data)
            context = "\n\n".join(
                [f"Source: {r.source_name}\nDate: {r.published_date}\n{r.chunk_text}" for r in search_results]
            )

            user_history = get_chat_history(user_id, limit=Config.CONTEXT_WINDOW)
            history_text = "\n".join(user_history) if user_history else ""

            prompt = (
                "You are an expert AI news analyst. Summarize the latest news based on the context.\n\n"
                f"NEWS CONTEXT:\n{context}\n\n"
                f"CONVERSATION HISTORY:\n{history_text}\n\n"
                f"USER QUESTION:\n{question}\n\n"
                "SUMMARY:"
            )

            answer = await self.generate_with_retry(prompt)
            save_message(user_id, "user", question)
            save_message(user_id, "assistant", answer)

            return RAGResponse(
                answer=answer,
                sources=search_results,
                confidence=sum(r.similarity_score for r in search_results) / len(search_results),
                timestamp=datetime.now(),
            )

        except Exception as exc:
            logger.error("RAG pipeline error: %s", exc)
            return RAGResponse(answer="Something went wrong.", sources=[], confidence=0.0, timestamp=datetime.now())
