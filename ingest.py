import asyncio
import feedparser
import logging
from datetime import datetime
from typing import List, Set
import hashlib

import requests
from pinecone import Pinecone
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import Config

print("[INGEST] v3 loaded - direct HTTP embeddings")
logger = logging.getLogger(__name__)


class NewsIngester:
    """Fetch, chunk, embed, and store news articles."""

    RSS_FEEDS = [
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://feeds.reuters.com/reuters/technologyNews",
        "https://feeds.techcrunch.com/techcrunch/startups/",
        "https://news.ycombinator.com/rss",
    ]

    YOUTUBE_CHANNELS = [
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCJScVaHCI0e2v3w7wIGBdEg",
    ]

    def __init__(self):
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index = self.pc.Index(Config.PINECONE_INDEX)
        self.processed_hashes: Set[str] = set()
        logger.info("NewsIngester initialized (direct HTTP embeddings)")

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
            "taskType": "RETRIEVAL_DOCUMENT",
        }
        resp = requests.post(url, json=payload, timeout=30)
        if not resp.ok:
            logger.error("Embed HTTP error %s: %s", resp.status_code, resp.text)
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]

    async def fetch_full_content(self, url: str) -> str:
        """Fetch the full text of an article using the Jina Reader API."""
        try:
            # Jina Reader turns any URL into clean Markdown for LLMs
            jina_url = f"https://r.jina.ai/{url}"
            headers = {
                "X-Return-Format": "markdown",
                "Accept": "text/event-stream",
            }
            async with httpx.AsyncClient() as client:
                resp = await client.get(jina_url, headers=headers, timeout=30)
                if resp.status_code == 200:
                    content = resp.text.strip()
                    # If content is too short, fall back to the RSS summary
                    if len(content) > 100:
                        return content
            return ""
        except Exception as e:
            logger.warning("Failed to fetch full content for %s: %s", url, e)
            return ""

    async def fetch_rss_feeds(self) -> List[dict]:
        articles = []
        feed_urls = self.RSS_FEEDS + self.YOUTUBE_CHANNELS

        for feed_url in feed_urls:
            try:
                logger.info("Fetching %s", feed_url)
                feed = feedparser.parse(feed_url)

                for entry in feed.entries[: Config.MAX_SOURCES_TO_FETCH]:
                    article = {
                        "title": entry.get("title", ""),
                        "content": entry.get("summary", ""),
                        "source_url": entry.get("link", ""),
                        "source_name": feed.feed.get("title", "RSS Feed"),
                        "published_date": self._parse_date(entry.get("published", "")),
                    }

                    if article["content"]:
                        articles.append(article)

            except Exception as exc:
                logger.error("Error fetching %s: %s", feed_url, exc)
                continue

        logger.info("Fetched %d articles", len(articles))
        return articles

    def _parse_date(self, date_str: str) -> datetime:
        try:
            from email.utils import parsedate_to_datetime

            return parsedate_to_datetime(date_str)
        except Exception:
            return datetime.now()

    def chunk_text(self, text: str, max_length: int = Config.CHUNK_SIZE) -> List[str]:
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += ("." if current_chunk else "") + sentence

        if current_chunk:
            chunks.append(current_chunk)

        return [c.strip() for c in chunks if c.strip()]

    def _hash_content(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

    async def ingest_articles(self, articles: List[dict]):
        vectors_to_upsert = []

        for article in articles:
            content_hash = self._hash_content(article["content"])

            if content_hash in self.processed_hashes:
                logger.info("Skipping duplicate: %s", article["title"][:50])
                continue

            chunks = self.chunk_text(article["content"])

            for chunk_idx, chunk in enumerate(chunks):
                embedding = await asyncio.to_thread(self._embed_sync, chunk)
                chunk_id = f"{content_hash}_{chunk_idx}"
                metadata = {
                    "source_url": article["source_url"],
                    "source_name": article["source_name"],
                    "title": article["title"],
                    "published_date": article["published_date"].isoformat(),
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "chunk_text": chunk,
                }

                vectors_to_upsert.append((chunk_id, embedding, metadata))

            self.processed_hashes.add(content_hash)
            logger.info("Processed: %s (%d chunks)", article["title"][:50], len(chunks))

        if vectors_to_upsert:
            try:
                self.index.upsert(vectors=vectors_to_upsert)
                logger.info("Upserted %d vectors to Pinecone", len(vectors_to_upsert))
            except Exception as exc:
                logger.error("Failed to upsert to Pinecone: %s", exc)

    async def run_ingest(self):
        try:
            logger.info("Starting ingestion cycle")
            articles = await self.fetch_rss_feeds()
            
            # Deeper ingestion: Scrape full content for each article
            processed_articles = []
            logger.info("Deep scraping %d articles...", len(articles))
            
            for article in articles:
                full_content = await self.fetch_full_content(article["source_url"])
                if full_content:
                    article["content"] = full_content
                    logger.debug("Successfully scraped full content for: %s", article["title"])
                processed_articles.append(article)
                
            await self.ingest_articles(processed_articles)
            logger.info("Ingestion complete (Deep Scraping applied)")
        except Exception as exc:
            logger.error("Ingestion failed: %s", exc)


def start_scheduler(loop: asyncio.AbstractEventLoop) -> AsyncIOScheduler:
    ingester = NewsIngester()
    scheduler = AsyncIOScheduler(event_loop=loop)
    
    # Run every morning at 8:00 AM
    scheduler.add_job(
        ingester.run_ingest,
        "cron",
        hour=8,
        minute=0,
        id="news_ingestion_morning",
        replace_existing=True,
    )
    
    # Also run once immediately on startup
    scheduler.add_job(
        ingester.run_ingest,
        "date",
        run_date=datetime.now(),
        id="news_ingestion_startup",
    )
    
    scheduler.start()
    logger.info("Scheduler started (Daily at 8:00 AM + Startup run)")
    return scheduler
