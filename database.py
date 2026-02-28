import sqlite3
import logging
from datetime import datetime
from typing import List
from models import ChatMessage

logger = logging.getLogger(__name__)

DB_PATH = "chat_history.db"

def init_db():
    """Initialize the SQLite database for chat history."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                role TEXT,
                content TEXT,
                timestamp DATETIME
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                user_id INTEGER,
                topic TEXT,
                PRIMARY KEY (user_id, topic)
            )
        """)
        conn.commit()
        logger.info("Database initialized successfully at %s", DB_PATH)

def save_message(user_id: int, role: str, content: str):
    """Save a single message to the chat history."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history (user_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
            """, (user_id, role, content, datetime.now().isoformat()))
            conn.commit()
    except Exception as e:
        logger.error("Failed to save message to DB: %s", e)

def get_chat_history(user_id: int, limit: int = 5) -> List[str]:
    """Get the last N messages for a specific user as a list of strings for RAG context."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content FROM chat_history 
                WHERE user_id = ? 
                ORDER BY timestamp DESC LIMIT ?
            """, (user_id, limit))
            rows = cursor.fetchall()
            # Return in chronological order
            return [f"{'Q' if r[0] == 'user' else 'A'}: {r[1]}" for r in reversed(rows)]
    except Exception as e:
        logger.error("Failed to fetch chat history for user %s: %s", user_id, e)
        return []

def clear_chat_history(user_id: int):
    """Clear all chat history for a specific user."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
            conn.commit()
    except Exception as e:
        logger.error("Failed to clear chat history for user %s: %s", user_id, e)

def save_subscription(user_id: int, topic: str):
    """Save a user topic subscription."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO subscriptions (user_id, topic) VALUES (?, ?)", (user_id, topic))
            conn.commit()
    except Exception as e:
        logger.error("Failed to save subscription for user %s: %s", user_id, e)

def get_subscriptions() -> List[tuple]:
    """Get all user subscriptions."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, topic FROM subscriptions")
            return cursor.fetchall()
    except Exception as e:
        logger.error("Failed to fetch all subscriptions: %s", e)
        return []

def remove_subscription(user_id: int, topic: str):
    """Remove a specific topic subscription for a user."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM subscriptions WHERE user_id = ? AND topic = ?", (user_id, topic))
            conn.commit()
    except Exception as e:
        logger.error("Failed to remove subscription for user %s: %s", user_id, e)
