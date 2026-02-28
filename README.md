# 🤖 AI News RAG Telegram Bot

An advanced, autonomous Telegram bot that provides real-time AI news summaries using Retrieval-Augmented Generation (RAG). It monitors elite news sources, scrapes full article content, and maintains persistent conversational memory.

---

## 🌟 Key Features

- **Deep Ingestion & Scraping:** Automatically monitors RSS feeds (Bloomberg, Reuters, TechCrunch, etc.) and YouTube. It uses the **Jina Reader API** to scrape full article Markdown for deep context.
- **Agentic RAG Pipeline:** 
  - **Intent Routing:** Automatically distinguishes between greetings, general chat, and news queries.
  - **LLM-Based Re-ranking:** Fetches the top 15 results from Pinecone and use Gemini to re-rank the best 5 for superior summary quality.
- **Persistent Memory:** Uses a local **SQLite** database to store chat history, ensuring the bot remembers you even after restarts.
- **Morning Briefings:** Proactive subscription system (`/subscribe`) that sends personalized news alerts every morning at 8:00 AM.
- **Quota Optimized:** Implements smart deduplication and article limits to stay within the Google Gemini Free Tier limits.

---

## 🛠️ Tech Stack

- **Framework:** FastAPI (Python 3.11+)
- **LLM:** Google Gemini 1.5 Flash (Generation) & Gemini Embedding 001 (Embeddings)
- **Vector Database:** Pinecone
- **Memory:** SQLite
- **Scraping:** Jina Reader API
- **Deployment:** Railway (with Webhook support)

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.11+
- [ngrok](https://ngrok.com/) (for local testing)
- API Keys: Telegram (BotFather), Google AI Studio, Pinecone.

### 2. Local Installation
```bash
# Clone the repository
git clone https://github.com/shreyasTalwar/ai_news_telegram_bot.git
cd ai_news_telegram_bot

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
TELEGRAM_TOKEN=your_telegram_token
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=ai-news
WEBHOOK_URL=https://your-app.railway.app  # Or your ngrok URL for local testing
```

### 4. Running the Bot
```bash
# Start uvicorn server
uvicorn main:app --port 8000
```

---

## 🤖 Bot Commands

- `/start` or `hi` - Initialize the bot and see categories.
- `/subscribe <topic>` - Get daily morning alerts for a specific AI topic.
- `/unsubscribe <topic>` - Stop receiving alerts for a topic.
- `[Any Question]` - Just ask a question about AI news!

---

## 🏛️ Architecture

1. **`ingest.py`**: Handles periodic RSS/YouTube fetching, deep scraping, and Pinecone upserts.
2. **`rag.py`**: Manages the query pipeline, embedding, re-ranking, and Gemini generation.
3. **`main.py`**: FastAPI entry point, Telegram webhook handler, and intent routing.
4. **`database.py`**: SQLite logic for persistent chat memory and subscriptions.

---

**Developed by Shreyas Talwar**
