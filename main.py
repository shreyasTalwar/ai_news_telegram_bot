import asyncio
import logging
import time
from datetime import datetime

from fastapi import FastAPI
from telegram import Bot, Update
from telegram.error import TelegramError
from dotenv import load_dotenv

load_dotenv()

from config import Config
from rag import RAGPipeline
from ingest import start_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

Config.validate()
app = FastAPI()
rag_pipeline = RAGPipeline()
bot = Bot(token=Config.TELEGRAM_TOKEN)

request_queue = asyncio.Queue(maxsize=Config.QUEUE_MAX_SIZE)
user_last_request = {}


@app.on_event("startup")
async def startup_event():
    logger.info("Starting bot")

    start_scheduler(asyncio.get_event_loop())
    asyncio.create_task(process_request_queue())

    logger.info("Bot started successfully")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "queue_size": request_queue.qsize(),
    }


async def process_request_queue():
    while True:
        try:
            user_id, update = await request_queue.get()

            now = time.time()
            if user_id in user_last_request:
                time_since_last = now - user_last_request[user_id]
                min_interval = 60 / max(Config.MAX_REQUESTS_PER_MINUTE, 1)
                if time_since_last < min_interval:
                    remaining = min_interval - time_since_last
                    await bot.send_message(
                        chat_id=user_id,
                        text=f"Please wait {remaining:.0f}s before asking again",
                    )
                    continue

            user_last_request[user_id] = now
            await handle_telegram_update(user_id, update)

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error("Queue processing error: %s", exc)


@app.post("/webhook")
async def telegram_webhook(request_data: dict):
    try:
        update = Update.de_json(request_data, bot)

        if not update.message or not update.message.text:
            return {"ok": True}

        user_id = update.message.from_user.id

        try:
            request_queue.put_nowait((user_id, update))
        except asyncio.QueueFull:
            await bot.send_message(
                chat_id=user_id,
                text="Bot is busy. Please try again in a moment.",
            )

        return {"ok": True}

    except Exception as exc:
        logger.error("Webhook error: %s", exc)
        return {"ok": False}


async def handle_telegram_update(user_id: int, update: Update):
    try:
        message_text = update.message.text.strip()

        if message_text.startswith("/"):
            if message_text == "/start":
                await bot.send_message(
                    chat_id=user_id,
                    text=(
                        "Welcome to AI News Bot.\n\n"
                        "Ask me anything about AI news and I'll search recent articles to answer.\n\n"
                        "Examples:\n"
                        "- What's new in LLMs?\n"
                        "- Latest AI regulations\n"
                        "- Breakthroughs in computer vision"
                    ),
                )
            return

        await bot.send_chat_action(chat_id=user_id, action="typing")

        logger.info("[User %s] Question: %s", user_id, message_text[:100])

        response = await rag_pipeline.answer_question(user_id, message_text)

        answer_text = f"{response.answer}\n\n"

        if response.sources:
            answer_text += "Sources:\n"
            for i, source in enumerate(response.sources[:3], 1):
                answer_text += f"{i}. {source.source_name} - {source.source_url}\n"
        else:
            answer_text += "No sources found"

        answer_text += f"\nRelevance: {response.confidence*100:.0f}%"

        if len(answer_text) > 4000:
            chunks = [answer_text[i : i + 4000] for i in range(0, len(answer_text), 4000)]
            for chunk in chunks:
                await bot.send_message(
                    chat_id=user_id,
                    text=chunk,
                    parse_mode=None,
                )
        else:
            await bot.send_message(
                chat_id=user_id,
                text=answer_text,
                parse_mode=None,
            )

        logger.info("[User %s] Response sent (%.2f confidence)", user_id, response.confidence)

    except TelegramError as exc:
        logger.error("Telegram error for user %s: %s", user_id, exc)
        try:
            await bot.send_message(chat_id=user_id, text="Error sending message. Please try again.")
        except Exception:
            pass

    except Exception as exc:
        logger.error("Error handling update for user %s: %s", user_id, exc)
        try:
            await bot.send_message(chat_id=user_id, text="Something went wrong. Please try again later.")
        except Exception:
            pass


@app.get("/")
async def root():
    return {
        "name": "AI News RAG Bot",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "webhook": "/webhook",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
