import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from telegram import Bot, Update
from telegram.error import TelegramError
from dotenv import load_dotenv

load_dotenv()

from config import Config
from rag import RAGPipeline
from ingest import start_scheduler
from database import init_db, save_subscription, get_subscriptions, remove_subscription

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

Config.validate()

request_queue = asyncio.Queue(maxsize=Config.QUEUE_MAX_SIZE)
user_last_request = {}


async def send_proactive_alerts():
    """Check all user subscriptions and send matching AI news alerts."""
    logger.info("Checking for proactive news alerts...")
    subs = get_subscriptions()
    if not subs:
        logger.info("No active subscriptions found.")
        return

    for user_id, topic in subs:
        try:
            logger.info("Processing alert for user %s on topic: %s", user_id, topic)
            # Use RAG to find news for this specific topic
            response = await rag_pipeline.answer_question(user_id, f"What is the latest news regarding {topic}?")
            
            if response.confidence > 0.4:  # Threshold for alerts
                text = (
                    f"🔔 **AI News Alert: {topic}**\n\n"
                    f"{response.answer}\n\n"
                    "I'll keep watching for more updates!"
                )
                await bot.send_message(chat_id=user_id, text=text, parse_mode="Markdown")
                logger.info("Alert sent to user %s", user_id)
        except Exception as e:
            logger.error("Failed to send alert to user %s: %s", user_id, e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting bot")
    
    # Initialize Persistent Chat History DB
    init_db()
    
    if Config.WEBHOOK_URL:
        webhook_url = f"{Config.WEBHOOK_URL.rstrip('/')}/webhook"
        logger.info("Setting webhook to: %s", webhook_url)
        try:
            await bot.set_webhook(url=webhook_url)
            logger.info("Webhook set successfully")
        except TelegramError as e:
            logger.error("Failed to set webhook: %s", e)
    else:
        logger.warning("WEBHOOK_URL is not set. Bot will NOT receive messages from Telegram.")
        
    start_scheduler(asyncio.get_event_loop(), on_complete_callback=send_proactive_alerts)
    asyncio.create_task(process_request_queue())
    logger.info("Bot started successfully")
    yield


app = FastAPI(lifespan=lifespan)
rag_pipeline = RAGPipeline()
bot = Bot(token=Config.TELEGRAM_TOKEN)


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
        message_lower = message_text.lower()

        # New Agentic Routing Step
        intent = await rag_pipeline.classify_intent(message_text)
        logger.info("[User %s] Intent detected: %s", user_id, intent)

        # Handle greetings and start command
        greetings = ["hi", "hello", "hey", "/start"]
        if intent == "GREETING" or message_lower in greetings:
            await bot.send_message(
                chat_id=user_id,
                text=(
                    "Hello! I am your AI News Bot. 🤖\n\n"
                    "I search through the latest articles from Bloomberg, Reuters, TechCrunch, and more to keep you updated.\n\n"
                    "What would you like to know about today?\n\n"
                    "Try one of these categories:\n"
                    "🔹 **LLMs & GenAI**\n"
                    "🔹 **AI Regulations & Ethics**\n"
                    "🔹 **Market Trends**\n"
                    "🔹 **Hardware & Chips (NVIDIA, etc.)**\n\n"
                    "Just ask a question and I'll find the answer for you!"
                ),
                parse_mode="Markdown"
            )
            return

        if message_text.startswith("/"):
            if message_text.startswith("/subscribe"):
                topic = message_text.replace("/subscribe", "").strip()
                if not topic:
                    await bot.send_message(chat_id=user_id, text="Use `/subscribe <topic>` to get daily updates.", parse_mode="Markdown")
                    return
                save_subscription(user_id, topic)
                await bot.send_message(chat_id=user_id, text=f"✅ Subscribed to **{topic}**!", parse_mode="Markdown")
                return
            
            if message_text.startswith("/unsubscribe"):
                topic = message_text.replace("/unsubscribe", "").strip()
                if not topic:
                    await bot.send_message(chat_id=user_id, text="Use `/unsubscribe <topic>` to stop alerts.")
                    return
                remove_subscription(user_id, topic)
                await bot.send_message(chat_id=user_id, text=f"❌ Unsubscribed from **{topic}**.")
                return
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
