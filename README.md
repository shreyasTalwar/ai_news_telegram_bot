# AI News RAG Telegram Bot

## Quick Start

```bash
python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
copy .env.example .env
```

Fill in `.env` with your API keys, then:

```bash
python main.py
```

Health check:

```bash
curl http://localhost:8000/health
```

## Endpoints
- `GET /health`
- `POST /webhook`
- `GET /`
