import os
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# ======================
# CORS (обязательно для Tilda)
# ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# ENV
# ======================
GROK_API_KEY = os.getenv("GROK_API_KEY")

GROK_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-beta"

# ======================
# Request schema
# ======================
class ChatRequest(BaseModel):
    text: str


# ======================
# CHAT ENDPOINT
# ======================
@app.post("/chat")
async def chat(req: ChatRequest):

    if not GROK_API_KEY:
        return {
            "text": "❌ Нет GROK_API_KEY на сервере",
            "trigger": "error",
            "tone": "neutral",
            "confidence": 0
        }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                GROK_URL,
                headers={
                    "Authorization": f"Bearer {GROK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": "Ты полезный бизнес-ассистент."},
                        {"role": "user", "content": req.text}
                    ],
                    "temperature": 0.7,
                },
            )

        data = response.json()

        # ======================
        # SAFE PARSING (без падений)
        # ======================
        if response.status_code != 200:
            return {
                "text": f"Grok error: {data}",
                "trigger": "error",
                "tone": "neutral",
                "confidence": 0
            }

        answer = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )

        if not answer:
            answer = str(data)

        return {
            "text": answer,
            "trigger": "ai_response",
            "tone": "neutral",
            "confidence": 1
        }

    except Exception as e:
        return {
            "text": "Сервис временно недоступен",
            "trigger": "error",
            "tone": "neutral",
            "confidence": 0,
            "debug": str(e)
        }


# ======================
# HEALTHCHECK
# ======================
@app.get("/")
def root():
    return {"status": "ok"}