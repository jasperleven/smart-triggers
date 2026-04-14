import os
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS (обязательно для Tilda)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== CONFIG ======
GROK_API_KEY = os.getenv("GROK_API_KEY")

GROK_URL = "https://api.x.ai/v1/chat/completions"

# ВАЖНО: актуальная модель (не grok-2!)
MODEL = "grok-beta"


# ====== INPUT ======
class ChatRequest(BaseModel):
    text: str


# ====== ROUTE ======
@app.post("/chat")
async def chat(req: ChatRequest):

    user_text = req.text

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
                        {"role": "system", "content": "Ты полезный ассистент для бизнеса."},
                        {"role": "user", "content": user_text}
                    ],
                    "temperature": 0.7,
                },
            )

        data = response.json()

        # защита от падений Grok
        try:
            answer = data["choices"][0]["message"]["content"]
        except:
            answer = "Ошибка ответа модели"

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


@app.get("/")
def root():
    return {"status": "ok"}