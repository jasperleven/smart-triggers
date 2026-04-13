from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

# CORS ДЛЯ TILDA
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_URL = "https://api.x.ai/v1/chat/completions"

class ChatRequest(BaseModel):
    text: str

@app.post("/chat")
async def chat(req: ChatRequest):
    if not GROK_API_KEY:
        return {"error": "GROK_API_KEY not set"}

    payload = {
        "model": "grok-2-latest",
        "messages": [
            {
                "role": "system",
                "content": "Ты ассистент Smart Triggers. Отвечай кратко и по делу."
            },
            {
                "role": "user",
                "content": req.text
            }
        ],
        "temperature": 0.3
    }

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(GROK_URL, json=payload, headers=headers)

    if r.status_code != 200:
        return {
            "error": "Grok request failed",
            "status": r.status_code,
            "details": r.text
        }

    data = r.json()
    answer = data["choices"][0]["message"]["content"]

    return {
        "response": answer
    }