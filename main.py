import os
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =====================
# CONFIG
# =====================
GROK_API_KEY = os.getenv("GROK_API_KEY")  # ← ИМЕННО ТАК
GROK_MODEL = "llama3-70b-8192"

if not GROK_API_KEY:
    raise RuntimeError("GROK_API_KEY is not set")

# =====================
# APP
# =====================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # обязательно для Tilda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# SCHEMAS
# =====================
class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    response: str

# =====================
# GROK CALL
# =====================
async def call_grok(text: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Ты ассистент Smart Triggers. Отвечай кратко и по делу."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "temperature": 0.3,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

# =====================
# ENDPOINT
# =====================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    answer = await call_grok(req.text)
    return {"response": answer}