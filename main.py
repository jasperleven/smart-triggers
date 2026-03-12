from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

# ---------- APP ----------
app = FastAPI()

# ---------- CORS (обязательно для Tilda) ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # можно потом сузить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- OpenAI ----------
# КЛЮЧ БЕРЁТСЯ ИЗ ENV: OPENAI_API_KEY
client = OpenAI()

# ---------- MODELS ----------
class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    trigger: str
    tone: str
    confidence: float

# ---------- HEALTH CHECK ----------
@app.get("/")
def health():
    return {"status": "ok"}

# ---------- CHAT ENDPOINT ----------
@app.post("/chat", response_model=ChatResponse)
def chat(data: ChatRequest):
    prompt = f"""
Проанализируй сообщение пользователя.

Сообщение:
"{data.text}"

Верни строго JSON:
{{
  "trigger": "...",
  "tone": "...",
  "confidence": число от 0 до 1
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ты аналитик пользовательских триггеров."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw = response.choices[0].message.content

    try:
        parsed = eval(raw)
    except:
        parsed = {
            "trigger": "unknown",
            "tone": "unknown",
            "confidence": 0.0
        }

    return parsed