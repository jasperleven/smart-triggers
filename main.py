from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

# =========================
# ИНИЦИАЛИЗАЦИЯ
# =========================

app = FastAPI()

# CORS — ОБЯЗАТЕЛЬНО для Tilda
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # можно потом заменить на tilda-домен
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()  # ключ берётся из переменной окружения OPENAI_API_KEY


# =========================
# SCHEMAS
# =========================

class ChatRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    trigger: str
    tone: str
    confidence: float


# =========================
# ROUTES
# =========================

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    text = req.text

    prompt = f"""
Определи триггер, тональность и уверенность текста.

Текст: "{text}"

Ответь строго в формате:
trigger: ...
tone: ...
confidence: число от 0 до 1
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ты аналитик маркетинговых триггеров."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content

    # Простейший парсинг
    lines = content.split("\n")
    data = {}

    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()

    return {
        "trigger": data.get("trigger", "unknown"),
        "tone": data.get("tone", "unknown"),
        "confidence": float(data.get("confidence", 0)),
    }