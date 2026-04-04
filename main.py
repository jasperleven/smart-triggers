from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

# Инициализация приложения
app = FastAPI()

# CORS — ОБЯЗАТЕЛЬНО для Tilda
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI клиент (ключ ТОЛЬКО из Render env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ====== MODELS ======
class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    trigger: str
    tone: str
    confidence: int

# ====== HEALTH ======
@app.get("/")
def health():
    return {"status": "ok"}

# ====== CHAT ======
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        prompt = f"""
Определи для текста:
1. trigger (complaint, warning, negative, praise, suggestion, question, info, spam, neutral)
2. tone (positive, neutral, negative)
3. confidence (0-100)

Ответь ТОЛЬКО в JSON:
{{
  "trigger": "...",
  "tone": "...",
  "confidence": 0
}}

Текст:
\"\"\"{req.text}\"\"\"
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        # fallback на случай кривого ответа
        import json
        data = json.loads(content)

        return ChatResponse(
            trigger=data.get("trigger", "unknown"),
            tone=data.get("tone", "unknown"),
            confidence=int(data.get("confidence", 0))
        )

    except Exception as e:
        return ChatResponse(
            trigger="error",
            tone="error",
            confidence=0
        )