# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json

# =====================
# Инициализация
# =====================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Smart Triggers API")

# Разрешаем CORS для всех источников (можно ограничить до Tilda)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # или ["https://project19041806.tilda.ws"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# Модели
# =====================
class CommentRequest(BaseModel):
    comment: str

class CommentResponse(BaseModel):
    trigger: str
    tone: str
    tone_percent: float
    avg_confidence: float

# =====================
# Эндпоинт анализа
# =====================
@app.post("/analyze", response_model=CommentResponse)
async def analyze_comment(request: CommentRequest):
    prompt = f"""
Определи для комментария следующие данные:
1. Триггер (коротко, одно слово, например: "complaint", "question", "praise")
2. Тон (например: "positive", "negative", "neutral")
3. Вероятность тона (0-100)
4. Уровень уверенности в триггере (0-100)

Комментарий: "{request.comment}"

Ответ строго в JSON:
{{
  "trigger": "",
  "tone": "",
  "tone_percent": 0,
  "avg_confidence": 0
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result_text = response.choices[0].message.content
        data = json.loads(result_text)
    except Exception:
        data = {
            "trigger": "unknown",
            "tone": "unknown",
            "tone_percent": 0.0,
            "avg_confidence": 0.0
        }

    return CommentResponse(**data)

# =====================
# Локальный запуск
# =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)