from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import json

# =====================
# Настройка API Key
# =====================
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Smart Triggers API")

# Разрешаем CORS для всех источников
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommentRequest(BaseModel):
    comment: str

class CommentResponse(BaseModel):
    trigger: str
    tone: str
    tone_percent: float
    avg_confidence: float

@app.post("/analyze", response_model=CommentResponse)
async def analyze_comment(request: CommentRequest):
    prompt = f"""
Определи для комментария следующие данные:
1. Триггер (коротко, одно слово)
2. Тон (positive / negative / neutral)
3. Вероятность тона (0-100)
4. Уровень уверенности в триггере (0-100)

Комментарий: "{request.comment}"

Ответ строго в JSON:
{{"trigger": "", "tone": "", "tone_percent": 0, "avg_confidence": 0}}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result_text = response.choices[0].message.content
        data = json.loads(result_text)
    except Exception:
        data = {"trigger":"unknown","tone":"unknown","tone_percent":0.0,"avg_confidence":0.0}

    return CommentResponse(**data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)