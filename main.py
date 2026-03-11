from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
import json

# =====================
# CONFIG
# =====================
openai.api_key = os.getenv("OPENAI_API_KEY")

ALLOWED_TRIGGERS = [
    "complaint", "criticism", "negative",
    "question", "suggestion", "praise",
    "info", "spam", "neutral"
]

TRIGGER_TO_TONE = {
    "complaint": "negative",
    "criticism": "negative",
    "negative": "negative",
    "spam": "neutral",
    "question": "neutral",
    "info": "neutral",
    "suggestion": "neutral",
    "praise": "positive",
    "neutral": "neutral"
}

# =====================
# APP
# =====================
app = FastAPI(title="Smart Triggers API")

class CommentRequest(BaseModel):
    comment: str

class CommentResponse(BaseModel):
    trigger: str
    tone: str
    tone_percent: float
    avg_confidence: float

# =====================
# CORE LOGIC
# =====================
def classify_comment(text: str) -> dict:
    system_prompt = """
Ты — аналитическая система классификации пользовательских комментариев.

Задачи:
1. Определи ОСНОВНОЙ смысловой триггер.
2. Учитывай сарказм, иронию, агрессию.
3. Риторические вопросы с негативом — НЕ question.
4. Используй ТОЛЬКО допустимые триггеры.

Допустимые триггеры:
complaint, criticism, negative, question, suggestion, praise, info, spam, neutral

Верни строго JSON:
{
  "trigger": "...",
  "confidence": число от 60 до 100
}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    data = json.loads(response.choices[0].message.content)

    trigger = data.get("trigger", "neutral")
    confidence = float(data.get("confidence", 80))

    if trigger not in ALLOWED_TRIGGERS:
        trigger = "neutral"

    tone = TRIGGER_TO_TONE[trigger]

    return {
        "trigger": trigger,
        "tone": tone,
        "tone_percent": round(confidence, 2),
        "avg_confidence": round(confidence, 2)
    }

# =====================
# ENDPOINT
# =====================
@app.post("/analyze", response_model=CommentResponse)
async def analyze(request: CommentRequest):
    try:
        return classify_comment(request.comment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def healthcheck():
    return {"status": "ok"}