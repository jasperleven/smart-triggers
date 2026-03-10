import os
import json
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# =====================
# OpenAI
# =====================
client = OpenAI(
    api_key=os.getenv("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")
)

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set")

# =====================
# FastAPI
# =====================
app = FastAPI(
    title="Smart Triggers API",
    description="AI-based trigger & tone detection",
    version="1.0"
)

# =====================
# Models
# =====================
class AnalyzeRequest(BaseModel):
    texts: List[str]

class AnalyzeItem(BaseModel):
    id: int
    text: str
    trigger: str
    confidence: float
    tone: str
    tone_percent: float
    avg_confidence: float

# =====================
# AI PROMPT
# =====================
SYSTEM_PROMPT = """
Ты — аналитический AI, который классифицирует пользовательские комментарии.

Определи ОДИН основной триггер комментария:

Возможные триггеры:
- spam — реклама, мусор, боты
- complaint — жалоба, возмущение, недовольство
- warning — угроза, призыв к наказанию, агрессия
- negative — негативная оценка без прямой жалобы
- suggestion — предложение, совет
- praise — поддержка, одобрение
- question — вопрос или риторический вопрос
- info — нейтральное сообщение или факт
- neutral — если триггер не выражен

ВАЖНО:
- сарказм, ирония, политическая критика = НЕ neutral
- риторические вопросы = question
- эмоциональная критика власти или институтов = negative или complaint
- neutral использовать ТОЛЬКО если реально нет оценки

Верни JSON строго в формате:
{
  "trigger": "...",
  "confidence": число от 0 до 100
}
"""

# =====================
# Helpers
# =====================
def tone_from_trigger(trigger: str):
    if trigger in ["complaint", "warning", "negative"]:
        return "negative", 100.0
    if trigger == "praise":
        return "positive", 100.0
    return "neutral", 100.0

def classify_text(text: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ]
        )

        content = response.choices[0].message.content.strip()
        data = json.loads(content)

        trigger = data.get("trigger", "neutral")
        confidence = float(data.get("confidence", 50))

        tone, tone_percent = tone_from_trigger(trigger)

        return trigger, confidence, tone, tone_percent

    except Exception as e:
        return "neutral", 50.0, "neutral", 50.0

# =====================
# Endpoint
# =====================
@app.post("/analyze", response_model=List[AnalyzeItem])
def analyze(req: AnalyzeRequest):
    results = []

    for i, text in enumerate(req.texts, start=1):
        trigger, conf, tone, tone_percent = classify_text(text)

        results.append({
            "id": i,
            "text": text,
            "trigger": trigger,
            "confidence": conf,
            "tone": tone,
            "tone_percent": tone_percent,
            "avg_confidence": conf
        })

    return results