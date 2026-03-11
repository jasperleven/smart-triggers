import openai
import json
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

ALLOWED_TRIGGERS = [
    "complaint", "warning", "negative",
    "question", "suggestion", "praise",
    "info", "spam", "neutral"
]

TRIGGER_TO_TONE = {
    "complaint": "negative",
    "warning": "negative",
    "negative": "negative",
    "praise": "positive",
    "suggestion": "neutral",
    "question": "neutral",
    "info": "neutral",
    "spam": "neutral",
    "neutral": "neutral"
}


SYSTEM_PROMPT = """
Ты — аналитическая система классификации пользовательских комментариев.

Задачи:
1. Определи ОСНОВНОЙ смысловой триггер сообщения.
2. Учитывай контекст, иронию, сарказм, агрессию.
3. Риторические вопросы с негативом — НЕ question.
4. Используй ТОЛЬКО допустимые триггеры.

Допустимые триггеры:
complaint, warning, negative, praise, suggestion,
question, info, spam, neutral

Верни строго JSON:
{
  "trigger": "...",
  "confidence": число от 60 до 100
}
"""


def analyze_comment(text: str) -> dict:
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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