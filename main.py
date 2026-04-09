from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re

app = FastAPI()

# CORS — обязательно для Tilda
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== MODELS ======

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    trigger: str
    tone: str
    confidence: float
    response: str


# ====== SIMPLE SMART TRIGGERS LOGIC ======

def detect_trigger(text: str) -> tuple[str, float]:
    text = text.lower()

    triggers = {
        "price_request": [
            "цена", "стоимость", "сколько стоит", "прайс"
        ],
        "interest": [
            "расскажи", "что это", "как работает", "что умеет"
        ],
        "support": [
            "не работает", "ошибка", "проблема", "сломалось"
        ],
        "greeting": [
            "привет", "здравствуйте", "алло", "ты тут"
        ]
    }

    for trigger, keywords in triggers.items():
        for kw in keywords:
            if kw in text:
                return trigger, 0.8

    return "unknown", 0.4


def detect_tone(text: str) -> str:
    if re.search(r"[!?]{2,}", text):
        return "emotional"
    if any(word in text.lower() for word in ["пожалуйста", "спасибо"]):
        return "polite"
    return "neutral"


def generate_response(trigger: str) -> str:
    responses = {
        "price_request": "Могу подсказать по стоимости. Уточни, что именно тебя интересует.",
        "interest": "С удовольствием расскажу. Что именно хочешь узнать?",
        "support": "Давай разберёмся. Опиши проблему подробнее.",
        "greeting": "Да, я на связи 🙂",
        "unknown": "Я тебя понял. Можешь уточнить запрос?"
    }

    return responses.get(trigger, responses["unknown"])


# ====== ENDPOINT ======

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    trigger, confidence = detect_trigger(request.text)
    tone = detect_tone(request.text)
    response_text = generate_response(trigger)

    return ChatResponse(
        trigger=trigger,
        tone=tone,
        confidence=confidence,
        response=response_text
    )