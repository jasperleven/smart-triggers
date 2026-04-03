from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

# инициализация клиента OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ChatRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    trigger: str
    tone: str
    confidence: float


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты анализируешь текст и возвращаешь JSON строго в формате:\n"
                        "{trigger: string, tone: string, confidence: number от 0 до 1}"
                    ),
                },
                {"role": "user", "content": req.text},
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content

        # fallback, если модель ответила не JSON
        return {
            "trigger": "unknown",
            "tone": "neutral",
            "confidence": 0.0,
        }

    except Exception as e:
        print("ERROR:", e)
        return {
            "trigger": "error",
            "tone": "error",
            "confidence": 0.0,
        }