from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    trigger: str
    tone: str
    confidence: int

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        # НОВЫЙ, стабильный Responses API
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": "Ответь строго в JSON: trigger, tone, confidence (0-100)."
                },
                {
                    "role": "user",
                    "content": req.text
                }
            ]
        )

        text = resp.output_text
        # временно просто возвращаем, чтобы проверить, что ответ есть
        return {
            "trigger": "ok",
            "tone": text[:50],
            "confidence": 100
        }

    except Exception as e:
        # ВАЖНО: логируем и НЕ роняем сервис молча
        print("OPENAI ERROR >>>", repr(e))
        raise HTTPException(status_code=500, detail=str(e))