from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    trigger: str
    tone: str
    confidence: int

@app.get("/")
def health():
    return "OK"

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Определи тип триггера, тон сообщения и уверенность от 0 до 100. Ответ строго в JSON."
                },
                {
                    "role": "user",
                    "content": req.text
                }
            ]
        )

        content = response.choices[0].message.content

        # временно — просто вернём текст, чтобы убедиться что OpenAI отвечает
        return {
            "trigger": "ok",
            "tone": content[:30],
            "confidence": 100
        }

    except Exception as e:
        print("OPENAI ERROR:", str(e))
        raise e