from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

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
    prompt = f"""
Определи для текста:
- trigger (complaint, warning, negative, praise, suggestion, question, info, spam, neutral)
- tone (positive, neutral, negative)
- confidence (0–1)

Ответь строго JSON.

Текст:
{req.text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        data = eval(content)
    except Exception:
        data = {
            "trigger": "unknown",
            "tone": "unknown",
            "confidence": 0.0
        }

    return data