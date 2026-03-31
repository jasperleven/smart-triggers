from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json

client = OpenAI()

app = FastAPI(title="Smart Triggers API")

# ✅ CORS — КРИТИЧНО
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # позже можно сузить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    trigger: str
    tone: str
    confidence: float

@app.get("/")
def health():
    return "Health"

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    prompt = f"""
Определи:
1. trigger (complaint, question, praise, suggestion, info, neutral, spam)
2. tone (positive, neutral, negative)
3. confidence (0-100)

Текст:
"{req.text}"

Ответ строго JSON:
{{
  "trigger": "",
  "tone": "",
  "confidence": 0
}}
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        data = json.loads(content)
    except:
        data = {
            "trigger": "unknown",
            "tone": "neutral",
            "confidence": 0
        }

    return data