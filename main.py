import os
import json
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# ===== ENV =====
GROK_API_KEY = os.getenv("GROK_API_KEY")
if not GROK_API_KEY:
    raise RuntimeError("GROK_API_KEY not set")

# ===== CONFIG =====
MODEL_NAME = "grok-2"
MAX_RETRIES = 3

# ===== CLIENT =====
client = OpenAI(
    api_key=GROK_API_KEY,
    base_url="https://api.x.ai/v1"
)

# ===== APP =====
app = FastAPI(title="Smart Triggers API")

# ===== SCHEMAS =====
class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    trigger: str
    tone: str
    confidence: float

# ===== HEALTH =====
@app.get("/")
def health():
    return {"status": "ok"}

# ===== MAIN ENDPOINT =====
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    prompt = (
        "Ты классификатор текста.\n"
        "Определи:\n"
        "1) trigger (намерение)\n"
        "2) tone (neutral, positive, negative)\n"
        "3) confidence (0–1)\n\n"
        "Ответ строго в JSON:\n"
        '{ "trigger": "...", "tone": "...", "confidence": 0.0 }'
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": req.text}
                ],
                temperature=0
            )

            content = completion.choices[0].message.content.strip()
            data = json.loads(content)

            return ChatResponse(
                trigger=data.get("trigger", "unknown"),
                tone=data.get("tone", "unknown"),
                confidence=float(data.get("confidence", 0))
            )

        except Exception as e:
            # 429 / quota / SSL / model not found
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            raise HTTPException(status_code=500, detail=str(e))

    # fallback — сервис НИКОГДА не падает
    return ChatResponse(
        trigger="unknown",
        tone="neutral",
        confidence=0.0
    )