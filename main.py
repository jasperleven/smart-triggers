import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from openai import OpenAIError

# ===== ENV =====
GROK_API_KEY = os.getenv("GROK_API_KEY")
if not GROK_API_KEY:
    raise RuntimeError("GROK_API_KEY is not set")

# ===== CLIENT =====
client = OpenAI(
    api_key=GROK_API_KEY,
    base_url="https://api.x.ai/v1"
)

MODEL_PRIMARY = "grok-1"
MODEL_FALLBACK = "grok-beta"

# ===== APP =====
app = FastAPI(title="Smart Triggers API")

# ===== SCHEMAS =====
class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1)

class ChatResponse(BaseModel):
    trigger: str
    tone: str
    confidence: float

# ===== HEALTH =====
@app.get("/")
def health():
    return {"status": "ok"}

# ===== INTERNAL =====
def call_grok(model: str, text: str) -> dict:
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты классификатор текста.\n"
                    "Верни ТОЛЬКО валидный JSON без комментариев:\n"
                    "{\n"
                    '  "trigger": "string",\n'
                    '  "tone": "neutral | positive | negative",\n'
                    '  "confidence": number от 0 до 1\n'
                    "}"
                )
            },
            {"role": "user", "content": text}
        ],
    )

    content = completion.choices[0].message.content
    return json.loads(content)

# ===== MAIN ENDPOINT =====
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        data = call_grok(MODEL_PRIMARY, req.text)

    except OpenAIError as e:
        # fallback при quota / model error / 429
        try:
            data = call_grok(MODEL_FALLBACK, req.text)
        except Exception as e2:
            raise HTTPException(
                status_code=502,
                detail=f"Grok failed: {str(e2)}"
            )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Invalid JSON from model"
        )

    return ChatResponse(
        trigger=data.get("trigger", "unknown"),
        tone=data.get("tone", "unknown"),
        confidence=float(data.get("confidence", 0))
    )