import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# =========================
# CONFIG
# =========================

XAI_API_KEY = os.getenv("XAI_API_KEY")

if not XAI_API_KEY:
    raise RuntimeError("XAI_API_KEY is not set")

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1"
)

MODEL_NAME = "grok-1"

SYSTEM_PROMPT = """
Ты анализируешь входящий текст пользователя и возвращаешь JSON
строго в формате:

{
  "trigger": "<короткое описание намерения>",
  "tone": "<positive|neutral|negative>",
  "confidence": <число от 0 до 1>
}

Без пояснений. Без текста вне JSON.
"""

# =========================
# APP
# =========================

app = FastAPI(
    title="Smart Triggers API",
    version="0.1.0"
)

# =========================
# SCHEMAS
# =========================

class ChatRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    trigger: str
    tone: str
    confidence: float


# =========================
# ROUTES
# =========================

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.text}
            ],
            temperature=0.2
        )

        content = response.choices[0].message.content

        # ожидаем, что модель вернула чистый JSON
        return ChatResponse.model_validate_json(content)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )