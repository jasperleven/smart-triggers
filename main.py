import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# ====== ENV ======
GROK_API_KEY = os.getenv("GROK_API_KEY")
if not GROK_API_KEY:
    raise RuntimeError("GROK_API_KEY not set")

# ====== CLIENT ======
client = OpenAI(
    api_key=GROK_API_KEY,
    base_url="https://api.x.ai/v1"
)

# ====== APP ======
app = FastAPI(title="Smart Triggers API")

# ====== SCHEMAS ======
class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    trigger: str
    tone: str
    confidence: float

# ====== HEALTH ======
@app.get("/")
def health():
    return {"status": "ok"}

# ====== MAIN ENDPOINT ======
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        completion = client.chat.completions.create(
            model="grok-2",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты классификатор текста. "
                        "Определи:\n"
                        "1) trigger (намерение пользователя)\n"
                        "2) tone (тон: neutral, positive, negative)\n"
                        "3) confidence (уверенность от 0 до 1)\n\n"
                        "Ответ строго в JSON:\n"
                        "{"
                        '"trigger": "...", '
                        '"tone": "...", '
                        '"confidence": 0.0'
                        "}"
                    )
                },
                {
                    "role": "user",
                    "content": req.text
                }
            ],
            temperature=0
        )

        content = completion.choices[0].message.content

        # ожидаем чистый JSON от модели
        data = eval(content) if isinstance(content, str) else content

        return ChatResponse(
            trigger=data.get("trigger", "unknown"),
            tone=data.get("tone", "unknown"),
            confidence=float(data.get("confidence", 0))
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))