from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os

app = FastAPI(title="Smart Triggers API")

# Получаем ключ Grok из переменных окружения Render
GROK_API_KEY = os.getenv("GROK_API_KEY")
if not GROK_API_KEY:
    raise ValueError("GROK_API_KEY не установлен в переменных окружения Render")

# Модель запроса
class ChatRequest(BaseModel):
    text: str

# Модель ответа
class ChatResponse(BaseModel):
    trigger: str
    tone: str
    confidence: float

# Эндпоинт
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": request.text,
            "max_tokens": 50
        }

        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                "https://api.grok.ai/v1/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

        # Простая заглушка для примера обработки результата
        result = data.get("choices", [{}])[0].get("text", "")
        return ChatResponse(trigger="success", tone="neutral", confidence=1.0)

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))