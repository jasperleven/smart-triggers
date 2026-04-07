import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Заменяй на реальную библиотеку клиента Grok, если она есть
# Например: from grok_sdk import GrokClient
# Здесь я оставлю заглушку для примера
class GrokClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def send_message(self, text: str):
        # Заглушка, на проде заменить реальным вызовом API Grok
        if text.strip() == "":
            raise ValueError("Empty text")
        return f"Grok ответ на: {text}"


# Получаем ключ из переменных окружения
GROK_API_KEY = os.getenv("GROK_API_KEY")
if not GROK_API_KEY:
    raise RuntimeError("GROK_API_KEY is not set")

# Инициализация клиента Grok
client = GrokClient(api_key=GROK_API_KEY)

# FastAPI
app = FastAPI()

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        result = client.send_message(req.text)
        return ChatResponse(response=result)
    except Exception as e:
        # Если что-то пошло не так с Grok
        raise HTTPException(status_code=500, detail=str(e))