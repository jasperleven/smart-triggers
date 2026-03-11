import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Инициализация клиента OpenAI с ключом из переменной окружения
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Модель запроса
class MessageRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "Smart Triggers API работает"}

@app.post("/chat")
async def chat(request: MessageRequest):
    # Отправка запроса в OpenAI
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": request.message}]
    )
    # Возвращаем текст ответа
    return {"response": response.choices[0].message.content}