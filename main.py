from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Инициализация клиента OpenAI
client = OpenAI()  # ключ берется из переменной окружения OPENAI_API_KEY

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # можно заменить на ["https://your-tilde-domain.tilda.ws"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Модель запроса ---
class ChatRequest(BaseModel):
    text: str

# --- Endpoint ---
@app.post("/chat")
async def chat(request: ChatRequest):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": request.text}]
    )
    return {"response": response.choices[0].message.content}