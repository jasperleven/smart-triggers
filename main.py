from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS — обязательно для Tilda
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- SCHEMA -----

class ChatRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    response: str


# ----- ENDPOINT -----

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_text = request.text.strip()

    if not user_text:
        return {"response": "Пустой запрос"}

    # Заглушка вместо Grok (пока)
    return {
        "response": f"Grok ответ на: {user_text}"
    }


# ----- ROOT (необязательно, но полезно) -----

@app.get("/")
async def root():
    return {"status": "ok"}