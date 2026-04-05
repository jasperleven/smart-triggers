import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# ====== CONFIG ======
XAI_API_KEY = os.getenv("XAI_API_KEY")

if not XAI_API_KEY:
    raise RuntimeError("XAI_API_KEY not set")

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1"
)

app = FastAPI()


class ChatRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="grok-2",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты анализируешь текст пользователя и возвращаешь "
                        "СТРОГО JSON без пояснений:\n"
                        "{"
                        "  \"trigger\": string,"
                        "  \"tone\": string,"
                        "  \"confidence\": number от 0 до 1"
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

        content = response.choices[0].message.content

        return content

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))