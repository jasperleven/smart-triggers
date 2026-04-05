import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GROK_API_KEY"),
    base_url="https://api.x.ai/v1"
)

app = FastAPI()


class ChatRequest(BaseModel):
    text: str


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="grok-2-latest",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты анализируешь текст и возвращаешь JSON строго в формате:\n"
                        "{trigger: string, tone: string, confidence: number от 0 до 1}"
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
        return eval(content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))