import os
import httpx
import pandas as pd
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODELS_URL = "https://api.groq.com/openai/v1/models"

app = FastAPI()

# === CORS для Тильды ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# Models
# =====================
class ChatRequest(BaseModel):
    text: str


# =====================
# Groq helpers
# =====================
async def get_available_model():
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            MODELS_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"}
        )
        r.raise_for_status()
        models = r.json()["data"]

        # Берём первую чат-модель
        for m in models:
            if "llama" in m["id"].lower() or "mixtral" in m["id"].lower():
                return m["id"]

        # fallback
        return models[0]["id"]


async def ask_groq(text: str):
    model = await get_available_model()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Ты классификатор пользовательских сообщений."},
            {"role": "user", "content": text}
        ],
        "temperature": 0
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# =====================
# Trigger logic
# =====================
def detect_trigger(text: str):
    t = text.lower()
    if any(x in t for x in ["цена", "стоимость", "сколько стоит"]):
        return "price_request"
    if any(x in t for x in ["ошибка", "не работает", "сбой"]):
        return "problem"
    if "?" in t:
        return "question"
    if any(x in t for x in ["спасибо", "круто", "отлично"]):
        return "praise"
    return "neutral"


# =====================
# API
# =====================
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        ai_answer = await ask_groq(req.text)

        return {
            "text": req.text,
            "trigger": detect_trigger(req.text),
            "tone": "neutral",
            "confidence": 0.9,
            "response": ai_answer
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    df = pd.read_excel(file.file) if file.filename.endswith("xlsx") else pd.read_csv(file.file)

    if "text" not in df.columns:
        return JSONResponse(status_code=400, content={"error": "Нужна колонка text"})

    results = []
    for text in df["text"].astype(str):
        results.append({
            "text": text,
            "trigger": detect_trigger(text),
            "tone": "neutral",
            "confidence": 0.9
        })

    out_df = pd.DataFrame(results)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False)

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=smart_triggers.xlsx"}
    )