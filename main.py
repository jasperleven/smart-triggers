import os
import pandas as pd
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import httpx

# =========================
# CONFIG
# =========================

GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_URL = "https://api.x.ai/v1/chat/completions"

if not GROK_API_KEY:
    raise RuntimeError("GROK_API_KEY is not set")

# =========================
# APP
# =========================

app = FastAPI()

# CORS — обязательно для Tilda
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODELS
# =========================

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    text: str
    trigger: str
    tone: str
    confidence: float

# =========================
# HELPERS
# =========================

def detect_trigger(text: str) -> str:
    t = text.lower()
    if "?" in t:
        return "question"
    if any(x in t for x in ["цена", "стоит", "сколько"]):
        return "price"
    if any(x in t for x in ["ошибка", "не работает", "сбой"]):
        return "problem"
    if any(x in t for x in ["спасибо", "класс", "отлично"]):
        return "praise"
    return "neutral"

async def ask_grok(text: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "grok-2",
        "messages": [
            {
                "role": "system",
                "content": "Ты ассистент Smart Triggers. Отвечай кратко и по делу."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "temperature": 0.3
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(GROK_URL, headers=headers, json=payload)

    if r.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Grok request failed",
                "status": r.status_code,
                "details": r.text
            }
        )

    data = r.json()
    return data["choices"][0]["message"]["content"]

# =========================
# ROUTES
# =========================

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    answer = await ask_grok(req.text)

    trigger = detect_trigger(req.text)

    return {
        "text": answer,
        "trigger": trigger,
        "tone": "neutral",
        "confidence": 0.85
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".csv", ".xlsx")):
        raise HTTPException(400, "Only CSV or XLSX")

    if file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    else:
        df = pd.read_excel(file.file)

    if "text" not in df.columns:
        raise HTTPException(400, "Column 'text' is required")

    results = []

    for text in df["text"].astype(str).tolist():
        answer = await ask_grok(text)
        results.append({
            "text": text,
            "answer": answer,
            "trigger": detect_trigger(text),
            "tone": "neutral",
            "confidence": 0.85
        })

    out_df = pd.DataFrame(results)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False, sheet_name="results")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": "attachment; filename=smart_triggers_result.xlsx"
        }
    )

@app.get("/")
def healthcheck():
    return {"status": "ok"}