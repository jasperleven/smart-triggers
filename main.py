import os
import io
import httpx
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ======================
# CONFIG
# ======================

GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_URL = "https://api.x.ai/v1/chat/completions"

if not GROK_API_KEY:
    raise RuntimeError("GROK_API_KEY is not set")

# ======================
# APP
# ======================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # обязательно для Tilda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# MODELS
# ======================

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    response: str

# ======================
# CHAT ENDPOINT (TILDA)
# ======================

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
payload = {
    "model": "grok-2",
    "messages": [
        {
            "role": "system",
            "content": "Ты ассистент Smart Triggers. Отвечай кратко и по делу."
        },
        {
            "role": "user",
            "content": req.text
        }
    ],
    "temperature": 0.3
}

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(GROK_URL, json=payload, headers=headers)

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
    answer = data["choices"][0]["message"]["content"]

    return ChatResponse(response=answer)

# ======================
# UPLOAD EXCEL
# ======================

@app.post("/upload")
async def upload_excel(file: UploadFile = File(...)):
    try:
        df = pd.read_excel(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Excel file")

    return {
        "rows": len(df),
        "columns": list(df.columns)
    }

# ======================
# DOWNLOAD EXCEL
# ======================

@app.get("/download")
async def download_excel():
    df = pd.DataFrame([
        {"text": "Сколько стоит?", "trigger": "price"},
        {"text": "Как заказать?", "trigger": "order"}
    ])

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=triggers.xlsx"}
    )