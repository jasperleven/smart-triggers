import os
import io
import pandas as pd
import httpx

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ======================
# CONFIG
# ======================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ======================
# APP
# ======================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для Tilda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# SCHEMAS
# ======================

class ChatRequest(BaseModel):
    text: str
    trigger: str | None = None
    tone: str | None = None
    confidence: str | None = None

# ======================
# HELPERS
# ======================

async def call_groq(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a marketing copywriter and analyst."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(GROQ_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

# ======================
# ENDPOINTS
# ======================

@app.post("/chat")
async def chat(req: ChatRequest):
    prompt = f"""
Текст:
{req.text}

Триггер: {req.trigger or "не указан"}
Тон: {req.tone or "не указан"}
Уровень уверенности: {req.confidence or "не указан"}

Сделай улучшенную версию текста.
"""

    result = await call_groq(prompt)

    return JSONResponse({"result": result})


@app.post("/upload")
async def upload_excel(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_excel(io.BytesIO(content))

    return JSONResponse({
        "rows": len(df),
        "columns": list(df.columns)
    })


@app.post("/process-excel")
async def process_excel(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_excel(io.BytesIO(content))

    # 👇 ПРИМЕР: обработка каждой строки через ИИ
    results = []

    for _, row in df.iterrows():
        text = str(row[0])

        improved = await call_groq(
            f"Улучши маркетинговый текст:\n{text}"
        )

        results.append(improved)

    df["AI_RESULT"] = results

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)

    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": "attachment; filename=result.xlsx"
        }
    )