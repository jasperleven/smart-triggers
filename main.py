# main.py
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from io import BytesIO
import httpx
from fastapi.responses import StreamingResponse, JSONResponse

GROK_API_KEY = os.environ.get("GROK_API_KEY")
GROK_API_URL = "https://api.openai.com/v1/grok-beta"

app = FastAPI(title="Smart Triggers via Grok")

# ===== CORS (для Тильды или фронта) =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # при желании ограничить фронтом
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Models =====
class TextRequest(BaseModel):
    text: str

# ===== Helpers =====
def analyze_text_grok(text: str):
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"input": text}
    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{GROK_API_URL}/analyze", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    # пример структуры, адаптируй под реальный ответ Grok
    result = {
        "trigger": data.get("trigger", "neutral"),
        "tone": data.get("tone", "neutral"),
        "confidence": data.get("confidence", 0),
        "response": data.get("response", "")
    }
    return result

def process_file(file: UploadFile):
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    else:
        df = pd.read_excel(file.file)

    if "text" not in df.columns:
        return None, "В файле должна быть колонка 'text'"

    results = []
    for i, txt in enumerate(df["text"].astype(str), 1):
        analysis = analyze_text_grok(txt)
        row = {
            "id": i,
            "text": txt,
            **analysis
        }
        results.append(row)

    df_result = pd.DataFrame(results)

    # Excel в байты
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_result.to_excel(writer, index=False, sheet_name="results")
    buffer.seek(0)
    return buffer, None

# ===== Endpoints =====
@app.post("/chat")
def chat_endpoint(req: TextRequest):
    result = analyze_text_grok(req.text)
    return JSONResponse(content=result)

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    buffer, err = process_file(file)
    if err:
        return JSONResponse(status_code=422, content={"detail": err})
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={file.filename.split('.')[0]}_results.xlsx"}
    )

# ===== Test endpoint =====
@app.get("/")
def root():
    return {"status": "Smart Triggers via Grok is running"}