import os
import json
import io
import pandas as pd
import re

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from openai import OpenAI  # исправлено

# Создание клиента OpenAI без лишних аргументов
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Пример маршрута для проверки
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Пример загрузки файла
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    # Работа с pandas
    df = pd.read_csv(io.StringIO(content.decode()))
    return JSONResponse(content={"rows": len(df)})

# Здесь остальные маршруты, функции, обработка данных остаются без изменений