from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import os, json, io, pandas as pd, re

# =====================
# Настройка OpenAI
# =====================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =====================
# FastAPI
# =====================
app = FastAPI(title="Smart Triggers API")

ALLOWED_TRIGGERS = [
    "complaint", "warning", "negative", "praise",
    "suggestion", "question", "info", "spam", "neutral"
]

TRIGGER_TO_TONE = {
    "complaint": "negative",
    "warning": "negative",
    "negative": "negative",
    "praise": "positive",
    "suggestion": "neutral",
    "question": "neutral",
    "info": "neutral",
    "spam": "neutral",
    "neutral": "neutral"
}

# =====================
# Анализ комментария через GPT
# =====================
def analyze_text(text: str):
    prompt = f"""
Определи для комментария следующие данные:
1. Триггер (одно слово: complaint, warning, negative, praise, suggestion, question, info, spam, neutral)
2. Тон (positive / neutral / negative)
3. Вероятность тона (0-100)
4. Уровень уверенности в триггере (0-100)

Комментарий: "{text}"

Ответ строго в JSON:
{{"trigger": "", "tone": "", "tone_percent": 0, "avg_confidence": 0}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result_text = response.choices[0].message.content
        match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            data = {}
    except Exception:
        data = {}

    data.setdefault("trigger", "unknown")
    data.setdefault("tone", "unknown")
    data.setdefault("tone_percent", 0.0)
    data.setdefault("avg_confidence", 0.0)
    return data

# =====================
# Главная страница с интерфейсом
# =====================
@app.get("/")
async def root():
    html = """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Smart Triggers</title>
<style>
body { font-family: Arial; padding: 20px; background: #f7f7f7; }
textarea { width: 100%; height: 120px; padding: 10px; font-size: 14px; }
button { background: #0066ff; color: #fff; border: none; padding: 10px 20px; cursor: pointer; font-size: 16px; }
button:hover { background: #0055cc; }
table { width: 100%; border-collapse: collapse; margin-top: 20px; }
th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
th { background: #eee; }
</style>
</head>
<body>
<h2>Smart Triggers — AI анализ комментариев</h2>
<textarea id="commentInput" placeholder="Вставьте комментарий или текст для анализа…"></textarea><br><br>
<input type="file" id="fileInput"><br><br>
<button onclick="analyzeComments()">Начать анализ</button>
<button onclick="downloadExcel()">Скачать Excel</button>

<table id="resultTable" style="display:none">
<thead>
<tr><th>ID</th><th>Text</th><th>Trigger</th><th>Confidence %</th><th>Tone</th></tr>
</thead>
<tbody></tbody>
</table>

<script>
let results = [];

async function analyzeComments() {
    results = [];
    const texts = [];
    const textVal = document.getElementById('commentInput').value.trim();
    if(textVal) texts.push(textVal);

    // Файл
    const fileInput = document.getElementById('fileInput');
    if(fileInput.files.length>0){
        const file = fileInput.files[0];
        const reader = new FileReader();
        reader.onload = async function(e){
            const content = e.target.result;
            const lines = content.split(/\\r?\\n/).filter(l=>l.trim()!=="");
            texts.push(...lines);
            await analyzeAll(texts);
        }
        reader.readAsText(file);
    } else {
        await analyzeAll(texts);
    }
}

async function analyzeAll(texts){
    const tbody = document.querySelector('#resultTable tbody');
    tbody.innerHTML = '';
    for(let i=0;i<texts.length;i++){
        const t = texts[i];
        const resp = await fetch('/analyze', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify({comment:t})
        });
        const data = await resp.json();
        results.push({id:i+1, text:t, ...data});

        const row = document.createElement('tr');
        row.innerHTML = `<td>${i+1}</td>
                         <td>${t}</td>
                         <td>${data.trigger}</td>
                         <td>${data.avg_confidence}</td>
                         <td>${data.tone}</td>`;
        tbody.appendChild(row);
    }
    document.getElementById('resultTable').style.display = 'table';
}

function downloadExcel(){
    if(results.length===0){ alert('Нет данных для экспорта'); return; }
    fetch('/export', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({results:results})
    }).then(res=>res.blob())
      .then(blob=>{
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'SmartTriggers.xlsx';
        document.body.appendChild(a);
        a.click();
        a.remove();
      });
}
</script>
</body>
</html>
"""
    return HTMLResponse(html)

# =====================
# API для анализа одного комментария
# =====================
class CommentRequest(BaseModel):
    comment: str

@app.post("/analyze")
async def analyze_endpoint(request: CommentRequest):
    data = analyze_text(request.comment)
    return data

# =====================
# Экспорт Excel
# =====================
class ExportRequest(BaseModel):
    results: list

@app.post("/export")
async def export_excel(request: ExportRequest):
    df = pd.DataFrame(request.results)
    # Сводка
    summary = df.groupby("tone").agg(
        tone_percent=("tone", lambda x: round(len(x)/len(df)*100,2)),
        avg_confidence=("avg_confidence", "mean")
    ).reset_index()
    # Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
        summary.to_excel(writer, index=False, sheet_name='Summary')
    output.seek(0)
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": "attachment; filename=SmartTriggers.xlsx"})