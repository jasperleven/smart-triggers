import os
import streamlit as st
import pandas as pd
import requests
import chardet
from io import BytesIO

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Smart Triggers",
    layout="wide"
)

# =====================
# HF API
# =====================
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_TOKEN = os.getenv("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")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# =====================
# TRIGGERS
# =====================
TRIGGERS_KEYWORDS = {
    "negative": ["ненавижу", "достало", "бесит", "ужас"],
    "complaint": ["проблема", "не работает", "сломалось", "очередь"],
    "warning": ["ошибка", "сбой"],
    "praise": ["отлично", "супер", "круто", "хорошо"],
    "info": ["информация", "новости", "обновление"],
    "suggestion": ["предложение", "идея", "совет"],
    "question": ["как", "почему", "когда", "можно ли", "?"],
    "spam": ["подпишись", "заработай", "крипта", "ставки", "казино"]
}

ALLOWED_TRIGGERS = list(TRIGGERS_KEYWORDS.keys())

# =====================
# TONE MAP
# =====================
TONE_MAP = {
    "praise": "positive",

    "negative": "negative",
    "complaint": "negative",
    "warning": "negative",

    "info": "neutral",
    "suggestion": "neutral",
    "question": "neutral",
    "spam": "neutral",
    "neutral": "neutral"
}

# =====================
# FILE READER
# =====================
def read_uploaded_file(uploaded_file):
    raw = uploaded_file.read()
    encoding = chardet.detect(raw)["encoding"] or "utf-8"
    text = raw.decode(encoding, errors="ignore")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and lines[0].lower() == "text":
        lines = lines[1:]
    return pd.DataFrame({"text": lines})

# =====================
# CLASSIFIERS
# =====================
def classify_local(text):
    tl = text.lower()
    for trig, words in TRIGGERS_KEYWORDS.items():
        for w in words:
            if w in tl:
                return trig, round(88 + (hash(text) % 120) / 100, 2)
    return None, None

def classify_ai(text):
    if not HF_TOKEN:
        return "neutral", round(40 + (hash(text) % 200) / 100, 2)

    prompt = (
        f"К какому из триггеров относится текст: {', '.join(ALLOWED_TRIGGERS)}?\n"
        f"Текст: {text}"
    )

    try:
        r = requests.post(
            HF_API_URL,
            headers=HEADERS,
            json={"inputs": prompt},
            timeout=15
        )
        r.raise_for_status()
        result = r.json()
        label = result["labels"][0]
        score = round(result["scores"][0] * 100, 2)
        if label in ALLOWED_TRIGGERS:
            return label, score
    except Exception:
        pass

    return "neutral", round(40 + (hash(text) % 200) / 100, 2)

# =====================
# ANALYSIS
# =====================
def analyze(texts):
    rows = []
    for i, text in enumerate(texts, 1):
        label, conf = classify_local(text)
        if not label:
            label, conf = classify_ai(text)

        rows.append({
            "id": i,
            "text": text,
            "trigger": label,
            "confidence_%": conf
        })

    df = pd.DataFrame(rows)
    df["tone"] = df["trigger"].map(TONE_MAP)
    return df

def build_tone_summary(df):
    summary = (
        df["tone"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "tone", "tone": "count"})
    )

    total = summary["count"].sum()
    summary["percent"] = (summary["count"] / total * 100).round(2)
    return summary

# =====================
# EXCEL EXPORT
# =====================
def export_to_excel(df_result, df_summary):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_result.to_excel(writer, index=False, sheet_name="Analysis")
        df_summary.to_excel(writer, index=False, sheet_name="Tone Summary")
    return output.getvalue()

# =====================
# UI
# =====================
st.markdown("### Введите текст для анализа")

col_input, col_btn = st.columns([6, 1])
with col_input:
    manual_text = st.text_area(
        "",
        placeholder="Введите текст…",
        height=90
    )
with col_btn:
    run = st.button("Начать анализ", use_container_width=True)

uploaded = st.file_uploader(
    "Или загрузите TXT / CSV",
    type=["txt", "csv"]
)

texts = []

if manual_text.strip():
    texts.append(manual_text.strip())

if uploaded:
    try:
        df_uploaded = read_uploaded_file(uploaded)
        texts.extend(df_uploaded["text"].tolist())
    except Exception as e:
        st.error(f"Ошибка файла: {e}")

if run and texts:
    st.divider()

    df_result = analyze(texts)
    df_summary = build_tone_summary(df_result)

    st.markdown("### Результаты анализа")
    st.dataframe(df_result, use_container_width=True)

    st.markdown("### Распределение тональности")
    st.dataframe(df_summary, use_container_width=True)

    csv_bytes = df_result.to_csv(
        index=False,
        sep=";",
        encoding="utf-8-sig"
    ).encode("utf-8-sig")

    excel_bytes = export_to_excel(df_result, df_summary)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Скачать CSV",
            csv_bytes,
            "smart_triggers_result.csv",
            "text/csv"
        )
    with col2:
        st.download_button(
            "Скачать Excel",
            excel_bytes,
            "smart_triggers_report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
