import os
import streamlit as st
import pandas as pd
import chardet
import requests

# =====================
# CONFIG
# =====================
st.set_page_config(page_title="Smart Triggers", layout="wide")

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_TOKEN = os.getenv("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")  # твой токен HuggingFace
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# =====================
# TRIGGERS
# =====================
TRIGGERS_KEYWORDS = {
    "negative": ["ненавижу", "достало", "бесит", "ужас"],
    "complaint": ["проблема", "не работает", "сломалось", "очередь"],
    "praise": ["отлично", "супер", "круто", "хорошо"],
    "warning": ["ошибка", "сбой"],
    "info": ["информация", "новости", "обновление"],
    "suggestion": ["предложение", "идея", "совет"],
    "question": ["как", "почему", "что если", "можно ли"]
}

ALLOWED_TRIGGERS = list(TRIGGERS_KEYWORDS.keys())

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
        if any(w in tl for w in words):
            return trig, 90.72
    return None, None

def classify_ai(text):
    if not HF_TOKEN:
        return "neutral", 40.0

    prompt = (
        f"К какому из триггеров относится текст: {', '.join(ALLOWED_TRIGGERS)}?\n"
        f"Текст: {text}"
    )
    try:
        r = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": prompt}, timeout=15)
        r.raise_for_status()
        result = r.json()
        label = result.get("labels", [None])[0]
        score = int(result.get("scores", [0])[0] * 100)
        if label in ALLOWED_TRIGGERS:
            return label, score
    except Exception:
        pass
    return "neutral", 40.0

def analyze(texts):
    rows = []
    for i, text in enumerate(texts, 1):
        label, conf = classify_local(text)
        if not label:
            label, conf = classify_ai(text)
        rows.append({"id": i, "text": text, "trigger": label, "confidence_%": conf})
    return pd.DataFrame(rows)

# =====================
# TONE SUMMARY
# =====================
def build_tone_summary(df):
    tone_map = {"negative": "Negative", "complaint": "Negative",
                "praise": "Positive", "warning": "Negative",
                "info": "Neutral", "suggestion": "Neutral", "question": "Neutral"}
    df["tone"] = df["trigger"].map(tone_map).fillna("Neutral")
    summary = df.groupby("tone").size().reset_index(name="count")
    total = summary["count"].sum()
    summary["percent"] = (summary["count"] / total * 100).round(2)
    return summary

# =====================
# HEADER
# =====================
st.markdown("### Smart Triggers Analyzer")

# =====================
# INPUT BLOCK
# =====================
col_text, col_analyze = st.columns([4, 1])
with col_text:
    manual_text = st.text_area("", placeholder="Введите текст…", height=80)
with col_analyze:
    analyze_click = st.button("Начать анализ", use_container_width=True)

col_file, col_upload = st.columns([4, 1])
with col_file:
    uploaded = st.file_uploader("", type=["csv", "txt"], label_visibility="collapsed")
with col_upload:
    upload_click = st.button("Загрузить файл", use_container_width=True)

# =====================
# COLLECT TEXTS
# =====================
texts = []
if manual_text.strip() and analyze_click:
    texts.append(manual_text.strip())

if uploaded:
    try:
        df_uploaded = read_uploaded_file(uploaded)
        texts.extend(df_uploaded["text"].tolist())
    except Exception as e:
        st.error(f"Ошибка файла: {e}")

# =====================
# ANALYSIS
# =====================
if texts:
    df_result = analyze(texts)
    st.divider()
    st.markdown("### Результаты анализа")
    st.dataframe(df_result, use_container_width=True)

    # Summary table
    df_summary = build_tone_summary(df_result)
    st.markdown("### Тональность в %")
    st.dataframe(df_summary, use_container_width=True)

    # EXPORT
    csv_bytes = df_result.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    excel_bytes = df_result.to_excel(index=False, engine="openpyxl").to_bytes()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Скачать CSV", csv_bytes, "smart_triggers_result.csv", "text/csv")
    with col2:
        st.download_button("Скачать Excel", excel_bytes, "smart_triggers_result.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
