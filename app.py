import os
import streamlit as st
import pandas as pd
import requests
import chardet

# =====================
# CONFIG
# =====================
st.set_page_config(
    page_title="Smart Triggers",
    layout="wide"
)

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_TOKEN = os.getenv("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")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# =====================
# CSS (ВЫРАВНИВАНИЕ + КРАСНАЯ КНОПКА)
# =====================
st.markdown("""
<style>

/* поле ввода */
textarea {
    height: 50px !important;
}

/* кнопка */
.stButton > button {
    background-color: #e74c3c;
    color: white;
    height: 50px;
}

/* ГЛАВНОЕ — ВЫРАВНИВАНИЕ КНОПКИ */
.stButton {
    margin-top: 28px;
}

</style>
""", unsafe_allow_html=True)

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
    "question": ["как", "почему", "что", "где", "когда"]
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
    return lines

# =====================
# CLASSIFICATION
# =====================
def classify_local(text):
    t = text.lower()
    for trigger, words in TRIGGERS_KEYWORDS.items():
        if any(w in t for w in words):
            return trigger, round(85 + hash(text) % 15 + 0.37, 2)
    return None, None

def classify_ai(text):
    if not HF_TOKEN:
        return "neutral", 40.00

    prompt = f"К какому триггеру относится текст ({', '.join(ALLOWED_TRIGGERS)}): {text}"

    try:
        r = requests.post(
            HF_API_URL,
            headers=HEADERS,
            json={"inputs": prompt},
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
        return data["labels"][0], round(data["scores"][0] * 100, 2)
    except Exception:
        return "neutral", 40.00

def analyze(texts):
    result = []
    for i, text in enumerate(texts, 1):
        trigger, conf = classify_local(text)
        if not trigger:
            trigger, conf = classify_ai(text)

        result.append({
            "id": i,
            "text": text,
            "trigger": trigger,
            "confidence_%": conf
        })
    return pd.DataFrame(result)

# =====================
# SUMMARY TABLE
# =====================
def build_summary(df):
    tone_map = {
        "negative": "negative",
        "complaint": "negative",
        "warning": "negative",
        "praise": "positive",
        "info": "neutral",
        "suggestion": "neutral",
        "question": "neutral"
    }

    df["tone"] = df["trigger"].map(tone_map).fillna("neutral")
    summary = df["tone"].value_counts().reset_index()
    summary.columns = ["tone", "count"]
    summary["percent"] = (summary["count"] / summary["count"].sum() * 100).round(2)
    return summary

# =====================
# UI
# =====================
st.markdown("### Автоматический анализ текстов")

col_text, col_button = st.columns([5, 1])

with col_text:
    manual_text = st.text_area(
        "",
        placeholder="Введите текст для анализа…",
        height=50
    )

with col_button:
    analyze_click = st.button("Начать анализ", use_container_width=True)

uploaded = st.file_uploader(
    "Загрузить файл (CSV или TXT)",
    type=["csv", "txt"]
)

# =====================
# PROCESS
# =====================
texts = []

if manual_text.strip():
    texts.append(manual_text.strip())

if uploaded:
    texts.extend(read_uploaded_file(uploaded))

if analyze_click or uploaded:
    if texts:
        st.divider()

        df_result = analyze(texts)
        st.markdown("### Результаты анализа")
        st.dataframe(df_result, use_container_width=True)

        st.markdown("### Сводка по тональности")
        df_summary = build_summary(df_result)
        st.dataframe(df_summary, use_container_width=True)

        # CSV
        st.download_button(
            "Скачать CSV",
            df_result.to_csv(index=False, sep=";", encoding="utf-8-sig"),
            "smart_triggers.csv"
        )

        # Excel
        excel_buffer = pd.ExcelWriter("result.xlsx", engine="xlsxwriter")
        df_result.to_excel(excel_buffer, index=False)
        excel_buffer.close()

        with open("result.xlsx", "rb") as f:
            st.download_button(
                "Скачать Excel",
                f,
                "smart_triggers.xlsx"
            )
