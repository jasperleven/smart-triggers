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
# TRIGGERS
# =====================
TRIGGERS_KEYWORDS = {
    "negative": ["ненавижу", "достало", "бесит", "ужас"],
    "complaint": ["проблема", "не работает", "сломалось", "очередь"],
    "praise": ["отлично", "супер", "круто", "хорошо"],
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
        r = requests.post(
            HF_API_URL,
            headers=HEADERS,
            json={"inputs": prompt},
            timeout=15
        )
        r.raise_for_status()
        result = r.json()
        label = result["labels"][0]
        score = float(result["scores"][0] * 100)
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
        rows.append({
            "id": i,
            "text": text,
            "trigger": label,
            "confidence_%": round(conf, 2)
        })
    return pd.DataFrame(rows)

# =====================
# SUMMARY TABLE
# =====================
def build_tone_summary(df):
    tone_map = {"negative": "Negative", "praise": "Positive", "info": "Neutral",
                "complaint": "Negative", "suggestion": "Neutral", "question": "Neutral"}
    df["tone"] = df["trigger"].map(tone_map).fillna("Neutral")
    summary = df.groupby("tone").size().reset_index(name="count")
    total = summary["count"].sum()
    summary["percent"] = (summary["count"] / total * 100).round(2)
    return summary

# =====================
# HEADER
# =====================
st.markdown("## Smart Triggers Analyzer")

# =====================
# INPUT AREA
# =====================
col_input, col_button = st.columns([5, 1])
with col_input:
    manual_text = st.text_area("", placeholder="Введите текст…", height=100)
with col_button:
    analyze_click = st.button(
        "Начать анализ",
        use_container_width=True,
        type="primary"
    )

uploaded = st.file_uploader("Загрузить файл (CSV/TXT)", type=["csv", "txt"])

# =====================
# PROCESSING
# =====================
texts = []

if manual_text.strip() and analyze_click:
    texts.append(manual_text.strip())

if uploaded:
    try:
        df_uploaded = read_uploaded_file(uploaded)
        texts.extend(df_uploaded["text"].tolist())
        analyze_click = True  # авто-анализ при загрузке
    except Exception as e:
        st.error(f"Ошибка файла: {e}")

if analyze_click and texts:
    st.divider()
    st.markdown("### Результаты анализа")

    df_result = analyze(texts)
    st.dataframe(df_result, use_container_width=True)

    # Таблица тонов сбоку
    df_summary = build_tone_summary(df_result)
    st.markdown("### Процентное распределение тонов")
    st.dataframe(df_summary, use_container_width=True)

    # Экспорт CSV и Excel
    csv_bytes = df_result.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    excel_bytes = df_result.to_excel(index=False).encode("utf-8")

    st.download_button("Скачать CSV", csv_bytes, "smart_triggers_result.csv", "text/csv")
    st.download_button("Скачать Excel", excel_bytes, "smart_triggers_result.xlsx", "application/vnd.ms-excel")
