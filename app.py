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
HF_TOKEN = os.getenv("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")  # вставь свой токен в переменные окружения
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# =====================
# TRIGGERS
# =====================
TRIGGERS_KEYWORDS = {
    "negative": ["ненавижу", "достало", "бесит", "ужас", "проблема", "не работает", "сломалось", "очередь"],
    "praise": ["отлично", "супер", "круто", "хорошо"],
    "info": ["информация", "новости", "обновление"],
    "suggestion": ["предложение", "идея", "совет"],
    "question": ["?", "вопрос", "как", "что", "почему"]
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
        return "neutral", 40.00

    prompt = (
        f"Which trigger does this text belong to: {', '.join(ALLOWED_TRIGGERS)}?\n"
        f"Text: {text}"
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
            return label, round(score, 2)
    except Exception:
        pass

    return "neutral", 40.00

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
    return pd.DataFrame(rows)

def build_tone_summary(df):
    summary = df.groupby("trigger").size().reset_index(name="count")
    total = summary["count"].sum()
    summary["percent"] = (summary["count"] / total * 100).round(2)
    return summary

# =====================
# INPUT BLOCK
# =====================
st.markdown("### Введите текст или загрузите файл")

# текстовое поле с кнопкой параллельно
col_text, col_analyze = st.columns([4, 1])
with col_text:
    manual_text = st.text_area("", placeholder="Введите текст…", height=80)
with col_analyze:
    analyze_click = st.button("Начать анализ", use_container_width=True, key="btn_text", help="Запустить анализ текста")

# файл с кнопкой параллельно
col_file, col_upload = st.columns([4, 1])
with col_file:
    uploaded = st.file_uploader("", type=["csv", "txt"], key="file_uploader")
with col_upload:
    st.markdown("")
    upload_click = st.button("Загрузить файл", use_container_width=True, key="btn_file", help="Загрузить и проанализировать файл")

# =====================
# PROCESSING
# =====================
texts = []

if manual_text.strip():
    texts.append(manual_text.strip())

if uploaded:
    try:
        df_uploaded = read_uploaded_file(uploaded)
        texts.extend(df_uploaded["text"].tolist())
        analyze_click = True  # автоматически запускаем анализ после загрузки
    except Exception as e:
        st.error(f"Ошибка файла: {e}")

if analyze_click and texts:
    st.divider()
    st.markdown("### Результаты анализа")

    df_result = analyze(texts)
    st.dataframe(df_result, use_container_width=True)

    # Summary table
    df_summary = build_tone_summary(df_result)
    st.markdown("### Сводка по тональности (%)")
    st.table(df_summary)

    # CSV download
    csv_bytes = df_result.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "Скачать CSV",
        csv_bytes,
        "smart_triggers_result.csv",
        "text/csv",
        key="download_csv"
    )

    # Excel download
    excel_path = "smart_triggers_result.xlsx"
    df_result.to_excel(excel_path, index=False, engine="xlsxwriter")
    with open(excel_path, "rb") as f:
        st.download_button(
            "Скачать Excel",
            f.read(),
            "smart_triggers_result.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel"
        )
