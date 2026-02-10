import os
import streamlit as st
import pandas as pd
import requests
import chardet

# =====================
# НАСТРОЙКИ
# =====================
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_TOKEN = os.getenv("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")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# =====================
# 10 триггеров с ключевыми словами
# =====================
TRIGGERS_KEYWORDS = {
    "negative": ["ненавижу", "достало", "бесит", "ужас"],
    "complaint": ["проблема", "не работает", "сломалось", "парковка", "очередь"],
    "spam": ["подпишись", "заработай", "крипта", "ставки", "казино"],
    "praise": ["отлично", "супер", "круто", "хорошо"],
    "service": ["обслуживание", "поддержка", "сервис", "доставка"],
    "feature": ["функция", "опция", "интерфейс", "возможность"],
    "warning": ["ошибка", "сбой", "неудача"],
    "info": ["информация", "новости", "обновление"],
    "suggestion": ["предложение", "идея", "совет", "рекомендую"]
}
ALLOWED_TRIGGERS = list(TRIGGERS_KEYWORDS.keys())

# =====================
# ЧТЕНИЕ ФАЙЛА
# =====================
def read_uploaded_file(uploaded_file):
    raw = uploaded_file.read()
    encoding = chardet.detect(raw)["encoding"] or "utf-8"
    text = raw.decode(encoding, errors="ignore")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Заголовок необязателен
    if lines and lines[0].lower() == "text":
        lines = lines[1:]
    return pd.DataFrame({"text": lines})

# =====================
# ЛОКАЛЬНАЯ КЛАССИФИКАЦИЯ (keyword priority)
# =====================
def classify_local(text: str):
    tl = text.lower()
    for trig, words in TRIGGERS_KEYWORDS.items():
        for w in words:
            if w in tl:
                return trig, 90
    return None, None

# =====================
# AI ФУНКЦИЯ (fallback)
# =====================
def classify_ai(text: str):
    if not HF_TOKEN:
        return "neutral", 40
    prompt = (
        "К какому из триггеров относится текст: "
        f"{', '.join(ALLOWED_TRIGGERS)}?\n"
        f"Текст: {text}"
    )
    try:
        r = requests.post(
            HF_API_URL, headers=HEADERS, json={"inputs": prompt}, timeout=15
        )
        r.raise_for_status()
        result = r.json()
        label = result["labels"][0]
        score = int(result["scores"][0] * 100)
        if label in ALLOWED_TRIGGERS:
            return label, score
    except Exception:
        pass
    return "neutral", 40

# =====================
# ОСНОВНАЯ
# =====================
def analyze(texts):
    rows = []
    for i, text in enumerate(texts, start=1):
        label, conf = classify_local(text)
        if not label:
            label, conf = classify_ai(text)
        rows.append({
            "id": i,
            "text": text,
            "final_trigger": label,
            "confidence_%": conf
        })
    return pd.DataFrame(rows)

# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="Smart Triggers AI", layout="wide")
st.title("Smart Triggers AI — анализ текстов")

# Поле для ручного ввода текста
manual_text = st.text_area("Или введите текст вручную для анализа:")

uploaded = st.file_uploader("Загрузите файл (txt/csv)", type=["txt", "csv"])
texts_to_analyze = []

if manual_text.strip():
    texts_to_analyze.append(manual_text.strip())

if uploaded:
    try:
        df_input = read_uploaded_file(uploaded)
        texts_to_analyze.extend(df_input["text"].tolist())
    except Exception as e:
        st.error(f"Ошибка обработки файла: {e}")

if texts_to_analyze:
    df_result = analyze(texts_to_analyze)
    st.dataframe(df_result)

    # --- Скачивание CSV с нормальной кодировкой для Excel ---
    csv_bytes = df_result.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button(
        "Скачать результат CSV",
        csv_bytes,
        "smart_triggers_result.csv",
        "text/csv"
    )
