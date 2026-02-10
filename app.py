import os
import streamlit as st
import pandas as pd
import requests

# =====================
# НАСТРОЙКИ
# =====================
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_TOKEN = os.getenv("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")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# =====================
# 10 УТВЕРЖДЁННЫХ ТРИГГЕРОВ
# =====================
TRIGGERS = {
    "negative": ["ненавижу", "ужас", "достало", "бесит", "кошмар"],
    "complaint": ["проблема", "не работает", "сломалось", "парковка", "очередь"],
    "spam": ["подпишись", "заработай", "крипта", "казино", "ставки"],
    "praise": ["отлично", "супер", "круто", "нравится"],
    "service": ["поддержка", "сервис", "обслуживание", "доставка"],
    "feature": ["функция", "опция", "интерфейс", "возможность"],
    "warning": ["ошибка", "сбой", "авария"],
    "info": ["информация", "новости", "обновление"],
    "suggestion": ["предложение", "идея", "совет", "рекомендую"],
    "other": []
}

ALLOWED_TRIGGERS = list(TRIGGERS.keys())

# =====================
# ЛОКАЛЬНЫЙ АНАЛИЗ
# =====================
def local_trigger(text: str):
    text = text.lower()
    for trigger, words in TRIGGERS.items():
        for w in words:
            if w in text:
                return trigger, 90
    return None, None

# =====================
# AI АНАЛИЗ (HF)
# =====================
def ai_trigger(text: str):
    prompt = (
        "Выбери ОДИН триггер из списка:\n"
        f"{', '.join(ALLOWED_TRIGGERS)}\n\n"
        f"Текст: {text}\n\n"
        "Ответь ТОЛЬКО названием триггера."
    )

    try:
        r = requests.post(
            HF_API_URL,
            headers=HEADERS,
            json={"inputs": prompt},
            timeout=15
        )
        r.raise_for_status()
        result = r.json()[0]["generated_text"].strip().lower()

        for trig in ALLOWED_TRIGGERS:
            if trig in result:
                return trig, 70

    except Exception:
        pass

    return "other", 40

# =====================
# ОСНОВНАЯ ЛОГИКА
# =====================
def analyze_texts(texts):
    rows = []
    for i, text in enumerate(texts, start=1):

        trig, conf = local_trigger(text)

        if trig is None:
            trig, conf = ai_trigger(text)

        rows.append({
            "id": i,
            "text": text,
            "final_trigger": trig,
            "confidence_%": conf
        })

    return pd.DataFrame(rows)

# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="Smart Triggers AI", layout="centered")
st.title("Smart Triggers AI")
st.write("Анализ текстов по триггерам с использованием AI")

# ---- Ручной ввод
user_text = st.text_area("Введите текст")

if st.button("Анализировать текст"):
    if user_text.strip():
        df = analyze_texts([user_text])
        st.dataframe(df)
        st.download_button(
            "Скачать CSV",
            df.to_csv(index=False, encoding="utf-8-sig"),
            "result.csv",
            "text/csv"
        )

# ---- CSV загрузка
uploaded_file = st.file_uploader("Или загрузите CSV (колонка text)", type="csv")

if uploaded_file:
    try:
        try:
            df_input = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            df_input = pd.read_csv(uploaded_file, encoding="cp1251")

        if "text" not in df_input.columns:
            st.error("CSV должен содержать колонку 'text'")
        else:
            df_result = analyze_texts(df_input["text"].astype(str).tolist())
            st.dataframe(df_result)
            st.download_button(
                "Скачать результат CSV",
                df_result.to_csv(index=False, encoding="utf-8-sig"),
                "smart_triggers_result.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Ошибка обработки файла: {e}")
