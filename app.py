# app.py
import streamlit as st
import pandas as pd
import requests

# =====================
# Настройки Hugging Face
# =====================
API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/cointegrated/rubert-tiny2"  # лёгкая русскоязычная модель
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# =====================
# Триггеры
# =====================
TRIGGERS = {
    "negative": ["надоел", "ужас", "плохо", "ненавижу", "достало", "бесит", "отвратительно", "кошмар"],
    "complaint": ["парковка", "дорога", "проблема", "не работает", "сломалось", "очередь"],
    "spam": ["подпишись", "заработок", "доход", "крипта", "казино", "ставки"]
}

# =====================
# Функции
# =====================
def classify_text_hf(text):
    """Классификация через Hugging Face Inference API"""
    payload = {"inputs": text}
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            return response.json()  # здесь можно добавить обработку ответа модели, если нужно
        else:
            return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}

def analyze_triggers(text):
    """Поиск ключевых триггеров в тексте"""
    found = []
    for trig, words in TRIGGERS.items():
        if any(word in text.lower() for word in words):
            found.append(trig)
    confidence = 100 if found else 0
    return {"text": text, "triggers": found if found else ["none"], "confidence": confidence}

def process_texts(texts):
    results = []
    for text in texts:
        # Можно раскомментировать, если хочешь использовать модель HF для анализа:
        # model_result = classify_text_hf(text)
        trigger_result = analyze_triggers(text)
        results.append(trigger_result)
    return pd.DataFrame(results)

# =====================
# Streamlit интерфейс
# =====================
st.title("Smart Triggers AI")
st.write("Загрузите CSV или введите текст для анализа триггеров.")

# --- Ввод текста вручную ---
user_input = st.text_area("Введите текст для анализа:")

if st.button("Анализировать текст"):
    if user_input.strip() != "":
        df_result = process_texts([user_input])
        st.dataframe(df_result)
        csv = df_result.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="Скачать результат CSV",
            data=csv,
            file_name="smart_triggers_result.csv",
            mime="text/csv"
        )
    else:
        st.warning("Введите текст для анализа!")

# --- Загрузка CSV ---
uploaded_file = st.file_uploader("Или загрузите CSV (одна колонка с текстом)", type=["csv"])
if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        if "text" not in df_uploaded.columns:
            st.error("CSV должен содержать колонку с названием 'text'")
        else:
            df_result = process_texts(df_uploaded["text"].tolist())
            st.dataframe(df_result)
            csv = df_result.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="Скачать результат CSV",
                data=csv,
                file_name="smart_triggers_result.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")
