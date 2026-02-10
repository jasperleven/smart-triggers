import streamlit as st
import pandas as pd
from io import BytesIO

st.title("Smart Triggers")

# --- Hugging Face токен ---
HF_TOKEN = "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"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
HF_API_URL = "https://api-inference.huggingface.co/models/your-model"

# --- Триггеры ---
TRIGGERS = {
    "negative": ["ужас", "ненавижу", "достало", "плохо", "бесит", "отвратительно", "кошмар"],
    "complaint": ["парковка", "дорога", "проблема", "не работает", "сломалось", "очередь"],
    "spam": ["подпишись", "заработок", "доход", "крипта", "казино", "ставки"],
    "praise": ["отлично", "супер", "круто", "хорошо", "идеально"],
    "feature": ["новая функция", "обновление", "добавить опцию", "функционал"],
    "warning": ["сбой", "ошибка", "не удалось", "проблема"],
    "info": ["новости", "информация", "сообщение"],
    "suggestion": ["предложение", "советую", "рекомендую"]
}

# --- Загрузка файла ---
uploaded_file = st.file_uploader("Загрузите CSV с колонкой 'text'", type=["csv", "txt"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="cp1251")
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
        st.stop()

    if "text" not in df.columns:
        st.error("Файл должен содержать колонку 'text'")
        st.stop()

    # --- Локальная обработка триггеров ---
    def detect_trigger(text):
        text_lower = str(text).lower()
        for key, words in TRIGGERS.items():
            for w in words:
                if w in text_lower:
                    return key
        return "neutral"

    df['final_trigger'] = df['text'].apply(detect_trigger)

    # --- Опционально: HF модель (AI) ---
    # def query_hf(text):
    #     import requests
    #     payload = {"inputs": text}
    #     try:
    #         response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    #         return response.json()
    #     except:
    #         return {"error": "Не удалось получить ответ от HF"}
    # df['hf_result'] = df['text'].apply(query_hf)

    st.subheader("Результаты анализа")
    st.dataframe(df)

    # --- Скачивание CSV с корректной кодировкой ---
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False, sep=';', encoding='utf-8-sig')
    st.download_button(
        label="Скачать CSV с результатами",
        data=csv_buffer.getvalue(),
        file_name="smart_triggers_result.csv",
        mime="text/csv"
    )

