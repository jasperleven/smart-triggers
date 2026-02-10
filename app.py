import streamlit as st
import pandas as pd
import os
from gpt4all import GPT4All

# -----------------------------
# 1. Настройки триггеров
# -----------------------------
TRIGGERS = [
    "negative",
    "complaint",
    "spam",
    "positive",
    "suggestion",
    "question",
    "news_info",
    "discussion",
    "irony_sarcasm"
]

# -----------------------------
# 2. Инициализация GPT4All
# -----------------------------
MODEL_PATH = "ggml-gpt4all-j-v1.3-groovy.bin"

# Скачивание модели при первом запуске (если файла нет)
if not os.path.exists(MODEL_PATH):
    st.info("Скачиваем модель GPT4All...")
    # Пример: можно дать ссылку на официальный источник
    # os.system(f"wget -O {MODEL_PATH} <ссылка-на-модель>")
    st.warning("Скачайте модель вручную и положите в папку с app.py")
else:
    st.success("Модель найдена, инициализация AI...")
    
model = GPT4All(MODEL_PATH)

# -----------------------------
# 3. Функция классификации
# -----------------------------
def ai_classify(text: str):
    prompt = f"""
    Классифицируй текст на один из триггеров:
    {', '.join(TRIGGERS)}
    
    Текст: "{text}"
    
    Верни JSON:
    {{
      "main_trigger": "<trigger>",
      "confidence": <0-1>
    }}
    """
    response = model.generate(prompt)
    import json
    try:
        json_line = response.split("\n")[0]
        result = json.loads(json_line)
        trigger = result.get("main_trigger", "discussion")
        confidence = float(result.get("confidence", 0.6))
        return trigger, confidence
    except Exception:
        return "discussion", 0.6

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.title("Smart Triggers — локальный AI")
st.write("Определение триггеров текста с использованием GPT4All")

# Загрузка CSV
uploaded_file = st.file_uploader("Загрузите CSV с колонкой 'text'", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("CSV должен содержать колонку 'text'")
    else:
        results = []
        for idx, row in df.iterrows():
            text = row["text"]
            main_trigger, confidence = ai_classify(text)
            results.append({
                "id": idx + 1,
                "text": text,
                "main_trigger": main_trigger,
                "confidence": round(confidence * 100, 2)
            })
        df_result = pd.DataFrame(results)
        st.write(df_result)

        csv = df_result.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="Скачать результат",
            data=csv,
            file_name="smart_triggers_result.csv",
            mime="text/csv"
        )

# Поле для ввода текста напрямую
st.write("Или попробуйте ввести текст вручную:")
user_text = st.text_area("Введите текст")
if st.button("Классифицировать"):
    if user_text:
        trigger, confidence = ai_classify(user_text)
        st.write(f"**Триггер:** {trigger} | **Confidence:** {round(confidence*100,2)}%")
    else:
        st.warning("Введите текст для классификации")
