# app.py
import streamlit as st
import pandas as pd
import requests

# =====================
# 10 утверждённых триггеров
# =====================
TRIGGERS = {
    "negative": ["надоел", "ужас", "плохо", "ненавижу", "достало", "бесит", "отвратительно", "кошмар"],
    "complaint": ["парковка", "дорога", "проблема", "не работает", "сломалось", "очередь"],
    "spam": ["подпишись", "заработок", "доход", "крипта", "казино", "ставки"],
    "praise": ["отлично", "супер", "круто", "хорошо", "нравится"],
    "service": ["поддержка", "сервис", "обслуживание", "доставка"],
    "feature": ["функция", "опция", "возможность", "интерфейс"],
    "warning": ["ошибка", "сбой", "проблема", "не работает"],
    "info": ["информация", "новости", "обновление", "уведомление"],
    "suggestion": ["предложение", "идея", "совет", "рекомендация"],
    "other": ["прочее", "другое", "разное"]
}

HF_API_URL = "https://api-inference.huggingface.co/models/<ваша_модель>"
HF_TOKEN = "<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>"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# =====================
# Функции анализа
# =====================
def analyze_local(text):
    """Локальная проверка текста по триггерам"""
    found = []
    for trig, words in TRIGGERS.items():
        if any(word in text.lower() for word in words):
            found.append(trig)
    if found:
        return found[0], ", ".join(found), 100
    return None, None, None

def analyze_hf(text):
    """Обработка через Hugging Face"""
    prompt = (
        "Определи, какой из триггеров подходит к тексту: "
        "negative, complaint, spam, praise, service, feature, warning, info, suggestion, other.\n"
        f"Текст: {text}"
    )
    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": prompt})
        response.raise_for_status()
        result_text = response.json()[0]["generated_text"].strip().lower()
        for trig in TRIGGERS.keys():
            if trig in result_text:
                return trig, trig, 100
    except:
        pass
    return "other", "other", 50

def analyze_text(text):
    """Полная обработка текста"""
    final, triggers, confidence = analyze_local(text)
    if final is not None:
        return {"text": text, "triggers": triggers, "confidence_%": confidence, "final_trigger": final}
    else:
        final_hf, triggers_hf, conf_hf = analyze_hf(text)
        return {"text": text, "triggers": triggers_hf, "confidence_%": conf_hf, "final_trigger": final_hf}

def process_texts(texts):
    return pd.DataFrame([analyze_text(t) for t in texts])

# =====================
# Streamlit интерфейс
# =====================
st.title("Smart Triggers AI")
st.write("Загрузите CSV или введите текст для анализа триггеров.")

# Ввод текста вручную
user_input = st.text_area("Введите текст для анализа:")

if st.button("Анализировать текст"):
    if user_input.strip() != "":
        df_result = process_texts([user_input])
        st.dataframe(df_result)
        csv = df_result.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("Скачать CSV", csv, "smart_triggers_result.csv", "text/csv")
    else:
        st.warning("Введите текст!")

# Загрузка CSV
uploaded_file = st.file_uploader("Или загрузите CSV (колонка 'text')", type=["csv"])
if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        if "text" not in df_uploaded.columns:
            st.error("CSV должен содержать колонку 'text'")
        else:
            df_result = process_texts(df_uploaded["text"].tolist())
            st.dataframe(df_result)
            csv = df_result.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("Скачать CSV", csv, "smart_triggers_result.csv", "text/csv")
    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")
