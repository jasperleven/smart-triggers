import os
import streamlit as st
import pandas as pd
import requests
import io

# =====================
# НАСТРОЙКИ
# =====================
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_TOKEN = os.getenv("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")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# =====================
# ТРИГГЕРЫ (10 ШТУК)
# =====================
TRIGGERS = {
    "negative": ["ненавижу", "достало", "бесит", "ужас", "кошмар"],
    "complaint": ["проблема", "не работает", "сломалось", "парковка", "очередь"],
    "spam": ["подпишись", "заработай", "крипта", "казино", "ставки"],
    "praise": ["отлично", "супер", "круто", "нравится"],
    "service": ["обслуживание", "поддержка", "сервис", "доставка"],
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
# AI АНАЛИЗ
# =====================
def ai_trigger(text: str):
    prompt = (
        "Определи ОДИН триггер строго из списка:\n"
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

        result = r.json()[0]["generated_text"].lower().strip()

        for t in ALLOWED_TRIGGERS:
            if t in result:
                return t, 70

    except Exception:
        pass

    return "other", 40

# =====================
# ОСНОВНАЯ ЛОГИКА
# =====================
def analyze_texts(texts):
    rows = []

    for i, text in enumerate(texts, start=1):
        trigger, confidence = local_trigger(text)

        if trigger is None:
            trigger, confidence = ai_trigger(text)

        rows.append({
            "id": i,
            "text": text,
            "final_trigger": trigger,
            "confidence_%": confidence
        })

    return pd.DataFrame(rows)

# =====================
# ЧТЕНИЕ ФАЙЛА (УЛЬТРА НАДЁЖНО)
# =====================
def read_any_text_file(uploaded_file):
    raw = uploaded_file.read()

    for enc in ["utf-8-sig", "utf-8", "cp1251"]:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Не удалось определить кодировку файла")

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # если первая строка = text → пропускаем
    if lines and lines[0].lower() == "text":
        lines = lines[1:]

    return pd.DataFrame({"text": lines})

# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="Smart Triggers AI", layout="centered")
st.title("Smart Triggers AI")
st.write("AI-анализ текстов и CSV без проблем с форматами")

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

# ---- Загрузка файла
uploaded_file = st.file_uploader(
    "Загрузите CSV / TXT / файл из Excel (одна строка = один текст)",
    type=["csv", "txt"]
)

if uploaded_file:
    try:
        df_input = read_any_text_file(uploaded_file)
        df_result = analyze_texts(df_input["text"].tolist())

        st.dataframe(df_result)

        st.download_button(
            "Скачать результат CSV",
            df_result.to_csv(index=False, encoding="utf-8-sig"),
            "smart_triggers_result.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Ошибка обработки файла: {e}")
