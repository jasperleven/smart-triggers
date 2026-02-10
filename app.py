import streamlit as st
import pandas as pd
import re

# =========================
# НАСТРОЙКИ
# =========================

st.set_page_config(
    page_title="Smart Triggers",
    page_icon="⚡",
    layout="centered"
)

# =========================
# ТРИГЕРЫ
# =========================

TRIGGERS = {
    "negative": [
        "надоел", "ужас", "плохо", "ненавижу", "достало",
        "бесит", "отвратительно", "кошмар"
    ],
    "complaint": [
        "парковка", "дорога", "проблема", "не работает",
        "сломалось", "очередь"
    ],
    "spam": [
        "подпишись", "заработок", "доход",
        "крипта", "казино", "ставки"
    ],
    "political": [
        "мэр", "власть", "правительство",
        "выборы", "чиновники"
    ]
}

# =========================
# ЛОГИКА
# =========================

def detect_triggers(text: str):
    text = text.lower()
    found = []

    for trigger, keywords in TRIGGERS.items():
        for word in keywords:
            if re.search(rf"\b{word}\b", text):
                found.append(trigger)
                break

    if not found:
        found.append("neutral")

    confidence = round(100 / len(found), 2)

    return found, confidence


def analyze_texts(texts):
    rows = []

    for idx, text in enumerate(texts, start=1):
        triggers, confidence = detect_triggers(text)

        rows.append({
            "id": idx,
            "text": text,
            "triggers": ", ".join(triggers),
            "confidence_%": confidence,
            "final_trigger": triggers[0]
        })

    return pd.DataFrame(rows)


# =========================
# ИНТЕРФЕЙС
# =========================

st.title("⚡ Smart Triggers")
st.write("Анализ текста и автоматическое определение триггеров")

input_method = st.radio(
    "Формат ввода данных",
    ["Вставить текст", "Загрузить CSV"]
)

texts = []

if input_method == "Вставить текст":
    raw_text = st.text_area(
        "Каждая строка — отдельный текст",
        height=200,
        placeholder="надоела эта парковка"
    )

    if raw_text:
        texts = [line.strip() for line in raw_text.split("\n") if line.strip()]

else:
    uploaded_file = st.file_uploader("Загрузите CSV с колонкой `text`", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        texts = df["text"].dropna().tolist()

# =========================
# РЕЗУЛЬТАТ
# =========================

if texts:
    result = analyze_texts(texts)

    st.subheader("Результат")
    st.dataframe(result, use_container_width=True)

    csv_bytes = result.to_csv(
        index=False,
        encoding="utf-8-sig"
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 Скачать CSV",
        data=csv_bytes,
        file_name="smart_triggers_result.csv",
        mime="text/csv; charset=utf-8"
    )
