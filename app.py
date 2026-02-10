import streamlit as st
import pandas as pd

st.set_page_config(page_title="Smart Triggers", layout="wide")
st.title("Smart Triggers")
st.write("Автоматический анализ текстов по умным триггерам")

TRIGGERS = {
    "negative": ["надоел", "ужас", "плохо", "ненавижу", "достало", "бесит"],
    "complaint": ["парковка", "дорога", "проблема", "не работает", "сломалось"],
    "spam": ["подпишись", "заработок", "доход", "крипта", "казино"],
    "political": ["мэр", "власть", "правительство", "выборы"]
}

def analyze_texts(texts):
    rows = []
    for i, text in enumerate(texts, start=1):
        text_l = text.lower()
        matched = [
            trigger for trigger, words in TRIGGERS.items()
            if any(word in text_l for word in words)
        ]
        rows.append({
            "id": i,
            "text": text,
            "triggers": ", ".join(matched) if matched else "none",
            "final_label": final_label(matched)
        })
    return pd.DataFrame(rows)

def final_label(triggers):
    if "spam" in triggers:
        return "spam"
    if "complaint" in triggers or "negative" in triggers:
        return "complaint"
    if "political" in triggers:
        return "political"
    return "neutral"

uploaded_file = st.file_uploader("Загрузите CSV с колонкой text", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    texts = df_input["text"].astype(str).tolist()

    if st.button("Анализировать"):
        result = analyze_texts(texts)
        st.dataframe(result)

        csv = result.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "Скачать результат CSV",
            csv,
            "smart_triggers_result.csv",
            "text/csv"
        )
