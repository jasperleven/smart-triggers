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

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

TRIGGERS = [
    "negative",
    "complaint",
    "spam",
    "praise",
    "service",
    "feature",
    "warning",
    "info",
    "suggestion",
    "other"
]

KEYWORDS = {
    "negative": ["ненавижу", "достало", "бесит", "ужас"],
    "complaint": ["проблема", "не работает", "очередь", "парковка"],
    "spam": ["подпишись", "заработай", "крипта"],
    "praise": ["отлично", "супер", "круто"],
    "warning": ["ошибка", "сбой"],
    "suggestion": ["предложение", "идея", "совет"]
}

# =====================
# ЧТЕНИЕ ФАЙЛА (БЕЗ ПОРЧИ КОДИРОВКИ)
# =====================
def read_uploaded_file(uploaded_file):
    raw = uploaded_file.read()
    encoding = chardet.detect(raw)["encoding"]
    text = raw.decode(encoding)

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines[0].lower() == "text":
        lines = lines[1:]

    return pd.DataFrame({"text": lines})

# =====================
# AI КЛАССИФИКАЦИЯ
# =====================
def ai_classify(text):
    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": TRIGGERS
        }
    }

    r = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=20)
    r.raise_for_status()
    res = r.json()

    label = res["labels"][0]
    score = int(res["scores"][0] * 100)

    return label, score

# =====================
# KEYWORD BOOST
# =====================
def keyword_boost(text, label, score):
    text_l = text.lower()
    for k, words in KEYWORDS.items():
        if k == label:
            for w in words:
                if w in text_l:
                    return min(score + 10, 95)
    return score

# =====================
# ОБЩАЯ ЛОГИКА
# =====================
def analyze(texts):
    rows = []

    for i, text in enumerate(texts, 1):
        try:
            label, conf = ai_classify(text)
            conf = keyword_boost(text, label, conf)
        except Exception:
            label, conf = "other", 40

        rows.append({
            "id": i,
            "text": text,
            "final_trigger": label,
            "confidence_%": conf
        })

    return pd.DataFrame(rows)

# =====================
# UI
# =====================
st.set_page_config("Smart Triggers AI")
st.title("Smart Triggers AI")

uploaded = st.file_uploader("Загрузите файл (txt / csv)", type=["txt", "csv"])

if uploaded:
    try:
        df_input = read_uploaded_file(uploaded)
        df_result = analyze(df_input["text"].tolist())

        st.dataframe(df_result)

        st.download_button(
            "Скачать CSV",
            df_result.to_csv(index=False, encoding="utf-8-sig"),
            "smart_triggers_result.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(str(e))
