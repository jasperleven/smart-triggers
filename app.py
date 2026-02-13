import os
import streamlit as st
import pandas as pd
import requests
import chardet

# =====================
# CONFIG
# =====================
st.set_page_config(
    page_title="Smart Triggers",
    layout="wide"
)

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_TOKEN = os.getenv("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")  # добавь свой токен через переменную окружения
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# =====================
# TRIGGERS
# =====================
TRIGGERS_KEYWORDS = {
    "negative": ["ненавижу", "достало", "бесит", "ужас", "проблема", "не работает", "сломалось", "очередь"],
    "praise": ["отлично", "супер", "круто", "хорошо"],
    "info": ["информация", "новости", "обновление"],
    "suggestion": ["предложение", "идея", "совет"],
    "question": ["?", "вопрос", "как", "что", "почему"]
}
ALLOWED_TRIGGERS = list(TRIGGERS_KEYWORDS.keys())

# =====================
# FILE READER
# =====================
def read_uploaded_file(uploaded_file):
    raw = uploaded_file.read()
    encoding = chardet.detect(raw)["encoding"] or "utf-8"
    text = raw.decode(encoding, errors="ignore")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and lines[0].lower() == "text":
        lines = lines[1:]
    return pd.DataFrame({"text": lines})

# =====================
# CLASSIFIERS
# =====================
def classify_local(text):
    tl = text.lower()
    for trig, words in TRIGGERS_KEYWORDS.items():
        if any(w in tl for w in words):
            return trig, 90.72
    return None, None

def classify_ai(text):
    if not HF_TOKEN:
        return "neutral", 40.00

    prompt = (
        f"Which trigger does this text belong to: {', '.join(ALLOWED_TRIGGERS)}?\n"
        f"Text: {text}"
    )
    try:
        r = requests.post(
            HF_API_URL,
            headers=HEADERS,
            json={"inputs": prompt},
            timeout=15
        )
        r.raise_for_status()
        result = r.json()
        label = result["labels"][0]
        score = float(result["scores"][0] * 100)
        if label in ALLOWED_TRIGGERS:
            return label, round(score, 2)
    except Exception:
        pass

    return "neutral", 40.00

def analyze(texts):
    rows = []
    for i, text in enumerate(texts, 1):
        label, conf = classify_local(text)
        if not label:
            label, conf = classify_ai(text)

        rows.append({
            "id": i,
            "text": text,
            "trigger": label,
            "confidence_%": conf
        })
    return pd.DataFrame(rows)

# =====================
# TONE SUMMARY
# =====================
def build_tone_summary(df):
    summary = df.groupby("trigger").size().reset_index(name="count")
    total = summary["count"].sum()
    summary["percent"] = (summary["count"] / total * 100).round(2)
    return summary

# =====================
# HEADER + INPUT
# =====================
st.markdown("### Enter text or upload file for analysis")

col_input, col_button = st.columns([4, 1])
with col_input:
    manual_text = st.text_area("", placeholder="Enter text…", height=80)
with col_button:
    analyze_click = st.button("Start Analysis", use_container_width=True, key="btn_text")

uploaded = st.file_uploader("Or upload CSV/TXT file", type=["csv", "txt"], key="file_uploader")

# =====================
# PROCESSING
# =====================
texts = []

if manual_text.strip():
    texts.append(manual_text.strip())

if uploaded:
    try:
        df_uploaded = read_uploaded_file(uploaded)
        texts.extend(df_uploaded["text"].tolist())
        analyze_click = True  # автоматически запускаем анализ после загрузки
    except Exception as e:
        st.error(f"File error: {e}")

if analyze_click and texts:
    st.divider()
    st.markdown("### Analysis Results")

    df_result = analyze(texts)
    st.dataframe(df_result, use_container_width=True)

    # Tone summary table
    df_summary = build_tone_summary(df_result)
    st.markdown("### Summary by Trigger (%)")
    st.table(df_summary)

    # CSV download
    csv_bytes = df_result.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "Download CSV",
        csv_bytes,
        "smart_triggers_result.csv",
        "text/csv",
        key="download_csv"
    )

    # Excel download
    excel_bytes = pd.ExcelWriter("smart_triggers_result.xlsx", engine="xlsxwriter")
    df_result.to_excel("smart_triggers_result.xlsx", index=False)
    st.download_button(
        "Download Excel",
        open("smart_triggers_result.xlsx", "rb").read(),
        "smart_triggers_result.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_excel"
    )
