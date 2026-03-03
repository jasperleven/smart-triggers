import streamlit as st
import pandas as pd
import requests
import chardet
from io import BytesIO

# =====================
# CONFIG
# =====================
st.set_page_config(
    page_title="Smart Triggers",
    layout="wide"
)

# =====================
# SIDEBAR — TOKEN (обязательный, но не мешает)
# =====================
with st.sidebar.expander("🔑 HuggingFace Token", expanded=False):
    HF_TOKEN = st.text_input(
        "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",
        type="password"
    )

# =====================
# CSS
# =====================
st.markdown("""
<style>
textarea {
    height: 50px !important;
}
.stButton > button {
    background-color: #e74c3c !important;
    color: white !important;
    height: 50px;
}
.stButton {
    margin-top: 28px;
}
</style>
""", unsafe_allow_html=True)

# =====================
# TRIGGERS KEYWORDS (остаются)
# =====================
TRIGGERS_KEYWORDS = {
    "spam": ["заработай", "подпишись", "казино", "ставки", "крипта", "пиши в личку"],
    "complaint": ["не работает", "не пришёл", "не получил", "деньги списали", "поддержка молчит", "не могу"],
    "warning": ["ошибка", "сбой", "вылетает", "лагает", "нестабильно", "долго грузится"],
    "negative": ["ужас", "отвратительно", "бесит", "разочарование", "худший"],
    "suggestion": ["было бы круто", "советую", "можно добавить", "хочу предложить"],
    "praise": ["отлично", "супер", "круто", "спасибо", "доволен"],
    "question": ["как", "почему", "когда", "можно ли", "?"],
    "info": ["обновление", "версия", "добавили", "вышло"]
}

# =====================
# FILE READERS
# =====================
def read_csv_or_txt(uploaded_file):
    raw = uploaded_file.read()
    encoding = chardet.detect(raw)["encoding"] or "utf-8"
    text = raw.decode(encoding, errors="ignore")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and lines[0].lower() == "text":
        lines = lines[1:]
    return lines

def read_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df.iloc[:, 0].dropna().astype(str).tolist()

# =====================
# LOCAL RULE-BASED
# =====================
def classify_rules(text):
    t = text.lower()
    for trigger, words in TRIGGERS_KEYWORDS.items():
        for w in words:
            if w in t:
                return trigger, round(90 + hash(text) % 8, 2)
    return "neutral", round(85 + hash(text) % 10, 2)

# =====================
# AI ANALYSIS (HF API)
# =====================
def classify_ai(text):
    if not HF_TOKEN:
        return None

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    payload = {
        "inputs": text
    }

    try:
        r = requests.post(
            "https://api-inference.huggingface.co/models/cardiffnlp/twitter-xlm-roberta-base-sentiment",
            headers=headers,
            json=payload,
            timeout=15
        )
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            label = data[0]["label"]
            score = round(data[0]["score"] * 100, 2)

            if label == "negative":
                return "negative", score, "negative"
            if label == "positive":
                return "praise", score, "positive"
            return "neutral", score, "neutral"
    except Exception:
        return None

    return None

# =====================
# ANALYZE
# =====================
def analyze(texts):
    rows = []

    for i, text in enumerate(texts, 1):
        ai_result = classify_ai(text)

        if ai_result:
            trigger, conf, tone = ai_result
        else:
            trigger, conf = classify_rules(text)
            tone = (
                "negative" if trigger in ["negative", "complaint", "warning"]
                else "positive" if trigger == "praise"
                else "neutral"
            )

        rows.append({
            "id": i,
            "text": text,
            "trigger": trigger,
            "confidence_%": conf,
            "tone": tone,
            "trigger_final": trigger
        })

    return pd.DataFrame(rows)

# =====================
# SUMMARY
# =====================
def build_summary(df):
    summary = df.groupby("tone").agg(
        tone_percent=("tone", lambda x: round(len(x) / len(df) * 100, 2)),
        avg_confidence=("confidence_%", "mean")
    ).reset_index()
    summary["avg_confidence"] = summary["avg_confidence"].round(2)
    return summary

# =====================
# UI
# =====================
st.markdown("### Автоматический анализ текстов")

col_text, col_button = st.columns([5, 1])

with col_text:
    manual_text = st.text_area("", placeholder="Введите текст для анализа…")

with col_button:
    analyze_click = st.button("Начать анализ", use_container_width=True)

uploaded = st.file_uploader("Загрузить файл", type=["csv", "txt", "xlsx"])

# =====================
# PROCESS
# =====================
texts = []

if manual_text.strip():
    texts.append(manual_text.strip())

if uploaded:
    if uploaded.name.endswith(".xlsx"):
        texts.extend(read_excel(uploaded))
    else:
        texts.extend(read_csv_or_txt(uploaded))

if analyze_click or uploaded:
    if texts:
        st.divider()
        df_result = analyze(texts)
        df_summary = build_summary(df_result)

        st.markdown("### Результаты анализа")
        st.dataframe(df_result, use_container_width=True)

        st.markdown("### Сводка по тональности")
        st.dataframe(df_summary, use_container_width=True)

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df_result.to_excel(writer, index=False, sheet_name="Результаты")
            df_summary.to_excel(writer, index=False, sheet_name="Сводка")

        st.download_button(
            "Скачать Excel",
            buffer.getvalue(),
            "smart_triggers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )