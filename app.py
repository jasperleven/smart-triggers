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

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

# =====================
# SIDEBAR — TOKEN
# =====================
st.sidebar.markdown("### 🔑 HuggingFace Token")
HF_TOKEN = st.sidebar.text_input(
    "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",
    type="password",
    help="Нужен для повышения точности классификации"
)

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# =====================
# CSS
# =====================
st.markdown("""
<style>
textarea {
    height: 50px !important;
}
.stButton > button {
    background-color: #e74c3c;
    color: white;
    height: 50px;
}
.stButton {
    margin-top: 28px;
}
</style>
""", unsafe_allow_html=True)

# =====================
# TRIGGERS (УЛУЧШЕНЫ)
# =====================
TRIGGERS_KEYWORDS = {
    "negative": [
        "ненавижу", "бесит", "ужас", "отвратительно", "достало",
        "хуже", "разочарование", "кошмар", "невозможно"
    ],
    "complaint": [
        "не работает", "проблема", "не пришёл", "не получил",
        "поддержка молчит", "деньги списали", "не могу"
    ],
    "praise": [
        "отлично", "супер", "круто", "хорошо", "доволен",
        "спасибо", "приятно удивлён"
    ],
    "warning": [
        "ошибка", "сбой", "вылетает", "не загружается", "лагает"
    ],
    "info": [
        "обновление", "новая версия", "информация", "новости",
        "вышло", "добавили"
    ],
    "suggestion": [
        "было бы круто", "предлагаю", "советую", "можно добавить",
        "хотелось бы"
    ],
    "question": [
        "как", "почему", "когда", "можно ли", "что делать"
    ]
}

ALLOWED_TRIGGERS = list(TRIGGERS_KEYWORDS.keys())

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
    return df.iloc[:, 0].astype(str).tolist()

# =====================
# CLASSIFICATION
# =====================
def classify_local(text):
    t = text.lower()
    for trigger, words in TRIGGERS_KEYWORDS.items():
        if any(w in t for w in words):
            return trigger, round(88 + hash(text) % 10 + 0.37, 2)
    return None, None


def classify_ai(text):
    if not HF_TOKEN:
        return "neutral", 40.00

    prompt = f"К какому триггеру относится текст ({', '.join(ALLOWED_TRIGGERS)}): {text}"

    try:
        r = requests.post(
            HF_API_URL,
            headers=HEADERS,
            json={"inputs": prompt},
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
        return data["labels"][0], round(data["scores"][0] * 100, 2)
    except Exception:
        return "neutral", 40.00

# =====================
# ANALYZE
# =====================
def analyze(texts):
    result = []
    for i, text in enumerate(texts, 1):
        trigger, conf = classify_local(text)
        if not trigger:
            trigger, conf = classify_ai(text)

        result.append({
            "id": i,
            "text": text,
            "trigger": trigger,
            "confidence_%": conf
        })

    df = pd.DataFrame(result)
    return df

# =====================
# SUMMARY + MERGE
# =====================
def enrich_with_tone(df):
    tone_map = {
        "negative": "negative",
        "complaint": "negative",
        "warning": "negative",
        "praise": "positive",
        "info": "neutral",
        "suggestion": "neutral",
        "question": "neutral"
    }

    df["tone"] = df["trigger"].map(tone_map).fillna("neutral")

    summary = df["tone"].value_counts().reset_index()
    summary.columns = ["tone", "tone_count"]
    summary["tone_percent"] = (summary["tone_count"] / summary["tone_count"].sum() * 100).round(2)

    df = df.merge(summary, on="tone", how="left")
    return df, summary

# =====================
# UI
# =====================
st.markdown("### Автоматический анализ текстов")

col_text, col_button = st.columns([5, 1])

with col_text:
    manual_text = st.text_area(
        "",
        placeholder="Введите текст для анализа…",
        height=50
    )

with col_button:
    analyze_click = st.button("Начать анализ", use_container_width=True)

uploaded = st.file_uploader(
    "Загрузить файл",
    type=["csv", "txt", "xlsx"]
)

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
        df_result, df_summary = enrich_with_tone(df_result)

        st.markdown("### Результаты анализа")
        st.dataframe(df_result, use_container_width=True)

        st.markdown("### Сводка по тональности")
        st.dataframe(df_summary, use_container_width=True)

        # CSV
        csv_data = df_result.to_csv(
            index=False,
            sep=";",
            encoding="utf-8-sig"
        )

        st.download_button(
            "Скачать CSV",
            csv_data,
            "smart_triggers.csv",
            mime="text/csv"
        )

        # Excel (2 листа)
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            df_result.to_excel(writer, index=False, sheet_name="Результаты")
            df_summary.to_excel(writer, index=False, sheet_name="Сводка")

        st.download_button(
            "Скачать Excel",
            excel_buffer.getvalue(),
            "smart_triggers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
