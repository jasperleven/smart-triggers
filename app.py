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
HF_TOKEN = os.getenv("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")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# =====================
# STYLES
# =====================
st.markdown("""
<style>
.main .block-container {
    max-width: 900px;
    padding-top: 2rem;
}

div[data-testid="column"] {
    padding-top: 0;
}

button[kind="primary"] {
    height: 38px;
    margin-top: 22px;
}
</style>
""", unsafe_allow_html=True)

# =====================
# TRIGGERS
# =====================
TRIGGERS_KEYWORDS = {
    "negative": ["ненавижу", "достало", "бесит", "ужас"],
    "complaint": ["проблема", "не работает", "сломалось", "очередь"],
    "spam": ["подпишись", "заработай", "крипта", "ставки", "казино"],
    "praise": ["отлично", "супер", "круто", "хорошо"],
    "service": ["обслуживание", "поддержка", "сервис", "доставка"],
    "feature": ["функция", "опция", "интерфейс"],
    "warning": ["ошибка", "сбой"],
    "info": ["информация", "новости", "обновление"],
    "suggestion": ["предложение", "идея", "совет"]
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
            return trig, 90
    return None, None

def classify_ai(text):
    if not HF_TOKEN:
        return "neutral", 40

    prompt = (
        f"К какому из триггеров относится текст: {', '.join(ALLOWED_TRIGGERS)}?\n"
        f"Текст: {text}"
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
        score = int(result["scores"][0] * 100)
        if label in ALLOWED_TRIGGERS:
            return label, score
    except Exception:
        pass

    return "neutral", 40

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
# HERO
# =====================
st.markdown("### Автоматический анализ текстов")
st.caption("Определение триггеров и уровня уверенности анализа")

# =====================
# INPUT (INLINE)
# =====================
st.markdown("#### Введите текст для анализа")

col_text, col_btn = st.columns([8, 2], gap="small")

with col_text:
    manual_text = st.text_input(
        "",
        placeholder="Введите текст для анализа"
    )

with col_btn:
    analyze_click = st.button(
        "Начать анализ",
        use_container_width=True,
        type="primary"
    )

uploaded = st.file_uploader(
    "Или загрузите CSV / TXT",
    type=["csv", "txt"],
    label_visibility="collapsed"
)

st.caption("Загрузка файла для массового анализа")

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
    except Exception as e:
        st.error(f"Ошибка файла: {e}")

if analyze_click and texts:
    st.divider()
    st.markdown("### Результаты анализа")

    df_result = analyze(texts)
    st.dataframe(df_result, use_container_width=True)

    csv_bytes = df_result.to_csv(
        index=False,
        sep=";",
        encoding="utf-8-sig"
    ).encode("utf-8-sig")

    st.download_button(
        "Скачать CSV",
        csv_bytes,
        "smart_triggers_result.csv",
        "text/csv"
    )
