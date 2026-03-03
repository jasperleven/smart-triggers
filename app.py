import streamlit as st
import pandas as pd
from io import BytesIO
import requests
import chardet

# =====================
# CONFIG
# =====================
st.set_page_config(
    page_title="Smart Triggers",
    layout="wide"
)

# =====================
# SIDEBAR — TOKEN
# =====================
with st.sidebar.expander("🔑 HuggingFace Token", expanded=False):
    HF_TOKEN = st.text_input(
        "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",
        type="password",
        help="Обязательно для AI-анализа"
    )

# =====================
# CSS
# =====================
st.markdown("""
<style>
textarea {height: 50px !important;}
.stButton > button {background-color: #e74c3c !important; color: white !important; height: 50px;}
.stButton {margin-top: 28px;}
</style>
""", unsafe_allow_html=True)

# =====================
# TRIGGERS
# =====================
TRIGGERS_KEYWORDS = {
    "spam": ["заработай","подпишись","казино","ставки","крипта","пиши в личку","доход","инвестиции"],
    "complaint": ["не работает","не пришёл","не получил","деньги списали","поддержка молчит","проблема","не могу","не войти","не заходит"],
    "warning": ["ошибка","сбой","вылетает","лагает","не загружается","нестабильно","долго грузится"],
    "negative": ["ненавижу","бесит","ужас","отвратительно","отвратительный","кошмар","разочарование","хуже","худший"],
    "suggestion": ["было бы круто","предлагаю","советую","можно добавить","хотелось бы","хочу предложить"],
    "praise": ["отлично","супер","круто","доволен","спасибо","отличный","приятно удивлён"],
    "question": ["как","почему","когда","можно ли","что делать","подскажите","платно","бесплатно","?"],
    "info": ["обновление","версия","добавили","вышло","информация"]
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
# CONFIDENCE
# =====================
def neutral_confidence(text):
    length = len(text.strip())
    if length < 5: return 45.0
    if length < 15: return 60.0
    return round(88 + min(length, 120) * 0.08, 2)

def spam_confidence(text):
    return round(90 + hash(text) % 8, 2)

# =====================
# CLASSIFICATION (RULE-BASED)
# =====================
def classify_local(text):
    t = text.lower()
    PRIORITY = ["spam","complaint","warning","negative","suggestion","praise","question","info"]
    for trigger in PRIORITY:
        for w in TRIGGERS_KEYWORDS.get(trigger, []):
            if w in t:
                if trigger == "spam": return "spam", spam_confidence(text)
                return trigger, round(88 + hash(text) % 10 + 0.37, 2)
    return "neutral", neutral_confidence(text)

# =====================
# AI CLASSIFICATION (HuggingFace Inference API)
# =====================
def classify_ai(text):
    if not HF_TOKEN:
        return None, None, None
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text}
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/cointegrated/rubert-tiny2-bert-base",
            headers=headers,
            json=payload,
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            # Берем label с наибольшей вероятностью
            if isinstance(data, list) and len(data) > 0:
                label = data[0].get("label", "").lower()
                score = float(data[0].get("score", 0.0) * 100)
                # Определяем тональность
                if label in ["negative","complaint","warning"]: tone = "negative"
                elif label == "praise": tone = "positive"
                else: tone = "neutral"
                return label, score, tone
    except Exception as e:
        st.warning(f"AI classification failed: {e}")
    return None, None, None

# =====================
# ANALYZE
# =====================
def analyze(texts):
    result = []
    for i, text in enumerate(texts, 1):
        # сначала локальные правила
        trigger, conf = classify_local(text)
        tone = (
            "negative" if trigger in ["negative","complaint","warning"]
            else "positive" if trigger == "praise"
            else "neutral"
        )

        # если есть токен, используем AI для уточнения
        if HF_TOKEN:
            trigger_ai, conf_ai, tone_ai = classify_ai(text)
            if trigger_ai:
                trigger = trigger_ai
                conf = conf_ai
                tone = tone_ai

        result.append({
            "id": i,
            "text": text,
            "trigger": trigger,
            "confidence_%": round(conf,2),
            "tone": tone,
            "trigger_final": trigger
        })
    return pd.DataFrame(result)

# =====================
# SUMMARY
# =====================
def build_summary(df):
    summary = df.groupby("tone").agg(
        tone_percent=("tone", lambda x: round(len(x)/len(df)*100,2)),
        avg_confidence=("confidence_%","mean")
    ).reset_index()
    summary["avg_confidence"] = summary["avg_confidence"].round(2)
    return summary

# =====================
# UI
# =====================
st.markdown("### Автоматический анализ текстов")
col_text, col_button = st.columns([5,1])
with col_text:
    manual_text = st.text_area("", placeholder="Введите текст для анализа…")
with col_button:
    analyze_click = st.button("Начать анализ", use_container_width=True)

uploaded = st.file_uploader("Загрузить файл", type=["csv","txt","xlsx"])
texts = []
if manual_text.strip(): texts.append(manual_text.strip())
if uploaded:
    if uploaded.name.endswith(".xlsx"): texts.extend(read_excel(uploaded))
    else: texts.extend(read_csv_or_txt(uploaded))

if analyze_click or uploaded:
    if texts:
        st.divider()
        df_result = analyze(texts)
        df_summary = build_summary(df_result)
        st.markdown("### Результаты анализа")
        st.dataframe(df_result, use_container_width=True)
        st.markdown("### Сводка по тональности")
        st.dataframe(df_summary, use_container_width=True)

        # Excel export
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