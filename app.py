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
CONFIDENCE_THRESHOLD = 55.0

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
# TRIGGERS
# =====================
TRIGGERS_KEYWORDS = {
    "complaint": [
        "не работает", "проблема", "не пришёл", "не получил",
        "поддержка молчит", "деньги списали", "не могу"
    ],
    "warning": [
        "ошибка", "сбой", "вылетает", "не загружается", "лагает"
    ],
    "negative": [
        "ненавижу", "бесит", "ужас", "отвратительно", "достало",
        "хуже", "худший", "разочарование", "кошмар", "невозможно"
    ],
    "question": [
        "как", "почему", "когда", "можно ли", "что делать",
        "платно", "бесплатно"
    ],
    "suggestion": [
        "было бы круто", "предлагаю", "советую", "можно добавить",
        "хотелось бы"
    ],
    "praise": [
        "отлично", "супер", "круто", "хорошо", "доволен",
        "спасибо", "приятно удивлён"
    ],
    "info": [
        "обновление", "новая версия", "информация", "новости",
        "вышло", "добавили"
    ]
}

TRIGGER_PRIORITY = [
    "complaint",
    "warning",
    "negative",
    "question",
    "suggestion",
    "praise",
    "info"
]

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
    col = df.iloc[:, 0]
    col = col.dropna()
    col = col.astype(str)
    col = col[col.str.strip() != ""]
    return col.tolist()

# =====================
# CLASSIFICATION
# =====================
def classify_local(text):
    t = text.lower()
    matches = []

    for trigger, words in TRIGGERS_KEYWORDS.items():
        count = sum(1 for w in words if w in t)
        if count > 0:
            matches.append((trigger, count))

    if not matches:
        return None, None

    matches.sort(
        key=lambda x: (
            TRIGGER_PRIORITY.index(x[0]),
            -x[1]
        )
    )

    trigger, count = matches[0]

    if count == 1:
        confidence = 78.0
    elif count == 2:
        confidence = 86.0
    elif count == 3:
        confidence = 91.0
    else:
        confidence = 96.0

    return trigger, confidence


def classify_ai(text):
    if not HF_TOKEN:
        return None, None

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
        return None, None

# =====================
# ANALYZE
# =====================
def analyze(texts):
    rows = []

    for i, text in enumerate(texts, 1):
        trigger, conf = classify_local(text)

        if not trigger:
            trigger, conf = classify_ai(text)

        if not trigger:
            trigger = "neutral"
            conf = 40.0

        if conf < CONFIDENCE_THRESHOLD:
            trigger_final = "neutral"
        else:
            trigger_final = trigger

        if trigger in ["complaint", "negative", "warning"]:
            tone = "negative"
        elif trigger == "praise":
            tone = "positive"
        else:
            tone = "neutral"

        rows.append({
            "id": i,
            "text": text,
            "trigger": trigger,
            "confidence_%": conf,
            "tone": tone,
            "trigger_final": trigger_final
        })

    return pd.DataFrame(rows)

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
    label="Загрузить файл",
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

        avg_confidence = round(df_result["confidence_%"].mean(), 2)
        st.markdown(f"**Средний confidence по файлу:** {avg_confidence}%")

        st.markdown("### Результаты анализа")
        st.dataframe(df_result, use_container_width=True)

        # =====================
        # Excel Export
        # =====================
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            # Лист 1: основные результаты, только 6 колонок
            df_result[['id', 'text', 'trigger', 'confidence_%', 'tone', 'trigger_final']] \
                .to_excel(writer, index=False, sheet_name="Результаты")

            # Лист 2: сводка по tone
            summary = df_result.groupby('tone')['confidence_%'].mean().reset_index()
            summary.rename(columns={'confidence_%': 'avg_confidence'}, inplace=True)
            summary.to_excel(writer, index=False, sheet_name="Сводка")

        st.download_button(
            "Скачать Excel",
            excel_buffer.getvalue(),
            "smart_triggers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
