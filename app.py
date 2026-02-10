import streamlit as st
import pandas as pd
import json

# --- 1. Список триггеров ---
TRIGGERS = [
    "negative",
    "complaint",
    "spam",
    "positive",
    "suggestion",
    "question",
    "news_info",
    "discussion",
    "irony_sarcasm"
]

# --- 2. Prompt для AI (для будущей интеграции с реальным LLM) ---
PROMPT_TEMPLATE = """
You are a text classification AI.

Classify the Russian text into ONE main trigger from the list:

negative
complaint
spam
positive
suggestion
question
news_info
discussion
irony_sarcasm

Return JSON:
{{
  "main_trigger": "<trigger>",
  "confidence": <0-1>
}}

Text:
"{text}"
"""

# --- 3. Заглушка AI функции (MVP) ---
def ai_classify(text):
    text_lower = text.lower()

    # 1. Negative
    if any(w in text_lower for w in ["надоел", "ужас", "плохо", "ненавижу", "достало", "бесит", "отвратительно", "кошмар"]):
        return "negative", 0.9

    # 2. Complaint
    elif any(w in text_lower for w in ["парковка", "дорога", "проблема", "не работает", "сломалось", "очередь"]):
        return "complaint", 0.9

    # 3. Spam
    elif any(w in text_lower for w in ["подпишись", "заработок", "доход", "крипта", "казино", "ставки"]):
        return "spam", 0.9

    # 4. Positive
    elif any(w in text_lower for w in ["отлично", "спасибо", "круто", "супер"]):
        return "positive", 0.9

    # 5. Suggestion
    elif any(w in text_lower for w in ["можно было бы", "предлагаю", "рекомендую"]):
        return "suggestion", 0.8

    # 6. Question
    elif "?" in text_lower or any(w in text_lower for w in ["почему", "как", "когда", "что"]):
        return "question", 0.8

    # 7. News / Info
    elif any(w in text_lower for w in ["новость", "событие", "сообщение"]):
        return "news_info", 0.7

    # 8. Discussion
    else:
        return "discussion", 0.6

# --- 4. Streamlit UI ---
st.title("Smart Triggers — AI классификация текста")
st.write("Загрузка текста или CSV для определения триггеров")

uploaded_file = st.file_uploader("Выберите CSV файл с колонкой 'text'", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("CSV должен содержать колонку 'text'")
    else:
        results = []
        for idx, row in df.iterrows():
            text = row["text"]
            main_trigger, confidence = ai_classify(text)
            results.append({
                "id": idx + 1,
                "text": text,
                "main_trigger": main_trigger,
                "confidence": round(confidence * 100, 2)
            })
        df_result = pd.DataFrame(results)
        st.write(df_result)

        # Сохраняем CSV с utf-8-sig чтобы русские символы отображались корректно
        csv = df_result.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="Скачать результат",
            data=csv,
            file_name="smart_triggers_result.csv",
            mime="text/csv"
        )

st.write("Пример текста:")
st.write("`надоела эта парковка` → main_trigger: complaint, confidence: 90%")
st.write("`подпишись и заработай на крипте` → main_trigger: spam, confidence: 90%")
st.write("`мэр снова ничего не сделал` → main_trigger: discussion, confidence: 60%")
st.write("`всё работает отлично` → main_trigger: positive, confidence: 90%")

