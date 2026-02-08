{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8b3af1ad-04aa-4381-a90a-b607137f5f13",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-02-08 06:45:34.689 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\ProgramData\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "\n",
    "st.set_page_config(page_title=\"Smart Triggers\", layout=\"wide\")\n",
    "st.title(\"Smart Triggers\")\n",
    "st.write(\"Автоматический анализ текстов по умным триггерам\")\n",
    "\n",
    "TRIGGERS = {\n",
    "    \"negative\": [\"надоел\", \"ужас\", \"плохо\", \"ненавижу\", \"достало\", \"бесит\"],\n",
    "    \"complaint\": [\"парковка\", \"дорога\", \"проблема\", \"не работает\", \"сломалось\"],\n",
    "    \"spam\": [\"подпишись\", \"заработок\", \"доход\", \"крипта\", \"казино\"],\n",
    "    \"political\": [\"мэр\", \"власть\", \"правительство\", \"выборы\"]\n",
    "}\n",
    "\n",
    "def analyze_texts(texts):\n",
    "    rows = []\n",
    "    for i, text in enumerate(texts, start=1):\n",
    "        text_l = text.lower()\n",
    "        matched = [\n",
    "            trigger for trigger, words in TRIGGERS.items()\n",
    "            if any(word in text_l for word in words)\n",
    "        ]\n",
    "        rows.append({\n",
    "            \"id\": i,\n",
    "            \"text\": text,\n",
    "            \"triggers\": \", \".join(matched) if matched else \"none\",\n",
    "            \"final_label\": final_label(matched)\n",
    "        })\n",
    "    return pd.DataFrame(rows)\n",
    "\n",
    "def final_label(triggers):\n",
    "    if \"spam\" in triggers:\n",
    "        return \"spam\"\n",
    "    if \"complaint\" in triggers or \"negative\" in triggers:\n",
    "        return \"complaint\"\n",
    "    if \"political\" in triggers:\n",
    "        return \"political\"\n",
    "    return \"neutral\"\n",
    "\n",
    "uploaded_file = st.file_uploader(\"Загрузите CSV с колонкой text\", type=[\"csv\"])\n",
    "\n",
    "if uploaded_file:\n",
    "    df_input = pd.read_csv(uploaded_file)\n",
    "    texts = df_input[\"text\"].astype(str).tolist()\n",
    "\n",
    "    if st.button(\"Анализировать\"):\n",
    "        result = analyze_texts(texts)\n",
    "        st.dataframe(result)\n",
    "\n",
    "        csv = result.to_csv(index=False, encoding=\"utf-8-sig\")\n",
    "        st.download_button(\n",
    "            \"Скачать результат CSV\",\n",
    "            csv,\n",
    "            \"smart_triggers_result.csv\",\n",
    "            \"text/csv\"\n",
    "        )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e431ccd6-54ef-4433-a2d0-77263390a42c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
